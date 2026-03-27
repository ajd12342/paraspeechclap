import datetime
import os
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import yaml
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
from sklearn.metrics import recall_score, f1_score, accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import List, Dict, Optional, Set

from paraspeechclap.dataset import ParaSpeechCapsDataset
from paraspeechclap.model import CLAP
from paraspeechclap.debug_utils import (
    logger, set_log_level
)
from paraspeechclap.evaluation_utils import get_model, CLASSIFICATION_TEMPLATE
from paraspeechclap.utils import collate_fn # Import the unified collate function


def evaluate_classification(
    cfg: DictConfig,
    model: CLAP,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Performs classification evaluation.

    Compares audio embeddings against text embeddings generated from templated
    candidate style prompts (e.g., "A person is speaking in a {} style.") to predict
    the class label (style) for each audio sample.

    Args:
        cfg: Hydra DictConfig containing evaluation parameters. Expected fields:
             - data.dataset_name, data.audio_root, data.split
             - gold_label_column: Name of the dataset column with true labels.
             - candidate_styles: List of strings representing the possible classes.
             - num_workers (optional)
             - meta.tqdm_disable (optional)
             - save_predictions (optional, boolean)
             - save_confusion_matrix (optional, boolean)
             - save_logits (optional, boolean): Save prediction probabilities for analysis
             - meta.results (optional, directory to save results)
        model: The CLAP model instance (already loaded and on `device`).
        tokenizer: The tokenizer instance compatible with the model's text branch.
        device: The torch device to use for computation.

    Returns:
        A dictionary containing classification metrics (Accuracy, UAR, Weighted_F1).
        Empty if evaluation fails.

    Raises:
        ValueError: If required configuration is missing, invalid, or if dataset
                    structure is incorrect (e.g., missing gold label column).
        FileNotFoundError: If dataset resources are not found.
        RuntimeError: If issues occur during processing (tokenization, embedding,
                      metric calculation, saving results).
    """
    logger.info('--- Starting Classification Evaluation ---')

    # --- Configuration Validation and Setup ---
    dataset_name: str = cfg.data.dataset_name
    audio_root: str = cfg.data.audio_root
    eval_split: str = cfg.data.split
    gold_label_column: str = cfg.gold_label_column # Required
    # Required, ensure it's a list
    candidate_styles_cfg = cfg.candidate_styles
    if not isinstance(candidate_styles_cfg, (list, ListConfig)):
        error_msg = f"'candidate_styles' must be a list in Classification config, found {type(candidate_styles_cfg)}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    candidate_styles: List[str] = OmegaConf.to_object(candidate_styles_cfg)
    if not candidate_styles:
        error_msg = "'candidate_styles' list cannot be empty for Classification evaluation."
        logger.error(error_msg)
        raise ValueError(error_msg)
    candidate_style_set: Set[str] = set(candidate_styles)

    # Optional config values
    num_workers: int = cfg.get("num_workers", 0)
    tqdm_disable: bool = cfg.meta.get("tqdm_disable", False)
    save_predictions: bool = cfg.get("save_predictions", False)
    save_conf_matrix: bool = cfg.get("save_confusion_matrix", False)
    save_logits: bool = cfg.get("save_logits", False)
    results_dir: Optional[str] = cfg.meta.get("results")
    eval_batch_size: int = cfg.data.get("eval_batch_size", 1)  # New: Batch size for Classification
    sort_by_duration: bool = cfg.data.get("sort_by_duration", False) # New: Flag for sorting in dataset

    logger.info(f"Dataset: {dataset_name}, Split: {eval_split}, Audio Root: {audio_root}")
    logger.info(f"Gold Label Column: '{gold_label_column}'")
    logger.info(f"Candidate Styles ({len(candidate_styles)}): {candidate_styles}")
    logger.info(f"Save predictions: {save_predictions}, Save confusion matrix: {save_conf_matrix}, Save logits: {save_logits}")

    # Validate split
    try:
        available_splits = get_dataset_split_names(dataset_name)
        if eval_split not in available_splits:
            error_msg = f"Configured split '{eval_split}' not found in '{dataset_name}'. Available: {available_splits}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Using evaluation split: '{eval_split}'")
    except Exception as e:
        logger.error(f"Could not retrieve splits for dataset '{dataset_name}'. Error: {e}")
        raise ValueError(f"Failed to get splits for '{dataset_name}'") from e

    # --- Dataset Loading (Requires labels) --- #
    logger.info(f"Loading dataset for Classification using gold label column: '{gold_label_column}'")
    try:
        eval_dataset = ParaSpeechCapsDataset(
            speech_model_name=cfg.models.speech,
            split=eval_split,
            dataset_name=dataset_name,
            audio_root=audio_root,
            gold_label_column=gold_label_column, # Critical for Classification
            sort_by_duration=sort_by_duration # Pass the sorting flag
        )
        logger.info(f"Loaded Classification dataset. Initial sample count: {len(eval_dataset)}")

        # --- Validate Gold Label Column Exists in Loaded Data --- #
        if gold_label_column not in eval_dataset.dataset.features:
            available_cols = list(eval_dataset.dataset.features.keys())
            error_msg = (f"Specified 'gold_label_column' '{gold_label_column}' not found "
                         f"in the loaded dataset '{dataset_name}' features. Available columns: {available_cols}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Confirmed gold label column '{gold_label_column}' exists in dataset.")

    except ValueError as ve:
        raise ve # Re-raise specific ValueErrors (like missing column)
    except Exception as e:
        logger.error(f"Failed to load ParaSpeechCapsDataset '{dataset_name}' with split '{eval_split}'. Error: {e}")
        raise RuntimeError(f"Dataset loading failed for {dataset_name}/{eval_split}") from e

    # --- Validate Dataset Labels Against Candidate Styles --- #
    logger.info(f"Validating gold labels in dataset against configured candidate styles...")
    try:
        unique_labels_in_dataset: Set[str] = set(eval_dataset.dataset[gold_label_column])
        logger.info(f"Found {len(unique_labels_in_dataset)} unique labels in dataset column '{gold_label_column}'.")
        logger.debug(f"Unique labels found: {sorted(list(unique_labels_in_dataset))}")

        # Debug: Show some example raw dataset items to verify label format
        logger.debug("--- Debug: Sample dataset items ---")
        for i in range(min(3, len(eval_dataset.dataset))):
            item = eval_dataset.dataset[i]
            raw_label = item.get(gold_label_column, 'MISSING')
            logger.debug(f"Sample {i}: {gold_label_column} = {raw_label}")
            if isinstance(raw_label, str) and raw_label.strip():
                # Check if raw label matches candidate styles (case sensitivity check)
                if raw_label.strip() not in candidate_style_set:
                    logger.warning(f"⚠️  Label '{raw_label.strip()}' from sample {i} not found in candidate_styles!")
                else:
                    logger.debug(f"✅ Label '{raw_label}' found in candidates")

        missing_labels = unique_labels_in_dataset - candidate_style_set
        if missing_labels:
            error_msg = (
                f"WARNING: The following gold labels found in the dataset are MISSING from the configured 'candidate_styles': "
                f"{sorted(list(missing_labels))}. Samples with these labels will likely be misclassified and metrics affected. "
                f"Consider updating 'candidate_styles' in the config if these labels should be included."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
             logger.info("All unique labels found in the dataset are present in the candidate styles list.")

    except Exception as e:
        # This might happen if the column exists but contains non-hashable types, etc.
        logger.error(f"Error extracting or processing unique labels from column '{gold_label_column}': {e}")
        raise RuntimeError(f"Failed to validate labels in column '{gold_label_column}'") from e

    # --- Dataloader --- #
    # Batch size for Classification can now be > 1. Audio embeddings will be batched.
    # The comparison against candidate_text_emb will also be batched.
    loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size, # Use configured batch size
        shuffle=False, # Keep shuffle False for Classification consistency
        num_workers=num_workers,
        collate_fn=collate_fn # Use unified collate_fn from utils
    )
    logger.info(f"DataLoader created: batch_size={eval_batch_size}, shuffle=False, num_workers={num_workers}, using utils.collate_fn. Dataset sorted by duration: {sort_by_duration}")

    # --- Candidate Prompt Setup --- #
    logger.info(f"Generating and tokenizing text prompts for {len(candidate_styles)} candidate styles...")
    try:
        templated_candidates = [CLASSIFICATION_TEMPLATE.format(style) for style in candidate_styles]
        logger.debug(f"Templated candidate prompts: {templated_candidates}")
        candidate_tokens = tokenizer.batch_encode_plus(
            templated_candidates,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        logger.info(f"Tokenized {len(templated_candidates)} candidate prompts.")
    except Exception as e:
        logger.error(f"Failed to tokenize candidate style prompts. Error: {e}")
        raise RuntimeError("Candidate prompt tokenization failed.") from e

    # --- Pre-compute Text Embeddings for Candidate Styles --- #
    logger.info("Pre-computing text embeddings for candidate styles...")
    try:
        with torch.no_grad():
            # Use the new method to get normalized text embeddings for candidates
            candidate_text_emb = model.get_text_embedding(candidate_tokens, normalize=True)
        logger.debug(f"Candidate text embeddings - Shape: {candidate_text_emb.shape}, Dtype: {candidate_text_emb.dtype}, Device: {candidate_text_emb.device}")
        logger.info(f"Computed candidate text embeddings with shape: {candidate_text_emb.shape}")
        
        # Debug: Check if text embeddings look reasonable (not all zeros/ones/same values)
        text_emb_stats = {
            'mean': candidate_text_emb.mean().item(),
            'std': candidate_text_emb.std().item(),
            'min': candidate_text_emb.min().item(),
            'max': candidate_text_emb.max().item()
        }
        logger.info(f"Text embedding stats: {text_emb_stats}")
        if text_emb_stats['std'] < 1e-6:
            logger.warning("⚠️  Text embeddings have very low variance - model might not be properly loaded!")
            
    except Exception as e:
        logger.error(f"Failed to compute text embeddings for candidate styles. Error: {e}")
        raise RuntimeError("Candidate text embedding computation failed.") from e

    # --- Evaluation Loop --- #
    model.eval() # Ensure model is in eval mode

    logger.info(f"Starting Classification evaluation loop for {len(loader)} batches...")

    # Phase 1: Data Loading (CPU-bound)
    all_audio_batches_cpu: List[torch.Tensor] = []
    all_audio_attention_mask_batches_cpu: List[torch.Tensor] = []
    all_gold_labels_batches_cpu: List[List[str]] = [] # List of lists of strings

    logger.info(f"Starting Phase 1: Data loading for Classification for {len(loader)} batches...")
    for i, batch in enumerate(tqdm.tqdm(
            loader,
            total=len(loader),
            desc="Phase 1: Classification Data Loading",
            disable=tqdm_disable
    )):
        # --- Input Validation (from collate_fn and dataset) ---
        if 'audio' not in batch or 'audio_attention_mask' not in batch:
            raise ValueError(f"Batch {i} is missing 'audio' or 'audio_attention_mask'. Check collate_fn and dataset.")
        if 'label' not in batch:
            # Labels are essential for Classification evaluation.
            raise ValueError(f"Batch {i} is missing required key 'label' (from column '{gold_label_column}'). Cannot perform Classification.")

        # --- Collect Batch Data (CPU side) ---
        audio_cpu_batch: torch.Tensor = batch['audio']
        audio_attention_mask_cpu_batch: torch.Tensor = batch['audio_attention_mask']
        gold_labels_in_batch: List[str] = batch['label'] # This is a list of strings from collate_fn

        logger.debug(f"Phase 1 Classification Batch {i} - audio_cpu - Shape: {audio_cpu_batch.shape}")
        logger.debug(f"Phase 1 Classification Batch {i} - audio_attention_mask_cpu - Shape: {audio_attention_mask_cpu_batch.shape}")
        logger.debug(f"Phase 1 Classification Batch {i} - gold_labels: {gold_labels_in_batch}")

        all_audio_batches_cpu.append(audio_cpu_batch)
        all_audio_attention_mask_batches_cpu.append(audio_attention_mask_cpu_batch)
        all_gold_labels_batches_cpu.append(gold_labels_in_batch)
    
    logger.info("Phase 1: Classification Data loading complete.")

    # --- Phase 2: Batched Audio Embedding and Classification Prediction (GPU-bound) ---
    all_gold_labels_flat: List[str] = []
    all_predictions_flat: List[str] = []
    
    # Storage for logits/probabilities (if save_logits is enabled)
    all_logits_data = [] if save_logits else None  # List of dicts with logits and metadata

    if not all_audio_batches_cpu:
        logger.warning("No data collected in Phase 1 for Classification. Skipping Phase 2 (embedding extraction and prediction).")
    else:
        logger.info(f"Starting Phase 2: Batched audio embedding and Classification prediction for {len(all_audio_batches_cpu)} preprocessed batches...")
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(len(all_audio_batches_cpu)),
                desc="Phase 2: Classification Inference",
                disable=tqdm_disable
            ):
                current_audio_cpu_batch = all_audio_batches_cpu[i]
                current_audio_attention_mask_cpu_batch = all_audio_attention_mask_batches_cpu[i]
                current_gold_labels_batch = all_gold_labels_batches_cpu[i]

                # Move current batch to device
                audio_gpu_batch = current_audio_cpu_batch.to(device, non_blocking=True)
                audio_attention_mask_gpu_batch = current_audio_attention_mask_cpu_batch.to(device, non_blocking=True)

                logger.debug(f"Phase 2 Classification Batch {i} - audio_gpu - Shape: {audio_gpu_batch.shape}")
                logger.debug(f"Phase 2 Classification Batch {i} - audio_attention_mask_gpu - Shape: {audio_attention_mask_gpu_batch.shape}")

                try:
                    # Get normalized audio embedding for the batch
                    audio_emb_batch = model.get_audio_embedding(
                        audio_gpu_batch, 
                        attention_mask=audio_attention_mask_gpu_batch, 
                        normalize=True
                    )
                    logger.debug(f"Audio embedding batch {i} - Shape: {audio_emb_batch.shape}")

                    # Compute similarity: (B, dim) @ (dim, num_candidates) -> (B, num_candidates)
                    similarity_batch = audio_emb_batch @ candidate_text_emb.T # candidate_text_emb is already on device
                    logger.debug(f"Similarity matrix (Audio x Styles) batch {i} - Shape: {similarity_batch.shape}")

                    probabilities_batch = torch.softmax(similarity_batch, dim=1)  # Shape: (B, num_candidates)
                    
                    # Get prediction indices for the batch
                    prediction_indices_batch = similarity_batch.argmax(dim=1) # Shape: (B,)
                    
                    # Map indices back to candidate style strings for each item in the batch
                    predictions_batch = [candidate_styles[idx.item()] for idx in prediction_indices_batch]
                    logger.debug(f"Batch {i} predictions: {predictions_batch}")

                    # Store logits data if requested
                    if save_logits:
                        probabilities_np = probabilities_batch.cpu().numpy()  # Shape: (B, num_candidates)
                        for sample_idx in range(probabilities_np.shape[0]):
                            # Global sample index across all batches
                            global_sample_idx = sum(len(batch) for batch in all_gold_labels_batches_cpu[:i]) + sample_idx
                            
                            # Store logits with metadata
                            logits_entry = {
                                'sample_idx': global_sample_idx,
                                'batch_idx': i,
                                'batch_sample_idx': sample_idx,
                                'probabilities': probabilities_np[sample_idx],  # Shape: (num_candidates,)
                                'gold_label': current_gold_labels_batch[sample_idx],
                                'predicted_label': predictions_batch[sample_idx],
                                'candidate_labels': candidate_styles  # Reference for probability indices
                            }
                            all_logits_data.append(logits_entry)

                    all_gold_labels_flat.extend(current_gold_labels_batch)
                    all_predictions_flat.extend(predictions_batch)

                except Exception as e:
                    # Log details of the batch that failed if possible
                    # current_gold_labels_batch provides context for which samples might have failed
                    logger.error(f"Error during Classification processing for preprocessed batch {i}. Gold Labels in batch: '{current_gold_labels_batch}'. Error: {e}")
                    raise RuntimeError(f"Classification processing failed for preprocessed batch {i}") from e
        logger.info("Phase 2: Classification inference complete.")

    # --- Log Prediction Statistics ---
    if all_predictions_flat and all_gold_labels_flat:
        from collections import Counter
        pred_counts = Counter(all_predictions_flat)
        gold_counts = Counter(all_gold_labels_flat)
        
        logger.info(f"--- Prediction Statistics (Single-label) ---")
        logger.info(f"Total samples processed: {len(all_predictions_flat)}")
        logger.info(f"Unique labels predicted: {len(pred_counts)}")
        logger.info(f"Unique gold labels: {len(gold_counts)}")
        logger.info(f"Most common predictions: {pred_counts.most_common(5)}")
        logger.info(f"Most common gold labels: {gold_counts.most_common(5)}")
        
        # Check for potential issues
        if len(pred_counts) == 1:
            logger.warning(f"⚠️  All predictions are the same label: '{list(pred_counts.keys())[0]}' - possible model issue!")
        
        # Check coverage
        predicted_labels = set(all_predictions_flat)
        gold_labels = set(all_gold_labels_flat)
        unpredicted_gold = gold_labels - predicted_labels
        if unpredicted_gold:
            logger.warning(f"⚠️  Gold labels never predicted: {sorted(unpredicted_gold)}")

    # --- Compute Metrics --- #
    logger.info("--- Calculating Classification Metrics ---")
    results: Dict[str, float] = {}
    if not all_gold_labels_flat or not all_predictions_flat:
        logger.warning("No gold labels or predictions were collected. Cannot calculate Classification metrics. Check data loading and evaluation loop.")
        return results
    if len(all_gold_labels_flat) != len(all_predictions_flat):
        # This case should ideally be prevented by the loop structure, but check just in case.
        logger.error(f"CRITICAL: Mismatch between collected gold labels ({len(all_gold_labels_flat)}) and predictions ({len(all_predictions_flat)}). Cannot calculate metrics reliably.")
        raise ValueError("Mismatch between gold labels and predictions. Check data loading and evaluation loop.")

    num_eval_samples = len(all_gold_labels_flat)
    logger.info(f"Calculating metrics on {num_eval_samples} samples.")

    try:
        # Use sklearn for standard metrics
        results["Classification_Accuracy"] = accuracy_score(all_gold_labels_flat, all_predictions_flat) * 100 # As percentage

        # Define the full set of labels for consistent UAR/F1 calculation
        # Use the candidate styles as the universe of possible labels for macro averages
        uar_f1_labels = sorted(list(candidate_style_set))
        logger.info(f"Calculating UAR and Macro F1 using label set: {uar_f1_labels}")

        results["Classification_UAR"] = recall_score(all_gold_labels_flat, all_predictions_flat, average='macro', labels=uar_f1_labels, zero_division=0) * 100 # Unweighted Average Recall
        results["Classification_Macro_F1"] = f1_score(all_gold_labels_flat, all_predictions_flat, average='macro', labels=uar_f1_labels, zero_division=0) * 100

        logger.info("--- Classification Results ---")
        # Format results nicely for logging
        results_formatted = {k: f"{v:.2f}%" for k, v in results.items()}
        logger.info(yaml.dump(results_formatted, sort_keys=False))

    except Exception as e:
        logger.error(f"Failed to calculate sklearn metrics. Error: {e}")
        logger.exception("Traceback:")
        # Return partial or empty results? Returning empty for now.
        return {}

    # --- Optional: Save logits/probabilities for analysis ---
    if save_logits and all_logits_data:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save as pickle for full data preservation
            pickle_path = os.path.join(results_dir, f"prediction_logits_singlelabel.pkl")
            try:
                import pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump(all_logits_data, f)
                logger.info(f"Prediction logits saved to pickle: {pickle_path}")
                logger.info(f"Pickle contains {len(all_logits_data)} samples with full probability distributions")
            except Exception as e:
                logger.error(f"Failed to save logits pickle to {pickle_path}: {e}")
            
            # Save as CSV for easy analysis/visualization
            csv_path = os.path.join(results_dir, f"prediction_logits_singlelabel.csv")
            try:
                # Create a flattened CSV with one row per sample-label combination
                csv_rows = []
                for entry in all_logits_data:
                    sample_idx = entry['sample_idx']
                    gold_label = entry['gold_label']
                    pred_label = entry['predicted_label']
                    
                    # Add row for each label with its probability
                    for label_idx, label in enumerate(entry['candidate_labels']):
                        probability = float(entry['probabilities'][label_idx])
                        is_gold = (label == gold_label)
                        is_predicted = (label == pred_label)
                        
                        csv_rows.append({
                            'sample_idx': sample_idx,
                            'label': label,
                            'probability': probability,
                            'is_gold_label': is_gold,
                            'is_predicted': is_predicted,
                            'gold_label': gold_label,
                            'predicted_label': pred_label
                        })
                
                logits_df = pd.DataFrame(csv_rows)
                logits_df.to_csv(csv_path, index=False)
                logger.info(f"Prediction logits CSV saved: {csv_path}")
                logger.info(f"CSV contains {len(csv_rows)} label-probability pairs for analysis")
                
                # Log some distribution statistics
                import numpy as np
                probs = logits_df['probability'].values
                logger.info(f"Probability distribution stats:")
                logger.info(f"  Mean: {np.mean(probs):.4f}, Std: {np.std(probs):.4f}")
                logger.info(f"  Min: {np.min(probs):.4f}, Max: {np.max(probs):.4f}")
                logger.info(f"  25th percentile: {np.percentile(probs, 25):.4f}")
                logger.info(f"  50th percentile (median): {np.percentile(probs, 50):.4f}")
                logger.info(f"  75th percentile: {np.percentile(probs, 75):.4f}")
                logger.info(f"  90th percentile: {np.percentile(probs, 90):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to save logits CSV to {csv_path}: {e}")
        else:
            logger.warning("`save_logits` is True, but `meta.results` directory is not specified in config. Cannot save logits.")
    elif save_logits and not all_logits_data:
        logger.warning("`save_logits` is True, but no logits data was collected.")

    # --- Optional: Save predictions --- #
    if save_predictions:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            # Sanitize dataset_name to prevent interpreting slashes as directory separators in the filename
            # sanitized_dataset_name_for_filename = dataset_name.replace('/', '_') # No longer needed for filename
            # save_path = os.path.join(results_dir, f"predictions_classification_{sanitized_dataset_name_for_filename}_{eval_split}.csv")
            save_path = os.path.join(results_dir, f"predictions_classification.csv")
            try:
                pred_df = pd.DataFrame({'prediction': all_predictions_flat, 'target': all_gold_labels_flat})
                pred_df.to_csv(save_path, index=False)
                logger.info(f"Classification predictions saved to: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save Classification predictions to {save_path}: {e}")
        else:
            logger.warning("`save_predictions` is True, but `meta.results` directory is not specified in config. Cannot save predictions.")

    # --- Optional: Save Confusion Matrix --- #
    if save_conf_matrix:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            # Sanitize dataset_name to prevent interpreting slashes as directory separators in the filename
            # sanitized_dataset_name_for_filename = dataset_name.replace('/', '_') # No longer needed for filename
            # save_path = os.path.join(results_dir, f"confusion_matrix_classification_{sanitized_dataset_name_for_filename}_{eval_split}.csv")
            save_path = os.path.join(results_dir, f"confusion_matrix_classification.csv")
            try:
                # Use the same label set as UAR/F1 for consistent ordering
                cm = pd.crosstab(pd.Categorical(all_gold_labels_flat, categories=uar_f1_labels, ordered=True),
                                 pd.Categorical(all_predictions_flat, categories=uar_f1_labels, ordered=True),
                                 rownames=['True Label'], colnames=['Predicted Label'],
                                 dropna=False)
                cm.to_csv(save_path)
                logger.info(f"Classification confusion matrix saved to: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save Classification confusion matrix to {save_path}: {e}")
        else:
            logger.warning("`save_confusion_matrix` is True, but `meta.results` directory is not specified in config. Cannot save matrix.")

    logger.info("--- Classification Evaluation Complete ---")
    return results

@hydra.main(config_path="../configs", config_name="eval/classification/base", version_base=None)
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for Classification evaluation script using Hydra."""
    # Set TOKENIZERS_PARALLELISM to false to avoid warnings with DataLoader workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    run_start_time = datetime.datetime.now()
    logger.info(f"Starting Classification evaluation script run at {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Log the configuration used for this run
    logger.info("--- Configuration ---")
    # Use OmegaConf.to_yaml for clean logging
    try:
        config_yaml = OmegaConf.to_yaml(cfg)
        logger.info(f"\n{config_yaml}")
    except Exception as e:
        logger.error(f"Could not serialize config to YAML: {e}. Raw config: {cfg}")
    logger.info("--------------------")

    # --- Setup --- #
    log_level_str = cfg.meta.get("log_level", "INFO") # Get log_level string from config
    set_log_level(log_level_str) # Set log level based on the string
    
    if cfg.meta.get("device"): # User-specified device in config
        device = torch.device(cfg.meta.device)
    else: # Auto-detect
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If CUDA is chosen, explicitly set the device if it's cuda:X
    # and update the device variable to the specific cuda device.
    if device.type == 'cuda':
        if device.index is not None: # e.g. "cuda:1"
            torch.cuda.set_device(device)
        elif torch.cuda.is_available(): # e.g. "cuda" and GPUs are available
             torch.cuda.set_device(0) # Default to cuda:0
             device = torch.device('cuda:0') # Update device to be specific

    logger.info(f"Using device: {device}")

    try:
        # Aligned tokenizer loading with evaluate_retrieval.py
        # Removed: if not cfg.models or not cfg.models.text: raise ValueError(...)
        # Removed: tokenizer_name = cfg.models.text
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
        logger.info(f"Tokenizer loaded successfully: {cfg.models.text}")
    # Removed: except ValueError as ve: logger.error(f"Configuration Error: {ve}"); raise
    except Exception as e: # General exception, like retrieval
        logger.error(f"Failed to load tokenizer '{cfg.models.text}'. Error: {e}") # Match retrieval error message
        raise RuntimeError("Tokenizer loading failed.") from e # Match retrieval error message

    # --- Load Model --- #
    try:
        model = get_model(cfg, device) # get_model handles its own logging and errors
    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Failed to load the model. Exiting. Error: {e}")
         return # Exit if model loading fails

    # --- Run Classification Evaluation --- #
    cls_results: Dict[str, float] = {}
    try:
        # Check required evaluation config keys before calling evaluate_classification
        required_eval_keys = ['data.dataset_name', 'data.split', 'gold_label_column', 'candidate_styles']
        missing_keys = [key for key in required_eval_keys if not OmegaConf.select(cfg, key, default=None)]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys for Classification evaluation: {missing_keys}")

        cls_results = evaluate_classification(cfg, model, tokenizer, device)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
         # Catch potential errors from evaluate_classification itself or config checks
         logger.error(f"An error occurred during the Classification evaluation process: {e}")
         logger.exception("Traceback (most recent call last):") # Log full traceback for debugging
         # Optionally: exit or indicate failure state
    except Exception as e:
        logger.error(f"An unexpected error occurred during Classification evaluation: {e}")
        logger.exception("Traceback (most recent call last):")

    # --- Final Report --- #
    logger.info("--- Final Classification Evaluation Metrics ---")
    if cls_results:
        # Pretty print the results using yaml dump for better readability
        try:
            results_yaml = yaml.dump(cls_results, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"\n{results_yaml}")
        except Exception as e:
            logger.error(f"Could not format results as YAML: {e}. Raw results: {cls_results}")
            # Log raw results as fallback
            for key, value in cls_results.items():
                logger.info(f"{key}: {value}") # Assumes values are simple floats
    else:
        logger.warning("Classification evaluation did not produce any results (or failed).")

    if cls_results and cfg.meta.get("results"):
        results_dir = cfg.meta.results
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "final_classification_metrics.yaml")
        try:
            with open(save_path, 'w') as f:
                yaml.dump(cls_results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"Final Classification metrics saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save final Classification metrics to {save_path}: {e}")
    elif not cfg.meta.get("results"):
        logger.warning("No `meta.results` directory specified in config. Final Classification metrics not saved to disk.")

    run_end_time = datetime.datetime.now()
    logger.info(f"Classification evaluation script finished at {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {run_end_time - run_start_time}")


if __name__ == "__main__":
    evaluate_main()
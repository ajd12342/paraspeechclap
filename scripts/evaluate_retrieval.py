import os
import torch
import tqdm
import yaml
import numpy as np
from transformers import AutoTokenizer
from datasets import get_dataset_split_names
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Optional
import datetime
import json

from paraspeechclap.dataset import ParaSpeechCapsDataset
from paraspeechclap.model import CLAP
from paraspeechclap.debug_utils import (
    logger, set_log_level
)
from paraspeechclap.evaluation_utils import get_model, calculate_audio_to_text_retrieval_metrics
from paraspeechclap.utils import collate_fn # Import the unified collate function

def evaluate_retrieval(
    cfg: DictConfig,
    model: CLAP,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Performs Cross-Modal Retrieval evaluation (Standard R@k, MedR and optional Label Recall).

    Args:
        cfg: Hydra DictConfig containing evaluation parameters. Expected fields:
             - data.dataset_name, data.audio_root, data.split
             - num_workers (optional, for DataLoader)
             - meta.tqdm_disable (optional, to disable progress bar)
             - save_similarity_matrix (optional, boolean)
             - meta.results (optional, directory to save matrix)
        model: The CLAP model instance (already loaded and on `device`).
        tokenizer: The tokenizer instance compatible with the model's text branch.
        device: The torch device to use for computation.

    Returns:
        A dictionary containing retrieval metrics. Empty if evaluation fails.

    Raises:
        ValueError: If required configuration is missing or invalid dataset split is provided.
        FileNotFoundError: If dataset resources (e.g., audio root) are not found (via ParaSpeechCapsDataset).
        RuntimeError: If inconsistencies are found during embedding extraction (e.g., mismatched counts).
    """
    logger.info("--- Starting Cross-Modal Retrieval Evaluation ---")

    # --- Configuration and Setup ---
    eval_split: str = cfg.data.split
    dataset_name: str = cfg.data.dataset_name
    audio_root: str = cfg.data.audio_root
    # Optional config values with defaults
    num_workers: int = cfg.get("num_workers", 0) # Default to 0 workers
    tqdm_disable: bool = cfg.meta.get("tqdm_disable", False) # Default to enabled
    save_similarity: bool = cfg.get("save_similarity_matrix", False)
    results_dir: Optional[str] = cfg.meta.get("results")
    eval_batch_size: int = cfg.data.get("eval_batch_size", 1)
    sort_by_duration: bool = cfg.data.get("sort_by_duration", False) # New: Flag for sorting in dataset
    # New config flags for saving top-K captions
    save_top_k_captions_flag: bool = cfg.get("save_top_k_captions", False)
    top_k_value_for_retrieval: int = cfg.get("top_k_value", 10)
    # Validate split
    try:
        available_splits = get_dataset_split_names(dataset_name)
    except Exception as e:
        logger.error(f"Could not retrieve splits for dataset '{dataset_name}'. Check name/availability. Error: {e}")
        raise ValueError(f"Failed to get splits for '{dataset_name}'") from e

    if eval_split not in available_splits:
        error_msg = f"Configured split '{eval_split}' not found in dataset '{dataset_name}'. Available splits: {available_splits}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Using evaluation split: '{eval_split}'")

    # --- Dataset Loading --- #
    logger.info(f"Loading dataset '{dataset_name}' ({eval_split})")

    try:
        eval_dataset = ParaSpeechCapsDataset(
            speech_model_name=cfg.models.speech,
            split=eval_split,
            dataset_name=dataset_name,
            audio_root=audio_root,
            sort_by_duration=sort_by_duration # Pass the sorting flag
        )
        logger.info(f"Successfully loaded dataset. Initial sample count: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load ParaSpeechCapsDataset '{dataset_name}' with split '{eval_split}'. Error: {e}")
        raise RuntimeError(f"Dataset loading failed for {dataset_name}/{eval_split}") from e

    # --- Subsample Dataset (Optional) --- #
    num_samples = len(eval_dataset)
    logger.info(f"Using full dataset split with {num_samples} samples.")

    # --- Dataloader --- #
    # The dataset might now be pre-sorted by duration if sort_by_duration was True.
    # The custom_collate_fn will handle padding for variable-length audio in batches.
    loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size, # Use configured batch_size
        shuffle=False, # Shuffle must be False for retrieval to maintain order (even if sorted by duration)
        num_workers=num_workers,
        collate_fn=collate_fn, # Use the unified collate function from utils
        # pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"DataLoader created: batch_size={eval_batch_size}, shuffle=False, num_workers={num_workers}, using utils.collate_fn. Dataset sorted by duration: {sort_by_duration}")

    # --- Phase 1: Data loading and CPU-bound preprocessing (e.g., text tokenization) --- #
    all_audio_batches_cpu: List[torch.Tensor] = []
    all_audio_attention_mask_batches_cpu: List[torch.Tensor] = []
    all_text_str_batches_for_top_k: List[List[str]] = [] # For storing original text strings per batch

    logger.info(f"Starting Phase 1: Data loading and audio/text preprocessing for {len(loader)} batches...")
    for i, batch in enumerate(tqdm.tqdm(
            loader,
            total=len(loader),
            desc="Phase 1: Data Loading & Audio Processing",
            disable=tqdm_disable
    )):
        # --- Input Validation ---
        if 'audio' not in batch:
            raise ValueError(f"Batch {i} is missing required key 'audio'. Check dataset structure.")
        if 'text' not in batch:
             raise ValueError(f"Batch {i} is missing required key 'text'. Check dataset structure.")
        # --- Prepare Audio Batch (CPU side) ---
        audio_cpu_batch: torch.Tensor = batch['audio'] # Already processed by ParaSpeechCapsDataset's __getitem__
        audio_attention_mask_cpu_batch: torch.Tensor = batch['audio_attention_mask']
        text_str_batch: List[str] = batch['text'] # List of strings

        logger.debug(f"Phase 1 Batch {i} - audio_cpu_batch - Shape: {audio_cpu_batch.shape}, Dtype: {audio_cpu_batch.dtype}")
        logger.debug(f"Phase 1 Batch {i} - audio_attention_mask_cpu_batch - Shape: {audio_attention_mask_cpu_batch.shape}")
        logger.debug(f"Phase 1 Batch {i} - text_str_batch: {text_str_batch}")

        # Store original text captions for reference
        all_text_str_batches_for_top_k.append(text_str_batch)

        # Store audio batches (always needed)
        all_audio_batches_cpu.append(audio_cpu_batch)
        all_audio_attention_mask_batches_cpu.append(audio_attention_mask_cpu_batch)

    logger.info("Phase 1: Data loading and audio preprocessing complete.")
    all_original_text_captions_flat = [caption for batch_texts in all_text_str_batches_for_top_k for caption in batch_texts]
    logger.info(f"Collected {len(all_original_text_captions_flat)} original text captions.")

    # --- Text Processing: Deduplication and Tokenization --- #
    all_text_token_batches_cpu: List[Dict[str, torch.Tensor]] = []

    logger.info("--- Text Processing: Unique Text Extraction and Tokenization ---")
    # Create mapping from original text to unique text indices
    unique_texts = []
    text_to_unique_idx = {}
    audio_to_unique_text_mapping = []

    for i, text in enumerate(all_original_text_captions_flat):
        if text not in text_to_unique_idx:
            text_to_unique_idx[text] = len(unique_texts)
            unique_texts.append(text)
        audio_to_unique_text_mapping.append(text_to_unique_idx[text])

    logger.info(f"Original texts: {len(all_original_text_captions_flat)}, Unique texts: {len(unique_texts)}")
    logger.info(f"Unique ratio: {len(unique_texts)}/{len(all_original_text_captions_flat)} = {len(unique_texts)/len(all_original_text_captions_flat):.3f}")

    # Create batches of unique texts for embedding extraction
    unique_text_batches = []
    for i in range(0, len(unique_texts), eval_batch_size):
        batch_texts = unique_texts[i:i + eval_batch_size]
        unique_text_batches.append(batch_texts)

    logger.info(f"Created {len(unique_text_batches)} batches for {len(unique_texts)} unique texts (batch_size={eval_batch_size})")

    # Tokenize unique text batches
    for i, batch_texts in enumerate(unique_text_batches):
        try:
            tokenized_batch = tokenizer.batch_encode_plus(
                batch_texts,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            all_text_token_batches_cpu.append(tokenized_batch)
            logger.debug(f"Tokenized unique text batch {i}/{len(unique_text_batches)} with {len(batch_texts)} texts")
        except Exception as e:
            logger.error(f"Failed to tokenize unique text batch {i}: {e}")
            raise RuntimeError(f"Unique text tokenization failed for batch {i}") from e

    logger.info("Text processing and tokenization complete.")

    # --- Phase 2: Batched Embedding Extraction (GPU-bound) ---
    all_audio_emb_list: List[torch.Tensor] = []
    all_text_emb_list: List[torch.Tensor] = []
    
    model.eval() # Ensure model is in evaluation mode

    if not all_audio_batches_cpu: # Handle empty dataset after Phase 1
        logger.warning("No data collected in Phase 1. Skipping Phase 2 (embedding extraction).")
    else:
        logger.info(f"Starting Phase 2: Batched model inference...")
        logger.info(f"Audio batches: {len(all_audio_batches_cpu)}, Text batches: {len(all_text_token_batches_cpu)}")
        
        # Extract audio embeddings
        logger.info("Extracting audio embeddings...")
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(len(all_audio_batches_cpu)),
                desc="Phase 2: Audio Embedding Extraction",
                disable=tqdm_disable
            ):
                current_audio_cpu_batch = all_audio_batches_cpu[i]
                current_audio_attention_mask_cpu_batch = all_audio_attention_mask_batches_cpu[i]

                # Move current batch to device
                audio_gpu_batch = current_audio_cpu_batch.to(device)
                audio_attention_mask_gpu_batch = current_audio_attention_mask_cpu_batch.to(device)
                
                logger.debug(f"Phase 2 Audio Batch {i} - Shape: {audio_gpu_batch.shape}, Device: {audio_gpu_batch.device}")

                try:
                    # Get normalized audio embedding
                    audio_emb_batch = model.get_audio_embedding(
                        audio_gpu_batch, 
                        attention_mask=audio_attention_mask_gpu_batch, 
                        normalize=True
                    )
                    logger.debug(f"Audio embedding batch {i} - Shape: {audio_emb_batch.shape}")

                    # Store embeddings (move to CPU to free GPU memory)
                    all_audio_emb_list.append(audio_emb_batch.cpu())

                except Exception as e:
                    logger.error(f"Audio embedding computation failed for batch {i}. Error: {e}")
                    raise RuntimeError(f"Audio embedding computation failed for batch {i}") from e
        
        # Extract text embeddings  
        logger.info("Extracting text embeddings...")
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(len(all_text_token_batches_cpu)),
                desc="Phase 2: Text Embedding Extraction",
                disable=tqdm_disable
            ):
                current_text_tokens_cpu_batch = all_text_token_batches_cpu[i]
                text_tokens_gpu_batch = {k: v.to(device) for k, v in current_text_tokens_cpu_batch.items()}
                
                try:
                    # Get normalized text embedding
                    text_emb_batch = model.get_text_embedding(text_tokens_gpu_batch, normalize=True)
                    logger.debug(f"Text embedding batch {i} - Shape: {text_emb_batch.shape}")

                    # Store embeddings (move to CPU to free GPU memory)
                    all_text_emb_list.append(text_emb_batch.cpu())

                except Exception as e:
                    logger.error(f"Text embedding computation failed for batch {i}. Error: {e}")
                    raise RuntimeError(f"Text embedding computation failed for batch {i}") from e
        
        logger.info("Phase 2: Batched model inference complete.")

    # --- Post-processing and Sanity Checks --- #
    if not all_audio_emb_list: # Handles empty dataset or all batches failing in Phase 2
        logger.error("Embedding extraction resulted in zero embeddings. Check dataset and processing loops.")
        return {}

    # Concatenate embeddings: list of (B, dim) -> (N, dim)
    try:
        all_audio_emb_tensor = torch.cat(all_audio_emb_list, dim=0)
        all_text_emb_tensor = torch.cat(all_text_emb_list, dim=0)
        logger.info(f"Collected audio embeddings shape: {all_audio_emb_tensor.shape}")
        logger.info(f"Collected text embeddings shape: {all_text_emb_tensor.shape}")
    except Exception as e:
        logger.error(f"Failed to concatenate embeddings: {e}")
        raise RuntimeError("Error during embedding concatenation.") from e

    # --- Critical Alignment Assertions ---
    num_extracted = all_audio_emb_tensor.shape[0]
    logger.info(f"{num_extracted} audio samples, {all_text_emb_tensor.shape[0]} unique text embeddings")

    if save_top_k_captions_flag:
        if len(all_original_text_captions_flat) != num_extracted:
            error_msg = (
                f"CRITICAL ERROR: Mismatch in number of audio embeddings ({num_extracted}) and "
                f"collected original text captions ({len(all_original_text_captions_flat)}). "
                f"Cannot save top-K captions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    # Final check for zero embeddings after all processing
    if num_extracted == 0:
        error_msg = "Zero embeddings available after processing. Cannot calculate any retrieval metrics."
        logger.error(error_msg)
        raise RuntimeError(error_msg) # Fail explicitly if no embeddings exist

    # --- Calculate Similarity Matrix --- #
    logger.info("Calculating full similarity matrix...")
    # Move embeddings back to the target device for efficient matrix multiplication
    all_audio_emb_tensor = all_audio_emb_tensor.to(device)
    all_text_emb_tensor = all_text_emb_tensor.to(device)
    try:
        with torch.no_grad():
            # similarity: (N_audio, N_unique_text)
            similarity_matrix = all_audio_emb_tensor @ all_text_emb_tensor.T
        logger.debug(f"Full Similarity Matrix (Audio x Text) - Shape: {similarity_matrix.shape}, Dtype: {similarity_matrix.dtype}, Device: {similarity_matrix.device}")
        logger.info(f"Similarity matrix calculated with shape: {similarity_matrix.shape}")
    except Exception as e:
        logger.error(f"Failed to calculate similarity matrix: {e}")
        raise RuntimeError("Similarity matrix calculation failed.") from e

    # --- Calculate Retrieval Metrics --- #
    logger.info("Calculating audio-to-text retrieval metrics...")
    retrieval_results = calculate_audio_to_text_retrieval_metrics(
        similarity_matrix.cpu(), audio_to_unique_text_mapping
    )
    if not retrieval_results:
        logger.warning("Audio-to-text retrieval metric calculation returned empty results.")

    # --- Optional: Save Similarity Matrix --- #
    if save_similarity:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            sim_matrix_save_path = os.path.join(results_dir, "similarity_matrix_retrieval.pt")
            try:
                torch.save(similarity_matrix.cpu(), sim_matrix_save_path)
                logger.info(f"Retrieval similarity matrix saved to: {sim_matrix_save_path}")
            except Exception as e:
                logger.error(f"Failed to save similarity matrix to {sim_matrix_save_path}: {e}")
        else:
            logger.warning("`save_similarity_matrix` is True, but `meta.results` directory is not specified in config. Cannot save matrix.")

    # --- Optional: Save Top-K Retrieved Captions --- #
    if save_top_k_captions_flag:
        if results_dir:
            if num_extracted > 0: # Ensure there's data to process
                logger.info(f"Preparing top-{top_k_value_for_retrieval} retrieved captions for each of the {num_extracted} audio samples...")
                output_data_top_k = []

                for i in range(num_extracted): # For each audio sample
                    ground_truth_caption = all_original_text_captions_flat[i]

                    audio_i_similarity_scores = similarity_matrix[i, :] # Scores for current audio against all unique texts

                    # Determine actual k for topk (cannot exceed number of unique texts)
                    current_k = min(top_k_value_for_retrieval, len(unique_texts))

                    top_k_scores, top_k_indices = torch.topk(audio_i_similarity_scores, k=current_k)

                    retrieved_captions_with_scores = []
                    for score, idx in zip(top_k_scores.cpu().tolist(), top_k_indices.cpu().tolist()):
                        retrieved_captions_with_scores.append({
                            "caption": unique_texts[idx],
                            "score": float(score)
                        })

                    output_data_top_k.append({
                        "audio_idx": i,
                        "ground_truth_caption": ground_truth_caption,
                        "retrieved_top_k": retrieved_captions_with_scores
                    })

                top_k_captions_save_path = os.path.join(results_dir, f"top_{top_k_value_for_retrieval}_retrieved_captions.json")
                try:
                    os.makedirs(results_dir, exist_ok=True) # Ensure directory exists
                    with open(top_k_captions_save_path, 'w') as f:
                        json.dump(output_data_top_k, f, indent=2)
                    logger.info(f"Top-K retrieved captions saved to: {top_k_captions_save_path}")
                except Exception as e:
                    logger.error(f"Failed to save top-K retrieved captions to {top_k_captions_save_path}: {e}")
            else:
                logger.warning("`save_top_k_captions` is True, but no embeddings were extracted. Skipping saving top-K captions.")
        else:
            logger.warning("`save_top_k_captions` is True, but `meta.results` directory is not specified in config. Cannot save top-K captions.")

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    logger.info("--- Cross-Modal Retrieval Evaluation Complete ---")
    return retrieval_results

@hydra.main(config_path="../configs", config_name="eval/retrieval", version_base=None)
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for Retrieval evaluation script using Hydra."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_start_time = datetime.datetime.now()
    logger.info(f"Starting Retrieval evaluation script run at {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Log the configuration used for this run
    logger.info("--- Configuration ---")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
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
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
        logger.info(f"Tokenizer loaded successfully: {cfg.models.text}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{cfg.models.text}'. Error: {e}")
        raise RuntimeError("Tokenizer loading failed.") from e

    # --- Load Model --- #
    try:
        model = get_model(cfg, device) # get_model handles its own logging and errors
    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Failed to load the model. Exiting. Error: {e}")
         return # Exit if model loading fails

    # --- Run Retrieval Evaluation --- #
    retrieval_results: Dict[str, float] = {}
    try:
        retrieval_results = evaluate_retrieval(cfg, model, tokenizer, device)
    except Exception as e:
         # Catch potential errors from evaluate_retrieval itself
         logger.error(f"An error occurred during the retrieval evaluation process: {e}")
         logger.exception("Traceback:") # Log full traceback for debugging

    # --- Final Report --- #
    logger.info("--- Final Retrieval Evaluation Metrics ---")
    if retrieval_results:
        # Pretty print the results using yaml dump for better readability
        try:
            results_yaml = yaml.dump(retrieval_results, sort_keys=False, allow_unicode=True)
            logger.info(f"\n{results_yaml}")
        except Exception as e:
            logger.error(f"Could not format results as YAML: {e}. Raw results: {retrieval_results}")
            # Log raw results as fallback
            for key, value in retrieval_results.items():
                logger.info(f"{key}: {value}")
    else:
        logger.warning("Retrieval evaluation did not produce any results (or failed).")

    # --- Save Final Results to Disk --- #
    if retrieval_results and cfg.meta.get("results"):
        results_dir = cfg.meta.results
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"final_retrieval_metrics.yaml")
        try:
            with open(save_path, 'w') as f:
                yaml.dump(retrieval_results, f, sort_keys=False, allow_unicode=True)
            logger.info(f"Final retrieval metrics saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save final retrieval metrics to {save_path}: {e}")
    elif not cfg.meta.get("results"):
        logger.warning("No `meta.results` directory specified in config. Final metrics not saved to disk.")

    run_end_time = datetime.datetime.now()
    logger.info(f"Retrieval evaluation script finished at {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {run_end_time - run_start_time}")


if __name__ == "__main__":
    # Add datetime import for logging timestamps
    evaluate_main() 
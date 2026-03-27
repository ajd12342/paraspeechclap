import os
import torch
import torch.nn.functional as F
import tqdm
import yaml
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Optional, Any, Tuple
import datetime
import json
import torchaudio
import torchaudio.transforms as T
import shutil
from torch.utils.data import Dataset, DataLoader

from paraspeechclap.debug_utils import logger, set_log_level
from paraspeechclap.evaluation_utils import get_model
from paraspeechclap.model import CLAP

# --- Audio Preprocessing Function ---
def preprocess_audio(
    audio_path: str,
    feature_extractor: AutoFeatureExtractor,
    target_sr: int,
    device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Loads, resamples, converts to mono, and feature extracts a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        feature_extractor (AutoFeatureExtractor): Initialized Hugging Face feature extractor.
        target_sr (int): Target sampling rate.
        device (torch.device): Device to move tensors to (for attention_mask, features are kept on CPU for batching).

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: 
            - audio_features (torch.Tensor): Processed audio features (1, num_features, seq_len) or (seq_len) on CPU.
            - attention_mask (torch.Tensor): Attention mask for the audio (1, seq_len) on CPU.
            Returns (None, None) if processing fails.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise RuntimeError(f"Error loading audio file {audio_path}") from e

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono by averaging channels if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Squeeze to (L,) shape for feature_extractor
    waveform_squeezed = waveform.squeeze(0)

    try:
        # Process with FeatureExtractor (e.g., Wav2Vec2FeatureExtractor)
        processed_output = feature_extractor(
            waveform_squeezed,
            sampling_rate=target_sr,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest"
        )
        # Squeeze batch dim if feature extractor adds it for single input
        audio_features = processed_output.input_values.squeeze(0) # Keep on CPU for now
        attention_mask = processed_output.attention_mask.squeeze(0) # Keep on CPU for now
        
        return audio_features, attention_mask
    except Exception as e:
        logger.error(f"Error processing audio {audio_path} with feature_extractor: {e}")
        raise RuntimeError(f"Error processing audio {audio_path} with feature_extractor") from e


class BestOfNDataset(Dataset):
    """
    Dataset that loads and preprocesses audio files in parallel.
    Text tokenization is handled separately to avoid variable-length issues.
    """
    
    def __init__(
        self, 
        text_prompts: List[str],
        input_base_dir: str,
        num_iterations: int,
        feature_extractor: AutoFeatureExtractor,
        target_sr: int,
        wer_lines_dict: Dict[int, List[str]],
        transcription_lines_dict: Dict[int, List[str]]
    ):
        self.text_prompts = text_prompts
        self.input_base_dir = input_base_dir
        self.num_iterations = num_iterations
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.wer_lines_dict = wer_lines_dict
        self.transcription_lines_dict = transcription_lines_dict
    
    def __len__(self):
        return len(self.text_prompts)
    
    def __getitem__(self, prompt_idx: int) -> Dict[str, Any]:
        text_prompt = self.text_prompts[prompt_idx]
        
        # Load and process all candidate audios for this prompt
        candidate_audios = []
        candidate_iter_indices = []
        wer_lines = []
        transcription_lines = []
        
        for iter_idx in range(1, self.num_iterations + 1):
            audio_file_name = f"{prompt_idx}.wav"
            audio_file_path = os.path.join(
                self.input_base_dir, f"iter_{iter_idx}", "audios", audio_file_name
            )
            
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file {audio_file_path} not found for prompt {prompt_idx}, iter {iter_idx}.")
                raise FileNotFoundError(f"Audio file {audio_file_path} not found.")
            
            # Preprocess audio (this will be parallelized by DataLoader workers)
            try:
                audio_features, attention_mask = preprocess_audio(
                    audio_file_path, self.feature_extractor, self.target_sr, torch.device('cpu')
                )
                
                candidate_audios.append({
                    'features': audio_features,
                    'attention_mask': attention_mask,
                    'audio_path': audio_file_path
                })
                candidate_iter_indices.append(iter_idx)
                
            except Exception as e:
                logger.error(f"Failed to preprocess audio {audio_file_path}: {e}")
                raise
            
            # Get corresponding WER and transcription lines
            if prompt_idx < len(self.wer_lines_dict.get(iter_idx, [])):
                wer_lines.append(self.wer_lines_dict[iter_idx][prompt_idx])
            else:
                wer_lines.append(f"ERROR: WER line not found for prompt {prompt_idx}, iter {iter_idx}")
            
            if prompt_idx < len(self.transcription_lines_dict.get(iter_idx, [])):
                transcription_lines.append(self.transcription_lines_dict[iter_idx][prompt_idx])
            else:
                transcription_lines.append(f"ERROR: Transcription line not found for prompt {prompt_idx}, iter {iter_idx}")
        
        return {
            'prompt_idx': prompt_idx,
            'text_prompt': text_prompt,
            'candidate_audios': candidate_audios,
            'candidate_iter_indices': candidate_iter_indices,
            'wer_lines': wer_lines,
            'transcription_lines': transcription_lines
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that groups batch data without trying to stack variable-length tensors.
    """
    return {
        'prompt_indices': [item['prompt_idx'] for item in batch],
        'text_prompts': [item['text_prompt'] for item in batch],
        'candidate_audios_batch': [item['candidate_audios'] for item in batch],
        'candidate_iter_indices_batch': [item['candidate_iter_indices'] for item in batch],
        'wer_lines_batch': [item['wer_lines'] for item in batch],
        'transcription_lines_batch': [item['transcription_lines'] for item in batch]
    }


def tokenize_texts_batch(text_prompts: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of texts with proper padding to handle variable lengths.
    """
    tokenized_texts = tokenizer(
        text_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # Adjust based on your model's max length
    )
    return tokenized_texts


# --- Main Best-of-N Selection Logic ---
def select_best_of_n(cfg: DictConfig, model: CLAP, tokenizer: AutoTokenizer, feature_extractor: AutoFeatureExtractor, device: torch.device) -> None:
    logger.info("--- Starting Best-of-N CLAP Selection (Parallel Audio Loading) ---")

    input_base_dir = cfg.input_base_dir
    num_iterations = cfg.num_iterations
    output_dir_name = cfg.output_dir_name
    target_sr = cfg.audio_processing.target_sr

    output_base_dir = os.path.join(input_base_dir, output_dir_name)
    output_audios_dir = os.path.join(output_base_dir, "audios")

    if os.path.exists(output_base_dir):
        logger.warning(f"Output directory {output_base_dir} already exists. Consider removing it if a clean run is needed.")
    os.makedirs(output_audios_dir, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_base_dir}")

    # === PHASE 1: Data Loading and Dataset Creation ===
    logger.info("--- Phase 1: Data Loading and Dataset Creation ---")

    # Load text prompts
    iter1_dir = os.path.join(input_base_dir, "iter_1")
    if not os.path.exists(iter1_dir):
        logger.error(f"iter_1 directory not found at {iter1_dir}. Cannot load prompts.")
        raise FileNotFoundError(f"iter_1 directory not found at {iter1_dir}")
    
    input_descriptions_path = os.path.join(iter1_dir, "input_descriptions.txt")
    try:
        with open(input_descriptions_path, 'r') as f:
            text_prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(text_prompts)} text prompts from {input_descriptions_path}")
    except FileNotFoundError:
        logger.error(f"input_descriptions.txt not found in {iter1_dir}")
        raise
    if not text_prompts:
        logger.error("No text prompts loaded. Exiting.")
        return

    # Pre-load WER and Transcription Lines
    all_iters_wer_lines: Dict[int, List[str]] = {}
    all_iters_transcription_lines: Dict[int, List[str]] = {}
    logger.info("Pre-loading WER and transcription files...")
    for i in range(1, num_iterations + 1):
        iter_dir = os.path.join(input_base_dir, f"iter_{i}")
        wer_path = os.path.join(iter_dir, "wer_per_example.txt")
        trans_path = os.path.join(iter_dir, "output_transcriptions.txt")
        try:
            with open(wer_path, 'r') as f:
                all_iters_wer_lines[i] = [line.strip() for line in f]
            with open(trans_path, 'r') as f:
                all_iters_transcription_lines[i] = [line.strip() for line in f]
        except FileNotFoundError as e:
            logger.error(f"Missing wer_per_example.txt or output_transcriptions.txt in {iter_dir}: {e}")
            raise
    logger.info("WER and transcription files pre-loaded.")

    # Create dataset and dataloader with parallel audio loading
    dataset = BestOfNDataset(
        text_prompts=text_prompts,
        input_base_dir=input_base_dir,
        num_iterations=num_iterations,
        feature_extractor=feature_extractor,
        target_sr=target_sr,
        wer_lines_dict=all_iters_wer_lines,
        transcription_lines_dict=all_iters_transcription_lines
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.meta.get('batch_size', 4),
        shuffle=False,
        num_workers=cfg.meta.get('num_workers', 4),  # Enable parallel audio loading
        collate_fn=collate_fn,
        pin_memory=False  # Will handle GPU transfer manually
    )
    
    logger.info(f"Created dataset with {len(dataset)} prompts and DataLoader with batch_size={dataloader.batch_size}, num_workers={dataloader.num_workers}")
    logger.info("--- Phase 1: Parallel Audio Loading ---")

    # Load all batches with parallel audio preprocessing
    preprocessed_batches = []
    for batch in tqdm.tqdm(dataloader, desc="Loading with parallel audio preprocessing", disable=cfg.meta.tqdm_disable):
        preprocessed_batches.append(batch)
    
    logger.info(f"Loaded {len(preprocessed_batches)} batches with parallel audio preprocessing")
    logger.info("--- Phase 1 Complete ---")

    # === PHASE 2: Text Tokenization ===
    logger.info("--- Phase 2: Text Tokenization ---")
    
    # Add tokenized texts to each batch
    for batch in tqdm.tqdm(preprocessed_batches, desc="Text tokenization", disable=cfg.meta.tqdm_disable):
        text_prompts_batch = batch['text_prompts']
        tokenized_texts = tokenize_texts_batch(text_prompts_batch, tokenizer)
        batch['tokenized_texts'] = tokenized_texts
    
    logger.info("--- Phase 2 Complete ---")

    # === PHASE 3: GPU Embedding Extraction and Selection ===
    logger.info("--- Phase 3: GPU Embedding Extraction and Selection ---")
    model.eval()
    
    selected_audio_sources_summary = []
    final_wer_lines = [''] * len(text_prompts)
    final_transcription_lines = [''] * len(text_prompts)
    selected_audio_copy_jobs: List[Tuple[str, str]] = []

    for batch in tqdm.tqdm(preprocessed_batches, desc="GPU processing", disable=cfg.meta.tqdm_disable):
        prompt_indices = batch['prompt_indices']
        text_prompts_batch = batch['text_prompts']
        tokenized_texts = batch['tokenized_texts']
        candidate_audios_batch = batch['candidate_audios_batch']
        candidate_iter_indices_batch = batch['candidate_iter_indices_batch']
        wer_lines_batch = batch['wer_lines_batch']
        transcription_lines_batch = batch['transcription_lines_batch']
        
        # Move tokenized texts to GPU and get text embeddings
        tokenized_texts_gpu = {k: v.to(device) for k, v in tokenized_texts.items()}
        
        with torch.no_grad():
            text_embeddings = model.get_text_embedding(tokenized_texts_gpu, normalize=True)  # (batch_size, embed_dim)
        
        if text_embeddings is None or text_embeddings.ndim != 2:
            logger.critical(f"FATAL: Failed to get text embeddings for batch. Expected 2D tensor, got {text_embeddings.shape if text_embeddings is not None else 'None'}")
            raise ValueError("Text embedding failed for batch")
        
        # Process each prompt's candidate audios
        for i, prompt_idx in enumerate(prompt_indices):
            candidate_audios = candidate_audios_batch[i]
            candidate_iter_indices = candidate_iter_indices_batch[i]
            num_candidates = len(candidate_audios)
            
            # Extract and batch audio features for this prompt
            audio_features_list = [candidate['features'] for candidate in candidate_audios]
            audio_masks_list = [candidate['attention_mask'] for candidate in candidate_audios]
            audio_paths_list = [candidate['audio_path'] for candidate in candidate_audios]
            
            # Batch the N candidate audios for this prompt
            batched_audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features_list, batch_first=True, padding_value=0.0
            )
            batched_audio_masks = torch.nn.utils.rnn.pad_sequence(
                audio_masks_list, batch_first=True, padding_value=0
            )
            
            # Move to GPU
            batched_audio_features_gpu = batched_audio_features.to(device)
            batched_audio_masks_gpu = batched_audio_masks.to(device)
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeddings = model.get_audio_embedding(
                    batched_audio_features_gpu,
                    attention_mask=batched_audio_masks_gpu,
                    normalize=True
                )  # (N, embed_dim)
            
            if audio_embeddings is None or audio_embeddings.ndim != 2 or audio_embeddings.shape[0] != num_candidates:
                logger.critical(f"FATAL: Audio embedding failed for prompt {prompt_idx}. Expected ({num_candidates}, embed_dim), got {audio_embeddings.shape if audio_embeddings is not None else 'None'}")
                raise ValueError(f"Audio embedding failed for prompt {prompt_idx}")
            
            # Calculate similarities
            prompt_text_embedding = text_embeddings[i:i+1]  # (1, embed_dim)
            similarity_scores = F.cosine_similarity(prompt_text_embedding, audio_embeddings, dim=1)  # (N,)
            
            # Select best candidate
            best_score_idx = torch.argmax(similarity_scores)
            best_overall_iter_idx = candidate_iter_indices[best_score_idx.item()]
            best_score = similarity_scores[best_score_idx].item()
            
            # Store results
            selected_audio_sources_summary.append({
                "prompt_idx": prompt_idx,
                "text_prompt": text_prompts_batch[i],
                "best_iteration_idx": best_overall_iter_idx,
                "best_score": best_score,
                "all_scores": dict(zip(candidate_iter_indices, similarity_scores.cpu().tolist()))
            })
            
            # Store info for final output
            best_audio_path = audio_paths_list[best_score_idx.item()]
            tgt_audio_path = os.path.join(output_audios_dir, f"{prompt_idx}.wav")
            selected_audio_copy_jobs.append((best_audio_path, tgt_audio_path))
            
            # Store the selected WER and transcription lines
            final_wer_lines[prompt_idx] = wer_lines_batch[i][best_score_idx.item()]
            final_transcription_lines[prompt_idx] = transcription_lines_batch[i][best_score_idx.item()]
    
    logger.info("--- Phase 3 Complete ---")

    # === PHASE 4: File Operations ===
    logger.info("--- Starting Final File Operations ---")
    
    # Copy selected audio files
    logger.info(f"Copying {len(selected_audio_copy_jobs)} selected audio files...")
    for src_path, tgt_path in tqdm.tqdm(selected_audio_copy_jobs, desc="Copying best audios", disable=cfg.meta.tqdm_disable):
        if os.path.exists(src_path):
            shutil.copy2(src_path, tgt_path)
        else:
            logger.critical(f"FATAL: Source audio {src_path} for copy operation not found during final file operations.")
            raise FileNotFoundError(f"Source audio {src_path} not found during final copy operation.")
    logger.info("Audio file copying complete.")

    # Copy common files
    common_files = [
        "input_descriptions.txt", "input_genders.txt",
        "input_negative_descriptions.txt", "input_texts.txt"
    ]
    logger.info(f"Copying common files: {common_files}...")
    for common_file in common_files:
        src_common_path = os.path.join(iter1_dir, common_file)
        tgt_common_path = os.path.join(output_base_dir, common_file)
        if os.path.exists(src_common_path):
            shutil.copy2(src_common_path, tgt_common_path)
        else:
            logger.warning(f"Common file {src_common_path} not found in {iter1_dir}. Skipping copy.")
    logger.info("Common file copying complete.")

    # Write new WER and transcriptions files
    wer_output_path = os.path.join(output_base_dir, "wer_per_example.txt")
    with open(wer_output_path, 'w') as f:
        for line in final_wer_lines:
            f.write(line + '\n')
    logger.info(f"Wrote selected WER lines to {wer_output_path}")

    trans_output_path = os.path.join(output_base_dir, "output_transcriptions.txt")
    with open(trans_output_path, 'w') as f:
        for line in final_transcription_lines:
            f.write(line + '\n')
    logger.info(f"Wrote selected transcription lines to {trans_output_path}")

    # Save selection summary
    if cfg.meta.selection_summary_file:
        summary_path = os.path.join(output_base_dir, cfg.meta.selection_summary_file)
        with open(summary_path, 'w') as f:
            json.dump(selected_audio_sources_summary, f, indent=2)
        logger.info(f"Selection summary saved to: {summary_path}")
    logger.info("--- Final File Operations Complete ---")

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    logger.info(f"--- Best-of-N CLAP Selection Complete. Output in {output_base_dir} ---")


# Usage: python scripts/best_of_n.py checkpoint_path=./ckpt.pth.tar input_base_dir=... output_dir_name=best_of_N_paraspeechclap
@hydra.main(config_path="../configs", config_name="best_of_n/base", version_base=None)
def main(cfg: DictConfig) -> None:
    run_start_time = datetime.datetime.now()
    # --- Setup --- #
    log_level_str = cfg.meta.get("log_level", "INFO")
    set_log_level(log_level_str)
    
    logger.info(f"Starting Best-of-N CLAP selection run at {run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("--- Configuration ---")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logger.info("--------------------")

    if cfg.checkpoint_path is None or not os.path.exists(cfg.checkpoint_path):
        logger.error(f"CLAP model checkpoint not specified or not found: {cfg.checkpoint_path}")
        raise FileNotFoundError("CLAP model checkpoint `checkpoint_path` must be set in the config.")

    # Device setup
    if cfg.meta.get("device"):
        device = torch.device(cfg.meta.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        if device.index is not None:
            torch.cuda.set_device(device)
        elif torch.cuda.is_available():
             torch.cuda.set_device(0)
             device = torch.device('cuda:0')
    logger.info(f"Using device: {device}")

    # --- Load Tokenizer and Feature Extractor ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
        logger.info(f"Text tokenizer loaded successfully: {cfg.models.text}")
    except Exception as e:
        logger.error(f"Failed to load text tokenizer '{cfg.models.text}'. Error: {e}")
        raise
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.models.speech)
        # Ensure feature extractor's sampling rate matches target_sr if it has one
        if hasattr(feature_extractor, 'sampling_rate') and feature_extractor.sampling_rate != cfg.audio_processing.target_sr:
            logger.warning(
                f"Feature extractor '{cfg.models.speech}' sampling rate ({feature_extractor.sampling_rate}) "
                f"does not match target_sr ({cfg.audio_processing.target_sr}). This might lead to issues. "
                f"The script will use target_sr ({cfg.audio_processing.target_sr}) for resampling audio BEFORE feature extraction."
            )
            # If your feature_extractor can be configured with a sampling_rate, do it here.
            # Otherwise, ensure the input to feature_extractor is at feature_extractor.sampling_rate
            # The current preprocess_audio resamples to target_sr, then passes that to feature_extractor.
            # This should be fine as long as the feature_extractor is robust or its internal sampling_rate matches target_sr.
        logger.info(f"Audio feature_extractor loaded successfully: {cfg.models.speech}")
    except Exception as e:
        logger.error(f"Failed to load audio feature_extractor '{cfg.models.speech}'. Error: {e}")
        raise

    # --- Load CLAP Model --- #
    try:
        # Pass the whole cfg. If get_model uses parts of it (like clap_checkpoint, clap_config_path, etc.)
        model = get_model(cfg, device) 
        logger.info(f"CLAP model loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load the CLAP model. Exiting. Error: {e}")
         logger.exception("Traceback for model loading failure:")
         return # Exit if model loading fails

    # --- Run Best-of-N Selection --- #
    try:
        select_best_of_n(cfg, model, tokenizer, feature_extractor, device)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"A critical error occurred during the best-of-N selection process: {e}")
        logger.exception("Traceback:")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Traceback:")

    run_end_time = datetime.datetime.now()
    logger.info(f"Best-of-N script finished at {run_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {run_end_time - run_start_time}")

if __name__ == "__main__":
    main() 
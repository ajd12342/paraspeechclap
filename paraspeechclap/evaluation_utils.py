import os
import torch
from omegaconf import DictConfig
from typing import Dict, List, Optional

from paraspeechclap.model import CLAP
from paraspeechclap.debug_utils import logger, debug_tensor

CLASSIFICATION_TEMPLATE = "A person is speaking in a {} style."

def get_model(cfg: DictConfig, device: torch.device) -> CLAP:
    """
    Loads or creates a CLAP model based on the evaluation configuration.

    Args:
        cfg: The Hydra DictConfig object containing model and checkpoint settings.
             Requires `cfg.models` (speech, text, embedding_dim) and
             optionally `cfg.checkpoint_path`.
        device: The torch device (e.g., 'cuda', 'cpu') to load the model onto.

    Returns:
        The loaded or initialized CLAP model instance on the specified device.

    Raises:
        FileNotFoundError: If `cfg.checkpoint_path` is provided but the file does not exist.
        ValueError: If required model configuration keys are missing.
    """
    logger.info("--- Loading/Creating Model ---")
    try:
        model = CLAP(
            speech_name=cfg.models.speech,
            text_name=cfg.models.text,
            embedding_dim=cfg.models.embedding_dim,
            projection_dropout=cfg.models.get("projection_dropout", 0.5),
        )
        logger.info(f"Model architecture created: Speech='{cfg.models.speech}', Text='{cfg.models.text}', Dim={cfg.models.embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to create model instance: {e}")
        logger.error("Ensure cfg.models.speech, cfg.models.text, and cfg.models.embedding_dim are set.")
        raise ValueError("Missing critical model configuration parameters.") from e
    ckpt_path = cfg.checkpoint_path

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Attempting to load checkpoint from: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Attempt strict loading first
            model.load_state_dict(state_dict, strict=True)
            logger.info("Checkpoint loaded successfully using strict loading.")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}. Checking if mismatch is related to 'logit_scale' keys.")
            
            current_model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            missing_keys = list(current_model_keys - checkpoint_keys)
            unexpected_keys = list(checkpoint_keys - current_model_keys)
            
            all_mismatched_keys = missing_keys + unexpected_keys
            
            is_logit_scale_issue = all(
                "logit_scale" in key for key in all_mismatched_keys
            ) if all_mismatched_keys else False

            if is_logit_scale_issue:
                logger.warning("All mismatched keys contain 'logit_scale'. Attempting non-strict loading.")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Checkpoint loaded non-strictly (ignored 'logit_scale' mismatches).")
                except Exception as final_e:
                    logger.error(f"Non-strict loading also failed: {final_e}")
                    raise RuntimeError(f"Error loading checkpoint (even non-strictly for logit_scale): {ckpt_path}") from final_e
            else:
                logger.error(f"Mismatched keys not exclusively 'logit_scale': Missing={missing_keys}, Unexpected={unexpected_keys}")
                raise RuntimeError(f"Error loading checkpoint due to key mismatches not related to 'logit_scale': {ckpt_path}") from e
        except Exception as e: # Catch other potential errors during loading
            logger.error(f"Failed to load state dict from {ckpt_path} due to an unexpected error: {e}")
            raise RuntimeError(f"Error loading checkpoint: {ckpt_path}") from e

    elif ckpt_path: # Path provided but doesn't exist
        error_msg = f"Evaluation checkpoint path '{ckpt_path}' specified but not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    else: # No path provided
        warn_msg = "No 'checkpoint_path' provided in config. Using randomly initialized model weights."
        logger.warning(warn_msg)

    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info(f"Model prepared successfully on device '{device}' and set to evaluation mode.")
    return model

def calculate_audio_to_text_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    audio_to_original_text_mapping: List[int]
) -> Dict[str, float]:
    """
    Calculates audio-to-text retrieval metrics with unique text embeddings.
    
    Args:
        similarity_matrix (torch.Tensor): Shape (N_audio, N_unique_text) where
                                         similarity_matrix[i, j] is the similarity between audio i and unique text j.
                                         Should ideally be on the CPU for these calculations.
        audio_to_original_text_mapping (List[int]): List of length N_audio where
                                                   audio_to_original_text_mapping[i] gives the index in the unique text set
                                                   that corresponds to the ground truth text for audio i.
    
    Returns:
        dict: A dictionary containing the calculated audio-to-text metrics:
              R@k_A2T (k=1,5,10), MedR_A2T, MeanR_A2T
    """
    logger.info("--- Calculating Audio-to-Text Retrieval Metrics ---")
    debug_tensor("Input similarity matrix for A2T metrics", similarity_matrix)

    n_audio, n_unique_text = similarity_matrix.shape
    if n_audio == 0 or n_unique_text == 0:
        logger.error("Similarity matrix is empty. Cannot calculate retrieval metrics.")
        raise ValueError("Similarity matrix is empty. Cannot calculate retrieval metrics.")

    if len(audio_to_original_text_mapping) != n_audio:
        error_msg = (f"Audio-to-text mapping length ({len(audio_to_original_text_mapping)}) "
                     f"must match number of audio samples ({n_audio}).")
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Ensure matrix is on CPU for consistency
    if similarity_matrix.device != torch.device('cpu'):
        logger.debug("Moving similarity matrix to CPU for metric calculation.")
        similarity_matrix = similarity_matrix.cpu()

    # Convert mapping to tensor for efficient operations
    gt_text_indices = torch.tensor(audio_to_original_text_mapping, dtype=torch.long, device=similarity_matrix.device)

    # Audio -> Text Retrieval:
    # For each audio query (row i), find the rank of the correct unique text.
    # argsort gives indices that would sort the similarities in descending order.
    sorted_sim_indices_a2t = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Find where the ground truth text index appears in the sorted list for each audio
    # For each audio i, we need to find where gt_text_indices[i] appears in sorted_sim_indices_a2t[i, :]
    ranks_a2t = []
    for i in range(n_audio):
        gt_idx = gt_text_indices[i]
        # Find the position of gt_idx in the sorted similarities for audio i
        rank_position = (sorted_sim_indices_a2t[i, :] == gt_idx).nonzero(as_tuple=True)[0]
        if len(rank_position) > 0:
            rank = rank_position[0] + 1  # Convert to 1-based rank
            ranks_a2t.append(rank.item())
        else:
            # This should not happen if mapping is correct, but handle gracefully
            logger.warning(f"Ground truth text index {gt_idx} not found for audio {i}")
            ranks_a2t.append(n_unique_text + 1)  # Worst possible rank

    ranks_a2t = torch.tensor(ranks_a2t, dtype=torch.long, device=similarity_matrix.device)

    debug_tensor("Audio->Text Ranks (1-based)", ranks_a2t)
    logger.info(f"Rank stats (A->T): Min={ranks_a2t.min().item()}, Max={ranks_a2t.max().item()}, "
                f"Mean={ranks_a2t.float().mean().item():.2f}, Median={ranks_a2t.median().item()}")

    results: Dict[str, float] = {}
    for k in [1, 5, 10]:
        results[f"R@{k}_A2T"] = (ranks_a2t <= k).float().mean().item() * 100

    results["MedR_A2T"] = ranks_a2t.median().item()
    results["MeanR_A2T"] = ranks_a2t.float().mean().item()
    
    logger.info(f"Calculated Audio-to-Text Retrieval Metrics: {results}")
    logger.info("--- Audio-to-Text Retrieval Metrics Calculation Complete ---")
    return results 
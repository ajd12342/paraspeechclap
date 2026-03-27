import torch
import logging

# Configure logging - basic configuration, level will be set by set_log_level
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger('paraspeechclap')

# Global debug flag is removed.

def set_log_level(level_name: str):
    """Set the global logger level."""
    # Removed global DEBUG update
    level_name_upper = level_name.upper()
    numeric_level = getattr(logging, level_name_upper, None)

    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        logger.info(f"Logger level set to {level_name_upper}")
        # Removed DEBUG flag setting based on numeric_level
        if numeric_level <= logging.DEBUG:
            logger.debug("Debug logging is active.")
    else:
        logger.warning(f"Invalid log level: {level_name_upper}. Defaulting to INFO.")
        logger.setLevel(logging.INFO)

def debug_tensor(name, tensor, stats_only=True):
    """Print debug information about a tensor."""
    # Removed 'if not DEBUG: return'
    try:
        if tensor is None:
            logger.debug(f"{name} is None")
            return
            
        if not isinstance(tensor, torch.Tensor):
            logger.debug(f"{name} (not a tensor): {tensor}")
            return
            
        if stats_only:
            if tensor.numel() > 0:
                try:
                    mean_val = tensor.float().mean().item()
                    # Calculate std only if numel > 1, otherwise set to nan
                    if tensor.numel() > 1:
                        std_val = tensor.float().std().item()
                    else:
                        std_val = float('nan') # or 0.0 if preferred, nan is more explicit

                    logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                                f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                                f"mean={mean_val:.6f}, std={std_val:.6f}, "
                                f"has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}")
                except Exception as e:
                    logger.warning(f"Error calculating tensor stats for {name}: {e}")
                    logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
            else:
                logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, EMPTY TENSOR")
        else:
            logger.debug(f"{name}: {tensor}")
    except Exception as e:
        logger.warning(f"Error in debug_tensor for {name}: {e}")

def debug_model_parameters(model, name="model"):
    """Print debug information about model parameters."""
    # Removed 'if not DEBUG: return'
    try:    
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.debug(f"{name} - Total parameters: {total_params:,}")
        logger.debug(f"{name} - Trainable parameters: {trainable_params:,}")
        
        # Check for NaN or Inf values in parameters
        has_nan = False
        has_inf = False
        
        try:
            for param_name, param in model.named_parameters():
                try:
                    if torch.isnan(param).any():
                        has_nan = True
                        logger.warning(f"NaN detected in {param_name}")
                    if torch.isinf(param).any():
                        has_inf = True
                        logger.warning(f"Inf detected in {param_name}")
                except Exception as e:
                    logger.warning(f"Error checking parameter {param_name}: {e}")
        except Exception as e:
            logger.warning(f"Error iterating model parameters: {e}")
        
        if has_nan or has_inf:
            logger.warning(f"{name} has NaN or Inf parameters!")
    except Exception as e:
        logger.warning(f"Error in debug_model_parameters: {e}")

def debug_batch_data(batch, batch_idx):
    """Debug information about a batch of data."""
    # Removed 'if not DEBUG: return'
    logger.debug(f"Processing debug_batch_data for batch_idx: {batch_idx}")
    try:    
        # The existing logic for checking batch structure and logging needs to be maintained.
        # However, we need to ensure it uses logger.debug for its outputs.
        # Assuming 'batch' is a dictionary as per collate_fn
        if isinstance(batch, dict):
            logger.debug(f"Batch {batch_idx}:")
            if 'audio' in batch:
                logger.debug(f"  Audio shape: {batch['audio'].shape if isinstance(batch['audio'], torch.Tensor) else 'N/A'}")
            if 'text_tokens' in batch and isinstance(batch['text_tokens'], dict) and 'input_ids' in batch['text_tokens']:
                 logger.debug(f"  Text token input_ids shape: {batch['text_tokens']['input_ids'].shape if isinstance(batch['text_tokens']['input_ids'], torch.Tensor) else 'N/A'}")
            # Add more specific logging as needed based on your batch structure
            # For example, if you still have 'text' as a list of strings before tokenization:
            if 'text' in batch and isinstance(batch['text'], list) and len(batch['text']) > 0:
                 logger.debug(f"  Text samples (first 2): {batch['text'][:2]}")
            if 'audio_path' in batch and isinstance(batch['audio_path'], list) and len(batch['audio_path']) > 0:
                 logger.debug(f"  Audio paths (first 2): {batch['audio_path'][:2]}")
            if 'label' in batch:
                 logger.debug(f"  Labels (first 2): {batch['label'][:2] if isinstance(batch['label'], (list, torch.Tensor)) and len(batch['label']) > 0 else batch.get('label')}")

        else:
            logger.debug(f"Batch {batch_idx} is not a dictionary. Type: {type(batch)}")

    except Exception as e:
        logger.warning(f"Error in debug_batch_data: {e}", exc_info=True)
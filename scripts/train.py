import os
import json
import random
import time
import logging # Added for logging.DEBUG
from typing import Dict, Optional, List # Added import

import audtorch
import numpy as np
import torch
import tqdm
import wandb
import hydra  # Add hydra import
from hydra.core.hydra_config import HydraConfig # Import for accessing Hydra runtime config
from omegaconf import (
    OmegaConf,
    DictConfig # Import DictConfig for type hinting
)
from transformers import AutoTokenizer
from transformers import logging as transformers_logging
import torch.nn.functional as F # Add F for normalization
import torch.distributed as dist # Added for DDP
from torch.nn.parallel import DistributedDataParallel as DDP # Added for DDP
from torch.utils.data.distributed import DistributedSampler # Added for DDP
import datetime # Added for DDP timeout

from paraspeechclap.dataset import ParaSpeechCapsDataset
from paraspeechclap.loss import ClipLoss, MultiTaskLoss
from paraspeechclap.model import (
    CLAP
)
from paraspeechclap.debug_utils import (
    set_log_level, debug_model_parameters, debug_batch_data, logger
)
from paraspeechclap.utils import collate_fn, TARGET_SR
from paraspeechclap.balanced_sampler import (
    BalancedTagSampler, DistributedBalancedTagSampler, 
    analyze_batch_tag_distribution, log_tag_distribution_stats
)


# --- Template Loading Functions ---
def load_training_templates(template_json_path: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """
    Loads training templates from a JSON file.
    
    Args:
        template_json_path: Path to JSON file containing templates.
                           Expected format: {"tag1": ["complete template string 1", "complete template string 2"], 
                                           "tag2": ["complete template string 3"]}
                           Templates should be complete strings, no placeholders needed.
    
    Returns:
        Dictionary mapping tags to lists of complete template strings, or None if no path provided
        or if loading fails.
    """
    if not template_json_path:
        # logger.info("No template JSON path provided for training, using default prompt template")
        return None
        
    if not os.path.exists(template_json_path):
        logger.error(f"Training template JSON file not found: {template_json_path}")
        raise FileNotFoundError(f"Training template JSON file not found: {template_json_path}")
    
    try:
        with open(template_json_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        if not isinstance(templates, dict):
            raise ValueError("Training templates JSON must be a dictionary")
        
        # Validate that all values are lists of strings
        for tag, template_list in templates.items():
            if not isinstance(template_list, list):
                raise ValueError(f"Templates for tag '{tag}' must be a list, got {type(template_list)}")
            if not template_list:
                raise ValueError(f"Template list for tag '{tag}' cannot be empty")
            for i, template in enumerate(template_list):
                if not isinstance(template, str):
                    raise ValueError(f"Template {i} for tag '{tag}' must be a string, got {type(template)}")
        
        logger.info(f"Loaded training templates for {len(templates)} tags from {template_json_path}")
        for tag, template_list in templates.items():
            logger.debug(f"Tag '{tag}': {len(template_list)} templates")
        
        return templates
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {template_json_path}: {e}")
        raise ValueError(f"Invalid JSON in training template file: {template_json_path}") from e
    except Exception as e:
        logger.error(f"Failed to load training templates from {template_json_path}: {e}")
        raise RuntimeError(f"Error loading training templates from {template_json_path}") from e

def generate_batch_tag_prompts(tag_vocabulary: List[str], 
                              templates: Optional[Dict[str, List[str]]] = None,
                              prompt_template: str = "A person is speaking in a {} style.") -> List[str]:
    """
    Generates prompts for tags using either loaded templates (with random selection) or default template.
    This is called for each batch to get fresh random template selections.
    
    Args:
        tag_vocabulary: List of tag names to generate prompts for
        templates: Optional dictionary mapping tags to complete template strings (no placeholders)
        prompt_template: Default template with {} placeholder for fallback
    
    Returns:
        List of prompts, one for each tag in tag_vocabulary
    """
    if templates is None:
        # Use default template for all tags
        logger.warning(f"FAILSAFE TRIGGERED: No templates provided, using default prompt template for all {len(tag_vocabulary)} tags. This may reduce training diversity!")
        return [prompt_template.format(tag) for tag in tag_vocabulary]
    
    prompts = []
    for tag in tag_vocabulary:
        if tag in templates:
            # Randomly select one complete template for this tag
            template_list = templates[tag]
            selected_template = random.choice(template_list)
            prompts.append(selected_template)  # Use template as-is, no formatting needed
            logger.debug(f"Training batch - Tag '{tag}': selected template '{selected_template}' from {len(template_list)} options")
        else:
            # Fall back to default template if tag not found in templates
            # logger.warning(f"FAILSAFE TRIGGERED: Training batch - Tag '{tag}' not found in templates, using default prompt template. This may reduce training diversity!")
            prompts.append(prompt_template.format(tag))
    
    return prompts

# --- DDP Helper Functions ---
def setup_ddp(local_rank_env_var: str = 'LOCAL_RANK', default_pg_timeout_minutes: int = 30):
    """Initializes the DDP process group with a timeout."""
    if not dist.is_available():
        # Fallback when torch.distributed is not available
        logger.info("DDP not available. Running in single-process mode.")
        return 0, 1, 0 # rank, world_size, local_rank

    if local_rank_env_var not in os.environ:
        # This error will be raised on the process that fails, and torchrun should terminate others.
        raise RuntimeError(f"Environment variable {local_rank_env_var} not set. DDP setup failed.")
    
    local_rank = int(os.environ[local_rank_env_var])
    
    timeout = datetime.timedelta(minutes=default_pg_timeout_minutes)
    logger.info(f"Setting DDP process group timeout to {timeout}")

    # init_method can be 'env://' if MASTER_ADDR and MASTER_PORT are set,
    # or automatically handled by launchers like torchrun.
    # Providing a timeout helps prevent indefinite hangs if a process dies.
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP Initialized: Rank {rank}/{world_size}, Local Rank: {local_rank}, Device: cuda:{local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    """Cleans up the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed.")

def is_main_process(rank: int) -> bool:
    """Checks if the current process is the main process (rank 0)."""
    return rank == 0

# --- End DDP Helper Functions ---




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.debug(f"Random seed set to {seed}")


def print_model_summary(model):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(model)
    logger.info(model)
    logger.info('# pars: {}'.format(num_pars))
    # Get device from model parameter if possible
    # For DDP model, access .module
    inner_model = model.module if isinstance(model, DDP) else model
    try:
        device = next(inner_model.parameters()).device
        logger.info('{} : {}'.format('device', device))
    except StopIteration:
        logger.info('{} : {}'.format('device', 'unknown (no parameters)'))
    
    # Debug model architecture
    debug_model_parameters(inner_model)


def load_training_dataset(cfg: DictConfig, tag_columns=None): 
    
    current_dataset_name_str = cfg.data.dataset_name
    current_audio_root = cfg.data.audio_root
    apply_random_crop = cfg.data.random_crop
    current_dataset_probabilities = cfg.data.get("dataset_probabilities", None)
    current_stopping_strategy = cfg.data.get("stopping_strategy", "all_exhausted")

    speech_model_config_name = cfg.models.speech

    logger.info(f"Loading training dataset: "
                f"HF dataset(s): {current_dataset_name_str}, "
                f"audio_root: {current_audio_root}, "
                f"speech_model_for_extractor: {speech_model_config_name}")
    logger.info(f"Training dataset interleaving params: probabilities='{current_dataset_probabilities}', stopping_strategy='{current_stopping_strategy}'")

    transform = None
    if apply_random_crop:
        transform = audtorch.transforms.RandomCrop(int(cfg.data.crop_duration * TARGET_SR), axis=-1)

    logger.debug(f"Instantiating ParaSpeechCapsDataset for training with:", extra={
        "speech_model_name": speech_model_config_name,
        "dataset_name": current_dataset_name_str,
        "split": cfg.data.split,
        "audio_root": current_audio_root,
        "dataset_probabilities": current_dataset_probabilities,
        "stopping_strategy": current_stopping_strategy,
        "transform_is_set": transform is not None
    })

    dataset_instance = ParaSpeechCapsDataset(
        speech_model_name=speech_model_config_name,
        split=cfg.data.split,
        dataset_name=current_dataset_name_str,
        transform=transform,
        audio_root=current_audio_root,
        dataset_probabilities=current_dataset_probabilities,
        stopping_strategy=current_stopping_strategy,
        concatenate_datasets=cfg.data.get("concatenate_datasets", False),
        is_train=True,
        tag_columns=tag_columns
    )
    return dataset_instance


def create_dataloader(cfg: DictConfig, dataset_instance, tokenizer, is_train: bool, rank: int, world_size: int, tag_vocabulary=None, tag_columns=None): # Add tag_columns for explicit tag handling
    
    collate_with_tokenizer = lambda batch: collate_fn(batch, tokenizer, tag_columns)

    num_workers = cfg.hparams.get('num_workers', 0)
    # Batch size can be different for train and eval, if specified in config
    batch_size = cfg.hparams.batch_size if is_train else cfg.hparams.get("eval_batch_size", cfg.hparams.batch_size)
    
    # Check if class-balanced sampling is enabled for training
    use_balanced_sampling = (
        is_train and 
        cfg.multitask.get("enable_classification", False) and 
        cfg.multitask.get("use_balanced_sampling", False) and
        tag_vocabulary is not None
    )
    
    if use_balanced_sampling:
        logger.info(f"Using class-balanced sampling for training (Rank {rank})")
        
        # Get balanced sampling configuration
        balanced_config = cfg.multitask.get("balanced_sampling", {})
        weighting_strategy = balanced_config.get("weighting_strategy", "inverse_frequency")
        min_rare_tags_per_batch = balanced_config.get("min_rare_tags_per_batch", 0)
        rare_tag_threshold = balanced_config.get("rare_tag_threshold", 0.05)
        tag_column = balanced_config.get("tag_column", tag_columns[0] if tag_columns else "intrinsic_tags")
        
        if world_size > 1: # DDP is active
            sampler = DistributedBalancedTagSampler(
                dataset_instance,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg.meta.seed,
                drop_last=False,
                tag_column=tag_column,
                weighting_strategy=weighting_strategy,
                batch_size=batch_size,
                min_rare_tags_per_batch=min_rare_tags_per_batch,
                rare_tag_threshold=rare_tag_threshold
            )
        else: # Single GPU/CPU
            sampler = BalancedTagSampler(
                dataset_instance,
                tag_column=tag_column,
                weighting_strategy=weighting_strategy,
                batch_size=batch_size,
                min_rare_tags_per_batch=min_rare_tags_per_batch,
                rare_tag_threshold=rare_tag_threshold
            )
        
        # When using custom sampler, shuffle in DataLoader must be False
        shuffle_dataloader = False
        
        logger.info(f"Balanced sampler configured with:")
        logger.info(f"  - Tag column: {tag_column}")
        logger.info(f"  - Weighting strategy: {weighting_strategy}")
        logger.info(f"  - Min rare tags per batch: {min_rare_tags_per_batch}")
        logger.info(f"  - Rare tag threshold: {rare_tag_threshold}")
        
    else:
        # Use standard sampling
        if world_size > 1: # DDP is active
            sampler = DistributedSampler(
                dataset_instance,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train, # Shuffle for training sampler
                drop_last=False    # Ensure all GPUs process same number of samples for consistent global_step
            )
            # When using DistributedSampler, shuffle in DataLoader must be False
            shuffle_dataloader = False
        else: # Single GPU/CPU
            sampler = None
            shuffle_dataloader = is_train

    logger.info(f"Creating DataLoader for {'training' if is_train else 'validation/evaluation'} (Rank {rank}): "
                f"num_workers={num_workers}, batch_size={batch_size}, shuffle_dataloader={shuffle_dataloader}, "
                f"sampler_type={type(sampler).__name__ if sampler else 'None'}, balanced_sampling={use_balanced_sampling}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        batch_size=batch_size,
        shuffle=shuffle_dataloader, 
        num_workers=num_workers,
        collate_fn=collate_with_tokenizer,
        drop_last=False, # Process all batches from sampler (which might be padded if drop_last=False in sampler)
        sampler=sampler
    )
    logger.info(f"{'Training' if is_train else 'Validation'} set size (Rank {rank}): {len(dataset_instance)}, iterations (Rank {rank}): {len(dataloader)}")
    return dataloader


def setup_optimizer_and_criterion(cfg: DictConfig, model_parameters_for_optimizer, criterion_class=ClipLoss): # Pass model_parameters_for_optimizer
    logger.debug("Setting up optimizer and criterion (simplified for pre-set requires_grad)", extra={
        "learning_rate": cfg.hparams.learning_rate
    })
    

    
    trainable_params = [p for p in model_parameters_for_optimizer if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Total trainable parameters for optimizer: {num_trainable}")
    if num_trainable == 0:
        logger.warning("No trainable parameters found for optimizer. Check model freezing configuration and requires_grad settings.")

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.hparams.learning_rate)
    logger.debug("Optimizer created", extra={'optimizer_details': str(optimizer)})
    
    # Choose criterion based on multi-task configuration
    if cfg.multitask.get("enable_classification", False):
        criterion = MultiTaskLoss()
        logger.info("Using MultiTaskLoss")
    else:
        criterion = criterion_class()
        logger.info("Using standard ClipLoss")
    
    logger.debug("Loss function created", extra={'criterion_details': str(criterion)})
    
    return optimizer, criterion






# --- Training Step Function ---
def run_training_step(
    model: DDP, # DDP wrapped model
    batch: Dict[str, any],
    device: torch.device,
    criterion, # Can be ClipLoss or MultiTaskLoss
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    global_step: int,
    epoch: int,
    rank: int,
    world_size: int,
    train_dataloader_len: int,
    num_train_batches_in_epoch_local: int, # Current batch index in this epoch on this GPU
    tokenizer=None, # Add tokenizer for tag embedding computation
    templates=None, # Add templates for random template selection
    classification_tag_columns=None, # List of tag columns used for classification
    tag_vocabularies_by_column=None # Dict[column]->vocabulary list
) -> float:
    """
    Runs a single training step, including forward, backward, and optimizer step.

    Supports multi-head classification: pass `classification_tag_columns` and
    `tag_vocabularies_by_column` to build one masked-BCE head per tag column.
    Each head contributes its own classification loss; losses are summed.
    Samples with None/empty tags for a head are masked out for that head.
    """

    debug_batch_data(batch, num_train_batches_in_epoch_local -1) 
    
    audio_batch = batch['audio'].to(device)
    text_tokens = batch['text_tokens'] 
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    audio_mask_batch = batch['audio_attention_mask'].to(device)

    # Forward pass (DDP model handles internal sync)
    local_text_emb, local_speech_emb, local_logit_scale_exp = model(audio_batch, text_tokens, audio_attention_mask=audio_mask_batch)

    if world_size > 1:
        # Gather text embeddings for loss
        world_text_emb_list_for_loss = [torch.empty_like(local_text_emb) for _ in range(world_size)]
        dist.all_gather(world_text_emb_list_for_loss, local_text_emb.contiguous())
        current_rank = dist.get_rank() # Get current rank
        world_text_emb_list_for_loss[current_rank] = local_text_emb
        all_text_features_for_loss = torch.cat(world_text_emb_list_for_loss, dim=0)

        # Gather speech embeddings for loss
        world_speech_emb_list_for_loss = [torch.empty_like(local_speech_emb) for _ in range(world_size)]
        dist.all_gather(world_speech_emb_list_for_loss, local_speech_emb.contiguous())
        world_speech_emb_list_for_loss[current_rank] = local_speech_emb # Use current_rank from above
        all_speech_features_for_loss = torch.cat(world_speech_emb_list_for_loss, dim=0)

    else: # Single GPU
        all_text_features_for_loss = local_text_emb
        all_speech_features_for_loss = local_speech_emb


    # Compute tag embeddings and classification loss if multi-task learning is enabled
    classification_heads = []
    classification_valid_samples = None
    classification_total_samples = None
    
    if cfg.multitask.get("enable_classification", False) and isinstance(criterion, MultiTaskLoss):
        if tokenizer is None:
            logger.error("FAILSAFE TRIGGERED: Tag vocabulary or tokenizer is None for multi-task learning - this will break classification training!")
            raise ValueError("Tag vocabulary and tokenizer must be provided for multi-task learning")
        
        columns_for_classification = classification_tag_columns

        # Shared prompt template
        prompt_template = cfg.get("multitask", {}).get("prompt_template", "A person is speaking in a {} style.")

        # For each column, build a classification head (masked BCE) if batch contains it
        for classification_tag_column in columns_for_classification:
            local_tags_batch = batch.get(classification_tag_column, None)
            if local_tags_batch is None:
                logger.warning(f"FAILSAFE TRIGGERED: Batch missing classification tag column '{classification_tag_column}' - skipping this head for this batch!")
                continue

            # Build vocabulary for this column
            if tag_vocabularies_by_column is None or classification_tag_column not in tag_vocabularies_by_column:
                logger.error(f"Missing tag vocabulary for column '{classification_tag_column}'")
                continue
            current_vocab = tag_vocabularies_by_column[classification_tag_column]
            if not current_vocab:
                logger.warning(f"Vocabulary for column '{classification_tag_column}' is empty; skipping head")
                continue

            # Generate deterministic tag prompts per rank per step
            if world_size > 1:
                if is_main_process(rank):
                    tag_prompts = generate_batch_tag_prompts(current_vocab, templates, prompt_template)
                    obj_list = [tag_prompts]
                else:
                    obj_list = [None]
                dist.broadcast_object_list(obj_list, src=0)
                tag_prompts = obj_list[0]
            else:
                tag_prompts = generate_batch_tag_prompts(current_vocab, templates, prompt_template)

            # Tokenize prompts and get embeddings
            tag_tokens = tokenizer.batch_encode_plus(
                tag_prompts,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            tag_tokens = {k: v.to(device) for k, v in tag_tokens.items()}
            inner_model = model.module if isinstance(model, DDP) else model
            current_tag_embeddings = inner_model.get_text_embedding(tag_tokens, normalize=False)

            # Optional consistency check across ranks
            # if world_size > 1 and logger.isEnabledFor(logging.DEBUG):
            #     tag_embeddings_sum = torch.sum(current_tag_embeddings)
            #     all_sums = [torch.zeros_like(tag_embeddings_sum) for _ in range(world_size)]
            #     dist.all_gather(all_sums, tag_embeddings_sum)
            #     if not all(torch.allclose(all_sums[0], s, atol=1e-6) for s in all_sums):
            #         logger.warning("Tag embeddings differ across ranks - possible DDP synchronization issue")

            # Gather tag lists across ranks to align with gathered audio features
            if world_size > 1:
                all_tags_batch = [None] * world_size
                dist.all_gather_object(all_tags_batch, local_tags_batch)
                combined_tags_batch = []
                for rank_tags in all_tags_batch:
                    combined_tags_batch.extend(rank_tags)
            else:
                combined_tags_batch = local_tags_batch

            # Log valid counts per head
            classification_total_samples = len(combined_tags_batch)
            classification_valid_samples = sum(1 for t in combined_tags_batch if t is not None and len(t) > 0)
            if is_main_process(rank) and (num_train_batches_in_epoch_local % 100 == 0):
                logger.info(f"[{classification_tag_column}] valid samples this batch: {classification_valid_samples}/{classification_total_samples}")

            # Build mapping and add head
            current_tag_to_idx = {tag: idx for idx, tag in enumerate(current_vocab)}
            classification_heads.append({
                'tag_embeddings': current_tag_embeddings,
                'rich_tags_batch': combined_tags_batch,
                'tag_to_idx': current_tag_to_idx,
                'name': classification_tag_column
            })

    # Compute loss (contrastive + classification if enabled)
    if isinstance(criterion, MultiTaskLoss):
        # Explicit multitask branch returns separate losses
        loss_out = criterion(
            all_text_features_for_loss,
            all_speech_features_for_loss,
            local_logit_scale_exp,
            classification_heads=classification_heads
        )
        current_train_loss = loss_out['total_loss']
        contrastive_loss = loss_out['contrastive_loss']
        classification_loss = loss_out['classification_loss']
    else:
        # Non-multitask branch returns a single scalar loss
        current_train_loss = criterion(
            all_text_features_for_loss,
            all_speech_features_for_loss,
            local_logit_scale_exp
        )
        contrastive_loss = current_train_loss
        classification_loss = torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    current_train_loss.backward() # DDP handles gradient averaging

    optimizer.step()
    
    # Log batch metrics to W&B (only on main process)
    if is_main_process(rank):
        log_dict = {
            'loss/train_batch': current_train_loss.item(),
            'loss/contrastive_batch': contrastive_loss.item(),
            'loss/classification_batch': classification_loss.item(),
            'logit_scale/train_batch': local_logit_scale_exp.item(),
            'step': global_step, 
            'epoch_detailed': float(epoch) - 1 + (num_train_batches_in_epoch_local / train_dataloader_len) if train_dataloader_len > 0 else float(epoch) -1
        }
        # Include classification valid sample counts if available
        if classification_valid_samples is not None and classification_total_samples is not None:
            log_dict['classification/valid_samples'] = classification_valid_samples
            log_dict['classification/valid_ratio'] = classification_valid_samples / max(1, classification_total_samples)
        wandb.log(log_dict)
    
    return current_train_loss.item()
# --- End Training Step Function ---


# --- Model Parameter Requires Grad Setup ---
def set_model_parameter_requires_grad(model: torch.nn.Module, rank: int):
    """
    Sets requires_grad flags for model parameters.
    Freezes the text pooler since we use CLS from last_hidden_state.
    This should be called on the CPU model BEFORE DDP wrapping.
    """
    if is_main_process(rank):
        logger.info("Setting model parameter requires_grad flags on CPU model...")

    # Freeze text_branch.base.pooler parameters (we use CLS from last_hidden_state, not the pooler)
    if hasattr(model, 'text_branch') and hasattr(model.text_branch, 'base'):
        base_model = model.text_branch.base
        if hasattr(base_model, 'pooler'):
            if is_main_process(rank):
                logger.info("Explicitly setting requires_grad=False for text_branch.base.pooler parameters.")
            for param_name, param in base_model.pooler.named_parameters():
                param.requires_grad = False
                if is_main_process(rank) and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Set requires_grad=False for pooler param: {param_name}")
        elif is_main_process(rank):
            logger.warning("Could not find pooler in text model to set requires_grad=False. Check model structure.")
    elif is_main_process(rank):
        logger.warning("Could not find text_branch.base on CPU model to set requires_grad=False. Check model structure.")
    
    # Count trainable parameters
    if is_main_process(rank):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.2f}%)")
        
        # Debug: Log parameter names and requires_grad status
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Parameter breakdown:")
            for name, param in model.named_parameters():
                logger.debug(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    if is_main_process(rank):
        logger.info("Finished setting model parameter requires_grad flags.")
# --- End Model Parameter Requires Grad Setup ---


# --- Helper function for checkpoint saving ---
def _save_checkpoint(
    model_to_save: torch.nn.Module, 
    is_ddp: bool, 
    ckpt_dir: str, 
    base_filename: str, 
    epoch: int, 
    global_step: int | None = None, # Optional, for step-based saving
    rank: int = 0 # For logging
):
    """Saves a model checkpoint.

    Args:
        model_to_save: The model instance (can be DDP-wrapped).
        is_ddp: Boolean indicating if the model is DDP-wrapped.
        ckpt_dir: Directory to save the checkpoint.
        base_filename: Base name for the checkpoint file (e.g., "epoch_X" or "step_Y").
        epoch: Current epoch number.
        global_step: Current global step number (optional).
        rank: DDP rank for logging.
    """
    if not is_main_process(rank):
        return

    if ckpt_dir is None:
        logger.warning(f"Rank {rank}: Checkpoint directory is None. Skipping saving {base_filename}.")
        return

    # Construct filename, adding .pth.tar convention
    filename_with_ext = f"{base_filename}.pth.tar"
    ckpt_path = os.path.join(ckpt_dir, filename_with_ext)
    
    # Get the underlying model (unwrap DDP if needed)
    underlying_model = model_to_save.module if is_ddp else model_to_save
    state_dict_to_save = underlying_model.state_dict()
    
    torch.save(state_dict_to_save, ckpt_path)
    log_msg = f"Rank {rank}: Model checkpoint saved to {ckpt_path} (Epoch {epoch}"
    if global_step is not None:
        log_msg += f", Global Step {global_step}"
    log_msg += ")"
    logger.info(log_msg)

    # Also update/save the 'last.pth.tar' checkpoint
    last_ckpt_path = os.path.join(ckpt_dir, 'last.pth.tar')
    torch.save(state_dict_to_save, last_ckpt_path)
    logger.info(f"Rank {rank}: Updated last model checkpoint to {last_ckpt_path} (Epoch {epoch}, Global Step {global_step if global_step else 'N/A'})")


@hydra.main(config_path="../configs", config_name="train/intrinsic", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_time = time.time()
    transformers_logging.set_verbosity_error()

    # --- Logger Setup (before DDP to ensure all processes configure it) ---
    # The logger is already configured in debug_utils, just set the level from config.
    # Default to INFO if not specified, though config.yaml provides a default.
    log_level_str = cfg.meta.get("log_level", "INFO")
    set_log_level(log_level_str)
    # The debug system now uses log_level instead of cfg.meta.debug.
    # Debug functions check logger.isEnabledFor(logging.DEBUG) or use logger.debug() calls.

    # --- DDP Setup ---
    # LOCAL_RANK is typically set by torchrun or similar launchers
    # Get DDP timeout from config, defaulting if not specified.
    ddp_timeout_minutes = cfg.meta.get("ddp_timeout_minutes", 30) 
    rank, world_size, local_rank = setup_ddp(local_rank_env_var='LOCAL_RANK', default_pg_timeout_minutes=ddp_timeout_minutes)

    # --- Hydra Setup & Logging ---
    # The set_log_level call above now handles logger configuration.
    # The global DEBUG variable in debug_utils is also set by set_log_level for now.

    if is_main_process(rank):
        logger.info("Starting ParaSpeechCLAP training with Hydra and DDP")
        logger.info(OmegaConf.to_yaml(cfg))
    else:
        # Suppress Hydra's verbose output on non-main processes if possible
        # or simply log less.
        logger.info(f"DDP Process Rank {rank}/{world_size} started.")


    # --- Directory Setup (Manual structure: results/experiment_name/timestamp) ---
    # Only main process should create directories
    ckpt_dir = None
    debug_dir_main_proc = None # Renamed to avoid conflict with train() function's debug_dir
    if is_main_process(rank):
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"Timestamp for experiment folder: {timestamp_str}")

        results_dir = cfg.meta.results
        base_experiment_name = cfg.meta.experiment_name
        experiment_folder = os.path.join(results_dir, base_experiment_name, timestamp_str)
        logger.info(f"Full experiment path: {experiment_folder}")

        ckpt_dir = os.path.join(experiment_folder, 'ckpt')
        debug_dir_main_proc = os.path.join(experiment_folder, 'debug') if logger.isEnabledFor(logging.DEBUG) else None
        os.makedirs(ckpt_dir, exist_ok=True)
        if debug_dir_main_proc:
            os.makedirs(debug_dir_main_proc, exist_ok=True)

        # --- WandB Setup ---
        # Use experiment name and timestamp for WandB run name for consistency
        wandb_run_name = f"{base_experiment_name}_{timestamp_str}" 
        logger.info(f"WandB run name: {wandb_run_name}")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=wandb_run_name,
            dir=experiment_folder # Log wandb files within the Hydra output directory
        )
    
    # --- Reproducibility & Device Setup ---
    setup_seed(cfg.meta.seed + rank) # Add rank for different seeds per process if desired, e.g. for data augmentation
    
    # Device is set by setup_ddp() using local_rank for DDP, otherwise use config or auto-detect
    if world_size > 1: # DDP is active
        device = torch.device(f'cuda:{local_rank}')
        # torch.cuda.set_device(local_rank) is already called in setup_ddp for DDP case
    else: # Single process mode (world_size == 1)
        if cfg.meta.get("device"): # User-specified device in config
            device = torch.device(cfg.meta.device)
        else: # Auto-detect for single process mode
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # If CUDA is chosen for single GPU, explicitly set the device if it's cuda:X
        # For DDP, local_rank handles this. For single GPU, if user says "cuda:1" and it's available.
        if device.type == 'cuda':
            if device.index is not None:
                torch.cuda.set_device(device) # e.g. cuda:0, cuda:1
            elif torch.cuda.is_available(): # User just said "cuda"
                 torch.cuda.set_device(0) # Default to cuda:0 if available and no specific index
                 device = torch.device('cuda:0') # Update device to be specific

    logger.info(f"Rank {rank} using device: {device}")

    # --- Tokenizer Setup ---
    # All processes need the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
    if is_main_process(rank):
        logger.info(f"Loaded tokenizer: {cfg.models.text}")

    # --- Data Setup ---
    # Determine which tag columns we need
    tag_columns = []
    if cfg.multitask.get("enable_classification", False):
        configured_columns = OmegaConf.to_container(cfg.multitask.tag_columns, resolve=True)
        tag_columns.extend(list(configured_columns))
    
    # Each process loads its own dataset portion via DistributedSampler
    train_dataset = load_training_dataset(cfg, tag_columns)
    
    # --- Build tag vocabulary and templates for multi-task learning ---
    tag_vocabulary = None
    tag_vocabularies_by_column = {}
    
    if cfg.multitask.get("enable_classification", False):
        # Build vocabulary for each requested classification column
        for col in tag_columns:
            logger.info(f"Rank {rank}: Building tag vocabulary for classification from column '{col}'...")
            vocab_for_col = train_dataset.get_tag_vocabulary_for_column(col)
            tag_vocabularies_by_column[col] = vocab_for_col
            logger.info(f"Rank {rank}: Built tag vocabulary for '{col}' with {len(vocab_for_col)} tags")
            if not vocab_for_col:
                logger.warning(f"Rank {rank}: Vocabulary for column '{col}' is empty.")
        # Ensure at least one non-empty vocabulary exists
        if all(len(v) == 0 for v in tag_vocabularies_by_column.values()):
            logger.error("FAILSAFE TRIGGERED: All classification vocabularies are empty!")
            raise ValueError("Empty tag vocabularies - cannot perform multi-task learning")

        # Canonicalize vocabularies across ranks by broadcasting from rank 0
        if world_size > 1:
            for col in tag_columns:
                if is_main_process(rank):
                    obj_list = [tag_vocabularies_by_column.get(col, [])]
                else:
                    obj_list = [None]
                dist.broadcast_object_list(obj_list, src=0)
                tag_vocabularies_by_column[col] = obj_list[0]

        # For backward-compat logging/analysis, set tag_vocabulary to first column's (now canonical) vocab if present
        if tag_columns:
            tag_vocabulary = tag_vocabularies_by_column.get(tag_columns[0], [])
    
    # Load templates for multi-task training if provided (only if classification is enabled)
    templates = None
    if cfg.multitask.get("enable_classification", False):
        template_json_path = cfg.multitask.get("template_json_path")
        if template_json_path:
            if is_main_process(rank):
                logger.info(f"Loading training templates from: {template_json_path}")
            try:
                templates = load_training_templates(template_json_path)
                if templates and is_main_process(rank):
                    logger.info(f"Successfully loaded templates for {len(templates)} tags")
                    # Log template counts for tags across all vocabularies
                    all_vocab_tags = []
                    for col in tag_columns:
                        all_vocab_tags.extend(tag_vocabularies_by_column.get(col, []))
                    for tag in all_vocab_tags:
                        if tag in templates:
                            logger.info(f"  Tag '{tag}': {len(templates[tag])} templates available")
                        else:
                            logger.info(f"  Tag '{tag}': not found in templates, will use default")
            except Exception as e:
                logger.error(f"Failed to load training templates from {template_json_path}: {e}")
                logger.warning("Falling back to default prompt template for training")
                templates = None
        else:
            if is_main_process(rank):
                logger.info("No template JSON path provided for training, using default prompt template")
        
        # Strict: Verify vocabulary equality across ranks in debug mode (per column)
        if world_size > 1 and logger.isEnabledFor(logging.DEBUG):
            for col in tag_columns:
                local_vocab = tag_vocabularies_by_column.get(col, [])
                gathered_vocabs = [None] * world_size
                dist.all_gather_object(gathered_vocabs, local_vocab)
                reference_vocab = gathered_vocabs[0]
                if not all(v == reference_vocab for v in gathered_vocabs):
                    mismatch_info = {i: len(v) for i, v in enumerate(gathered_vocabs)}
                    logger.error(f"Rank {rank}: Tag vocabularies differ across ranks for '{col}'. Sizes per rank: {mismatch_info}")
                    # Log a few differing entries for diagnostics
                    for i, v in enumerate(gathered_vocabs):
                        if v != reference_vocab:
                            logger.error(f"  Rank {i} first 10 entries: {v[:10] if isinstance(v, list) else type(v)}")
                    raise RuntimeError(f"Inconsistent tag vocabularies across ranks for column '{col}'")
        
        if is_main_process(rank):
            logger.info("Multi-task learning enabled with classification objective")
            for col in tag_columns:
                logger.debug(f"Tag vocabulary size for '{col}': {len(tag_vocabularies_by_column.get(col, []))}")
    else:
        if is_main_process(rank):
            logger.info("Multi-task learning disabled - using contrastive loss only")

    # Choose tag column and vocabulary for balanced sampling (single column)
    sampling_tag_column = cfg.multitask.get("balanced_sampling", {}).get("tag_column")
    if not sampling_tag_column:
        sampling_tag_column = tag_columns[0] if tag_columns else None
    sampling_vocab_for_dataloader = tag_vocabularies_by_column.get(sampling_tag_column, []) if cfg.multitask.get("enable_classification", False) else None

    # Create training dataloader (now that we have tag vocabularies)
    # Pass the sampling vocabulary and tag columns to the dataloader creation
    train_dataloader = create_dataloader(cfg, train_dataset, tokenizer, is_train=True, rank=rank, world_size=world_size, tag_vocabulary=sampling_vocab_for_dataloader, tag_columns=tag_columns)

    # --- Model Setup ---
    if is_main_process(rank):
        logger.info(f"Creating CLAP model with speech_name={cfg.models.speech}, text_name={cfg.models.text}, embed_dim={cfg.models.embedding_dim}")
    
    # Model created on CPU first, then moved to device, then wrapped by DDP
    model = CLAP(
        speech_name=cfg.models.speech,
        text_name=cfg.models.text,
        embedding_dim=cfg.models.embedding_dim,
        projection_dropout=cfg.models.get("projection_dropout", 0.5),
    )

    # Barrier to ensure all ranks are synchronized before DDP wrapping
    if world_size > 1:
        dist.barrier()

    # --- Modify requires_grad flags (on CPU model, before DDP wrapping) ---
    set_model_parameter_requires_grad(model, rank)
    # --- End requires_grad modifications ---

    model.to(device) # Move model to the assigned GPU for this rank

    if world_size > 1: # DDP is active
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) 
        # Enable static graph for gradient checkpointing compatibility
        # model._set_static_graph()
        if is_main_process(rank): 
            logger.info("Model wrapped with DistributedDataParallel (find_unused_parameters=False).")
            
    if is_main_process(rank):
        logger.info('-' * 64)
        print_model_summary(model)

    # --- Optimizer and Criterion Setup ---
    # Optimizer uses model.parameters() which DDP handles. Critically, requires_grad flags are now final.
    optimizer, criterion = setup_optimizer_and_criterion(cfg, model.parameters(), criterion_class=ClipLoss)

    # Barrier before starting training to ensure all setup is complete
    if world_size > 1:
        logger.info(f"Rank {rank} waiting at barrier before training loop.")
        dist.barrier()
        logger.info(f"Rank {rank} passed barrier, starting training loop.")

    # --- Training Loop ---
    if is_main_process(rank): logger.info("Starting training loop...")
    global_step = 0 # Initialize global step counter
    
    num_epochs = cfg.hparams.get('epochs', None) or None
    max_steps = cfg.hparams.get('max_steps', None) or None

    if num_epochs is None and max_steps is None:
        raise ValueError("Either hparams.epochs or hparams.max_steps must be specified (both are null).")

    if num_epochs is not None and max_steps is not None:
        if is_main_process(rank):
            logger.warning(f"Both epochs ({num_epochs}) and max_steps ({max_steps}) are set. "
                           f"Training will stop at whichever limit is reached first.")

    if num_epochs is None:
        num_epochs = int(1e9)
        if is_main_process(rank):
            logger.info(f"No epoch limit set; training until max_steps={max_steps}")

    if max_steps is None:
        max_steps = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        if is_main_process(rank): logger.info(f"Starting epoch {epoch}" + (f"/{num_epochs}" if num_epochs < int(1e9) else ""))

        # Set sampler epoch for distributed training
        if world_size > 1 and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Initialize tag distribution tracking for balanced sampling analysis (per column)
        batch_tag_distributions = {col: [] for col in (tag_columns if cfg.multitask.get("enable_classification", False) else [])}
        log_tag_distribution_every_n_batches = cfg.multitask.get("log_tag_distribution_every_n_batches", 100)

        # === Training Phase ===
        model.train()
        total_epoch_train_loss_local = 0 # Loss on this GPU
        num_train_batches_in_epoch_local = 0 # Batches processed on this GPU

        # Use tqdm only on main process to avoid multiple bars
        train_iterator = train_dataloader
        if is_main_process(rank):
            train_iterator = tqdm.tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f'Epoch {epoch} Train (Rank {rank})', # Clarify rank for main process bar
                disable=cfg.meta.tqdm_disable
            )

        for index, batch in enumerate(train_iterator): # Removed tqdm for non-main processes
            if batch is None: 
                error_msg = f"FAILSAFE TRIGGERED: TRAINING (Rank {rank}): Encountered a None batch at batch_idx {index} from train_dataloader. Check data loading and collate_fn."
                logger.error(error_msg)
                raise ValueError(error_msg)

            global_step +=1 
            num_train_batches_in_epoch_local += 1
            
            # Log tag distribution periodically for balanced sampling analysis (per column)
            if (cfg.multitask.get("enable_classification", False) and 
                cfg.multitask.get("use_balanced_sampling", False) and
                is_main_process(rank) and
                num_train_batches_in_epoch_local % log_tag_distribution_every_n_batches == 0):
                for col in tag_columns:
                    col_vocab = tag_vocabularies_by_column.get(col, [])
                    batch_tags = batch.get(col, [])
                    if batch_tags and col_vocab:
                        batch_tag_freq = analyze_batch_tag_distribution(batch_tags, col_vocab)
                        batch_tag_distributions[col].append(batch_tag_freq)
                        log_tag_distribution_stats(
                            batch_tag_freq,
                            prefix=f"Batch {num_train_batches_in_epoch_local} (Epoch {epoch}) [Column: {col}]: "
                        )
                        non_zero_tags = sum(1 for freq in batch_tag_freq.values() if freq > 0)
                        tag_coverage = non_zero_tags / len(col_vocab) if col_vocab else 0
                        wandb.log({
                            f'tag_distribution/{col}/batch_coverage': tag_coverage,
                            f'tag_distribution/{col}/batch_unique_tags': non_zero_tags,
                            'step': global_step
                        })
            
            current_train_loss_item = run_training_step(
                model=model,
                batch=batch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                cfg=cfg,
                global_step=global_step,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
                train_dataloader_len=len(train_dataloader),
                num_train_batches_in_epoch_local=num_train_batches_in_epoch_local,
                tokenizer=tokenizer,
                templates=templates,
                classification_tag_columns=tag_columns,
                tag_vocabularies_by_column=tag_vocabularies_by_column
            )
                            
            total_epoch_train_loss_local += current_train_loss_item

            # --- Periodic Checkpoint Saving (by step) ---
            save_every_n_steps = cfg.hparams.get('save_every_n_steps', 0)
            if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                _save_checkpoint(
                    model_to_save=model,
                    is_ddp=(world_size > 1),
                    ckpt_dir=ckpt_dir, # ckpt_dir is defined in main() and available here
                    base_filename=f"step_{global_step}_model",
                    epoch=epoch,
                    global_step=global_step,
                    rank=rank
                )

            # --- Early stop if max_steps reached ---
            if max_steps > 0 and global_step >= max_steps:
                if is_main_process(rank):
                    logger.info(f"Reached max_steps={max_steps} at global_step={global_step}. Stopping training.")
                break
        
        # Aggregate and log average training loss for the epoch (main process only)
        if world_size > 1:
            # Sum local losses and batch counts, then average on rank 0
            total_epoch_loss_tensor = torch.tensor([total_epoch_train_loss_local, num_train_batches_in_epoch_local], dtype=torch.float64, device=device)
            dist.reduce(total_epoch_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            
            if is_main_process(rank):
                total_epoch_train_loss_global = total_epoch_loss_tensor[0].item()
                num_train_batches_global = total_epoch_loss_tensor[1].item()
                avg_epoch_train_loss = total_epoch_train_loss_global / num_train_batches_global if num_train_batches_global > 0 else 0.0
        else: # Single GPU
            if is_main_process(rank): # Should always be true for single GPU if rank is 0
                avg_epoch_train_loss = total_epoch_train_loss_local / num_train_batches_in_epoch_local if num_train_batches_in_epoch_local > 0 else 0.0

        if is_main_process(rank):
            logger.info(f'Epoch {epoch} average training loss (aggregated): {avg_epoch_train_loss:.4f}')
            
            # Log epoch-level tag distribution statistics per column
            wandb_log_payload = {
                'loss/train_epoch_avg': avg_epoch_train_loss,
                'epoch': epoch
            }
            if (cfg.multitask.get("enable_classification", False) and 
                cfg.multitask.get("use_balanced_sampling", False)):
                for col in tag_columns:
                    col_freq_list = batch_tag_distributions.get(col, [])
                    col_vocab = tag_vocabularies_by_column.get(col, [])
                    if col_freq_list and col_vocab:
                        avg_coverage = np.mean([
                            sum(1 for freq in batch_freq.values() if freq > 0) / len(col_vocab)
                            for batch_freq in col_freq_list
                        ])
                        avg_unique_tags = np.mean([
                            sum(1 for freq in batch_freq.values() if freq > 0)
                            for batch_freq in col_freq_list
                        ])
                        logger.info(f"Epoch {epoch} tag distribution stats for '{col}':")
                        logger.info(f"  - Average tag coverage: {avg_coverage:.3f}")
                        logger.info(f"  - Average unique tags per batch: {avg_unique_tags:.1f}")
                        wandb_log_payload[f'tag_distribution/{col}/epoch_avg_coverage'] = avg_coverage
                        wandb_log_payload[f'tag_distribution/{col}/epoch_avg_unique_tags'] = avg_unique_tags
            wandb.log(wandb_log_payload)

        # Clamp the log_logit_scale (original logic, happens per epoch end)
        # For DDP: Only rank 0 clamps, then broadcasts to maintain synchronization
        # Since manual parameter updates aren't synced by DDP, we use explicit broadcast
        
        # Access .module for DDP model
        current_model_instance = model.module if isinstance(model, DDP) else model
        with torch.no_grad():
            prev_log_logit_scale = current_model_instance.log_logit_scale.item()
            max_log_logit = np.log(cfg.models.get("max_logit_scale", 100.0))
            
            if world_size > 1:
                if is_main_process(rank):
                    # Only rank 0 clamps
                    current_model_instance.log_logit_scale.clamp_(min=0.0, max=max_log_logit)
                    clamped_value = current_model_instance.log_logit_scale.data.clone()
                else:
                    # Other ranks prepare to receive the clamped value
                    clamped_value = torch.zeros_like(current_model_instance.log_logit_scale.data)
                
                # Broadcast the clamped value from rank 0 to all ranks
                dist.broadcast(clamped_value, src=0)
                
                # All ranks update their parameter with the broadcasted value
                current_model_instance.log_logit_scale.data = clamped_value
                clamped_log_logit = clamped_value.item()
            else:
                # Single GPU case - clamp directly
                current_model_instance.log_logit_scale.clamp_(min=0.0, max=max_log_logit)
                clamped_log_logit = current_model_instance.log_logit_scale.item()

        if is_main_process(rank):
            logger.debug(f"Logit scale clamped (log scale)", extra={
                 "before (log)": prev_log_logit_scale,
                 "after (log)": clamped_log_logit,
                 "max_allowed (log)": max_log_logit
            })
            wandb.log({"logit_scale_clamped": np.exp(clamped_log_logit), "epoch": epoch})


        # --- Checkpoint Saving (only on main process) ---
        if is_main_process(rank) and ckpt_dir is not None:
            # Save epoch-specific checkpoint
            _save_checkpoint(
                model_to_save=model,
                is_ddp=(world_size > 1),
                ckpt_dir=ckpt_dir,
                base_filename=f"epoch_{epoch}_model", # Changed to include _model for consistency
                epoch=epoch,
                global_step=global_step, # Pass global_step for more complete logging of 'last'
                rank=rank
            )

        epoch_end = time.time()
        epoch_time_minutes = (epoch_end - epoch_start) / 60
        if is_main_process(rank):
            logger.info(f'Epoch {epoch} time: {epoch_time_minutes:.2f} minutes')
            wandb.log({
                'epoch_time_minutes': epoch_time_minutes,
                'epoch': epoch
            })

        # Break out of epoch loop if max_steps was reached
        if max_steps > 0 and global_step >= max_steps:
            break

    # --- Final Logging ---
    if is_main_process(rank):
        end_time = time.time()
        total_training_hours = (end_time - start_time) / 3600
        logger.info(f'Total training time: {total_training_hours:.2f} hours')
        wandb.log({'total_training_hours': total_training_hours})
        wandb.finish()
        logger.info("Training finished successfully on main process.")

    # --- DDP Cleanup ---
    cleanup_ddp()


if __name__ == '__main__':
    main() # Entry point for the script

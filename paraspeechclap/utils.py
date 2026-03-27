import torch
import torch.nn.utils.rnn as rnn_utils
from transformers import PreTrainedTokenizerBase
from typing import Optional, List
from paraspeechclap.debug_utils import logger

TARGET_SR = 16000

def collate_fn(batch, tokenizer: Optional[PreTrainedTokenizerBase] = None, tag_columns: Optional[List[str]] = None):
    """
    Collate function for DataLoader.
    - Pads audio sequences and generates audio attention masks and explicit audio lengths.
    - If a tokenizer is provided, tokenizes positive text sequences and returns 'text_tokens'.
    - If no tokenizer is provided, collects raw text strings under 'text'.
    - Collects 'audio_path' and 'label' (if present) into lists.

    Args:
        batch (list): A list of dictionaries, each potentially containing 'audio', 'text',
                      'audio_path', 'label'.
        tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer for text processing. 
                                                       If None, raw text is returned.
        tag_columns (Optional[List[str]]): Specific tag columns to include in output.
                                          If None, no tag columns are included.

    Returns:
        dict: A dictionary containing processed batch data with keys:
              - 'audio': padded audio tensors
              - 'audio_attention_mask': attention mask for padded audio
              - 'audio_lengths': original audio lengths before padding
              - 'text_tokens' or 'text': positive text data (processed or raw)
              - 'audio_path', 'label': optional fields if present in input
              - tag columns: specified tag columns if tag_columns is provided
              Returns None if the batch is empty.
    """
    # Filter out None items if any
    original_batch_size = len(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) != original_batch_size:
        logger.warning(f"FAILSAFE TRIGGERED: collate_fn filtered out {original_batch_size - len(batch)} None items from batch of size {original_batch_size}")
    
    if not batch:
        logger.error("FAILSAFE TRIGGERED: collate_fn received empty batch after filtering None items, returning None - this will cause training issues!")
        return None

    # Separate audio and text, get audio lengths
    audio_list = []
    for i, item in enumerate(batch):
        if 'audio' not in item:
            logger.error(f"FAILSAFE TRIGGERED: collate_fn batch item {i} missing 'audio' key!")
            raise ValueError(f"Batch item {i} missing required 'audio' key")
        
        audio_tensor = torch.as_tensor(item['audio'])
        if audio_tensor.numel() == 0:
            logger.warning(f"FAILSAFE TRIGGERED: collate_fn batch item {i} has empty audio tensor!")
        audio_list.append(audio_tensor)
    
    audio_lengths = [len(audio) for audio in audio_list]

    # Pad variable-length audio sequences to longest in batch
    audio_padded = rnn_utils.pad_sequence(audio_list, batch_first=True, padding_value=0.0)

    # Create audio attention mask (1 for real tokens, 0 for padding)
    max_audio_len = audio_padded.shape[1]
    audio_attention_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
    for i, length in enumerate(audio_lengths):
        audio_attention_mask[i, :length] = 1

    output_dict = {
        'audio': audio_padded,
        'audio_attention_mask': audio_attention_mask,
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
    }

    # Handle positive texts
    if batch and 'text' in batch[0]:
        positive_texts = []
        
        for i, item in enumerate(batch):
            if 'text' in item:
                text_value = item['text']
                if text_value is None or (isinstance(text_value, str) and text_value.strip() == ""):
                    raise ValueError(f"collate_fn batch item {i} has None text!")
                else:
                    positive_texts.append(text_value)
            else:
                logger.error(f"FAILSAFE TRIGGERED: collate_fn batch item {i} missing 'text' key!")
                raise ValueError(f"Text is not present in batch item {i}")

        if tokenizer and positive_texts:
            # Tokenize positive texts
            text_tokens = tokenizer.batch_encode_plus(
                positive_texts,
                padding='longest',
                truncation=True,
                max_length=512, # Explicit max length is good practice
                return_tensors='pt'
            )
            output_dict['text_tokens'] = text_tokens
        elif positive_texts:
            # Return raw positive text strings if no tokenizer is provided
            output_dict['text'] = positive_texts

    # Handle optional keys by checking the first item (assuming batch items are uniform)
    # This assumes that if one item has audio_path/label, all should, or it should be handled by dataset logic
    if batch and 'audio_path' in batch[0]:
        output_dict['audio_path'] = [item['audio_path'] for item in batch]
    
    if batch and 'label' in batch[0]:
        output_dict['label'] = [item['label'] for item in batch]
    
    # Handle specific tag columns if requested
    if tag_columns and batch:
        for tag_column in tag_columns:
            output_dict[tag_column] = [item[tag_column] for item in batch]
    
        
    return output_dict



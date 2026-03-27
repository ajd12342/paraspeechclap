import os
import torch
import random
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset, interleave_datasets, concatenate_datasets as hf_concatenate_datasets # Renamed for clarity and added interleave/concatenate
from transformers import Wav2Vec2FeatureExtractor
from paraspeechclap.debug_utils import logger, debug_tensor
from paraspeechclap.utils import TARGET_SR # Import from utils
from typing import Optional, List

class ParaSpeechCapsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            speech_model_name: Optional[str] = None,
            split="train_base",
            dataset_name="ajd12342/paraspeechcaps", # Can be "dset1+dset2" for multiple
            transform=None,
            audio_root=None,
            gold_label_column: str | None = None,
            dataset_probabilities: str | None = None, # e.g., "0.7+0.3"
            stopping_strategy: str | None = None, # "all_exhausted" or "first_exhausted". Only used for training.
            concatenate_datasets: bool = False, # If True, concatenate datasets instead of interleaving
            is_train: bool = False,
            sort_by_duration: bool = False,
            tag_columns: List[str] | None = None  # Specific tag columns to include in output
    ):
        """
        Dataset class for the ParaSpeechCaps dataset from Hugging Face.
        Can handle single or multiple datasets for training (either interleaved or concatenated).
        
        Args:
            speech_model_name (str): The Hugging Face model name for the speech encoder.
            split (str): The dataset split to use.
            dataset_name (str): HF dataset name. For multiple training datasets,
                                separate with '+': e.g., "org/dataset1+org/dataset2".
            transform: Optional transform to apply to the audio.
            audio_root (str): Optional root directory for audio files.
            gold_label_column (str | None): Gold label column for classification.
                                           If provided, script fails if not in ALL datasets.
            dataset_probabilities (str | None): Sampling probabilities for multiple datasets,
                                                e.g., "0.7+0.3". Uniform if None. Only used for interleaving.
            stopping_strategy (str | None): For multiple datasets: "all_exhausted"
                                            or "first_exhausted". Only used for interleaving.
            concatenate_datasets (bool): If True, concatenate datasets instead of interleaving them.
                                        When True, dataset_probabilities and stopping_strategy are ignored.
            is_train (bool): Whether the dataset is for training or not.
            sort_by_duration (bool): If True, sort the dataset by the 'duration' column (if available).
            tag_columns (List[str] | None): Specific tag columns to include in output. If None, no tag columns are included.
        """
        logger.info(f"Initializing ParaSpeechCapsDataset (split={split}, speech_model_name={speech_model_name})")
        logger.info(f"Dataset name(s): {dataset_name}, probabilities: {dataset_probabilities}, stopping: {stopping_strategy}")
        logger.info(f"Concatenate datasets: {concatenate_datasets}")
        logger.info(f"Audio root: {audio_root}, gold_label_column: {gold_label_column}, sort_by_duration: {sort_by_duration}")
        logger.info(f"Tag columns to include: {tag_columns}")

        self.gold_label_column = gold_label_column
        self.concatenate_datasets = concatenate_datasets
        self.tag_columns = tag_columns or []
        
        self.feature_extractor = None
        if speech_model_name:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_name)
            logger.info(f"Initialized Wav2Vec2FeatureExtractor from {speech_model_name}")
        else:
            logger.info("No speech_model_name provided. Dataset will return raw audio waveforms.")

        self.transform = transform
        self.audio_root = audio_root

        # Load the dataset(s)
        if "+" in dataset_name and is_train: # Multiple datasets only for training splits
            dataset_names_list = [name.strip() for name in dataset_name.split("+")]
            num_datasets = len(dataset_names_list)
            logger.info(f"Multiple training datasets detected: {num_datasets} datasets -> {dataset_names_list}")

            individual_hf_datasets = []
            for name in dataset_names_list:
                logger.debug(f"Loading individual Hugging Face dataset: {name} for split: {split}")
                current_hf_dataset = hf_load_dataset(name, split=split)
                # Check for gold_label_column in each dataset if specified
                if self.gold_label_column and self.gold_label_column not in current_hf_dataset.features:
                    error_msg = (
                        f"Gold label column '{self.gold_label_column}' not found in features of dataset '{name}': "
                        f"{list(current_hf_dataset.features.keys())}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                individual_hf_datasets.append(current_hf_dataset)
            
            if self.concatenate_datasets:
                # Concatenate datasets
                logger.info("Concatenating datasets...")
                if dataset_probabilities:
                    logger.warning("dataset_probabilities specified but concatenate_datasets=True. Probabilities will be ignored.")
                if stopping_strategy:
                    logger.warning("stopping_strategy specified but concatenate_datasets=True. Stopping strategy will be ignored.")
                
                self.dataset = hf_concatenate_datasets(individual_hf_datasets)
                logger.info(f"Successfully concatenated {len(individual_hf_datasets)} datasets. Total size: {len(self.dataset)}")
            else:
                # Interleave datasets (original behavior)
                probabilities_list = None
                if dataset_probabilities:
                    probabilities_list = [float(p.strip()) for p in dataset_probabilities.split("+")]
                    if len(probabilities_list) != num_datasets:
                        raise ValueError("The length of dataset_probabilities does not match the number of datasets.")  
                    else:
                        logger.info(f"Using specified probabilities: {probabilities_list}")
                
                if probabilities_list is None and num_datasets > 1:
                    logger.info("Using uniform probabilities for dataset sampling.")

                logger.info(f"Interleaving datasets with strategy: '{stopping_strategy}'")

                self.dataset = interleave_datasets(
                    datasets=individual_hf_datasets,
                    probabilities=probabilities_list,
                    stopping_strategy=stopping_strategy
                )
                logger.info(f"Successfully interleaved {len(individual_hf_datasets)} datasets. Total size: {len(self.dataset)}")

        else: # Single dataset (or validation/test split which don't use multiple datasets here)
            if "+" in dataset_name and not is_train:
                raise ValueError("Multiple datasets specified for non-train split.")

            logger.info(f"Loading single Hugging Face dataset: {dataset_name} for split: {split}")
            self.dataset = hf_load_dataset(dataset_name, split=split)
            # Check if gold_label_column exists if provided for this single dataset
            if self.gold_label_column and self.gold_label_column not in self.dataset.features:
                error_msg = (
                    f"Gold label column '{self.gold_label_column}' not found in dataset features: "
                    f"{list(self.dataset.features.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(f"Loaded {len(self.dataset)} samples in total for split '{split}'")

        # Note: Tag vocabulary building is now done externally by the training script
        # when specific tag columns are needed, rather than building a default vocabulary

        # --- Optional: Sort dataset by duration ---
        if sort_by_duration:
            logger.info("Attempting to sort dataset by duration.")
            if self.dataset and 'duration' in self.dataset.column_names:
                try:
                    self.dataset = self.dataset.sort("duration", reverse=True)
                    logger.info(f"Successfully sorted dataset by 'duration'. New first sample after sort: {self.dataset[0] if len(self.dataset) > 0 else 'Empty dataset'}")
                except Exception as e:
                    logger.error(f"Error sorting dataset by 'duration': {e}. Proceeding with unsorted dataset.")
            elif not self.dataset:
                logger.warning("Dataset is not loaded. Cannot sort by duration.")
            else:
                logger.warning(f"'duration' column not found in dataset features. Cannot sort by duration. Available columns: {self.dataset.column_names}")
        else:
            logger.info("sort_by_duration is False, dataset will not be sorted by duration.")

        logger.debug("Final dataset features", extra={"features": str(self.dataset.features)})
        logger.debug("First sample", extra={"sample": str(self.dataset[0]) if len(self.dataset) > 0 else "Empty dataset"}) # Can be verbose

    def get_tag_vocabulary_for_column(self, tag_column: str) -> List[str]:
        """Build and return tag vocabulary for a specific tag column."""
        return self._build_tag_vocabulary_for_column(tag_column)
    
    def _build_tag_vocabulary_for_column(self, tag_column: str) -> List[str]:
        """Build vocabulary of all unique tags from the specified tag column."""
        all_tags = set()
        
        sample_indices = range(len(self.dataset))
        
        logger.info(f"Building tag vocabulary from column '{tag_column}'...")
        for idx in tqdm(sample_indices, desc=f"Building vocab from {tag_column}"):
            item = self.dataset[idx]
            if tag_column in item:
                if item[tag_column] is not None:
                    tags = item[tag_column]
                    if isinstance(tags, list):
                        for tag in tags:
                            if tag and isinstance(tag, str):
                                all_tags.add(tag.strip().lower())
                    else:
                        tag = tags
                        if tag and isinstance(tag, str):
                            all_tags.add(tag.strip().lower())
            else:
                raise ValueError(f"Item {idx} missing tag column '{tag_column}'.")
        
        logger.info(f"Built tag vocabulary from '{tag_column}' with {len(all_tags)} unique tags")
        return sorted(list(all_tags))  # Return sorted list for consistent ordering


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        logger.debug(f"Getting item {idx}", extra={"item_details": str(item)})
        
        if self.audio_root is None:
            raise ValueError(
                "audio_root must be specified. Set data.audio_root in your config to the "
                "common root directory containing per-source audio subdirectories."
            )
        source = item["source"]
        relative_audio_path = item["relative_audio_path"]
        audio_path = os.path.join(self.audio_root, source, relative_audio_path)
        logger.debug(f"Full audio path", extra={"full_path": audio_path})
        
        # Load audio using torchaudio
        try:
            audio, fs = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise e
        
        logger.debug(f"Audio loaded successfully using torchaudio", extra={"shape": str(audio.shape), "original_fs": fs})

        # Resample if necessary
        if fs != TARGET_SR:
            logger.debug(f"Resampling from {fs} Hz to {TARGET_SR} Hz")
            resampler = T.Resample(orig_freq=fs, new_freq=TARGET_SR)
            audio = resampler(audio)
            debug_tensor("Audio after resampling to TARGET_SR Hz", audio)
        
        # Convert to mono by averaging channels if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            debug_tensor("Audio after mono conversion", audio)

        # Squeeze to (L,) shape
        audio = audio.squeeze(0)
        debug_tensor("Audio squeezed to (L,)", audio)

        # Process with Wav2Vec2FeatureExtractor for normalization
        # Ensure audio is a 1D tensor, TARGET_SR should match feature_extractor.sampling_rate
        # The feature_extractor handles normalization if do_normalize=True in its config.
        if self.feature_extractor:
            if self.feature_extractor.sampling_rate != TARGET_SR:
                logger.warning(f"Mismatch between feature extractor sampling rate ({self.feature_extractor.sampling_rate}) and TARGET_SR ({TARGET_SR}). Ensure consistency.")
            
            input_features = self.feature_extractor(
                audio, 
                sampling_rate=TARGET_SR, 
                return_tensors="pt",
                padding="do_not_pad" 
            )
            audio = input_features.input_values.squeeze(0) # Remove batch dim, get normalized audio
            debug_tensor("Audio after Wav2Vec2FeatureExtractor processing", audio)

        # Handle text description (positive prompt)
        if "text_description" in item:
            text_descriptions = item["text_description"]
            # Randomly sample one description if multiple are present.
            # Note: For true randomness across batches in multiprocessing, ensure
            # DataLoader uses a worker_init_fn to set unique random seeds per worker.
            if isinstance(text_descriptions, list):
                if not text_descriptions:  # Empty list failsafe
                    raise ValueError(f"Item {idx} has empty text_descriptions list.")
                else:
                    text_description = random.choice(text_descriptions)
            else:
                if not text_descriptions or text_descriptions.strip() == "":  # Empty string failsafe
                    raise ValueError(f"Item {idx} has empty/whitespace-only text_description '{text_descriptions}'.")
                else:
                    text_description = text_descriptions
        else:
            logger.warning(f"Item {idx} missing 'text_description' key. Setting text_description to None. If training, very bad!")
            text_description = None
        
        # Apply transform to audio if specified
        if self.transform is not None:
            audio = self.transform(audio)
            logger.debug(f"Applied transform", extra={"new_shape": str(audio.shape)})
            debug_tensor("Audio after transform", audio)
        
        # Prepare output dictionary
        output_dict = {
            'audio': audio,
            'audio_path': audio_path
        }

        # Add text information
        if text_description is not None:
            output_dict['text'] = text_description

        # Add optional fields
        if self.gold_label_column:
            if self.gold_label_column not in item:
                raise ValueError(f"Item {idx} missing gold_label_column '{self.gold_label_column}'.")
            else:
                label = item[self.gold_label_column]
                if label is None or (isinstance(label, str) and label.strip() == ""):
                    raise ValueError(f"Item {idx} has empty/None gold_label '{label}'. This may impact classification training!")
                output_dict['label'] = label
        
        
        # Add only specific tag columns if requested
        for column_name in self.tag_columns:
            if column_name in item:
                tags = item[column_name]
                if isinstance(tags, list):
                    valid_tags = [tag.strip().lower() for tag in tags if tag and isinstance(tag, str) and tag.strip()]
                    if len(valid_tags) != len(tags):
                        empty_count = len(tags) - len(valid_tags)
                        logger.warning(f"FAILSAFE TRIGGERED: Item {idx} column '{column_name}' had {empty_count} empty/invalid tags out of {len(tags)} total. Filtered to {len(valid_tags)} valid tags.")
                    output_dict[column_name] = valid_tags
                else:
                    if tags and isinstance(tags, str) and tags.strip():
                        output_dict[column_name] = [tags.strip().lower()]
                    else:
                        # logger.warning(f"FAILSAFE TRIGGERED: Item {idx} column '{column_name}' has empty/invalid single tag '{tags}'. Setting to empty list.")
                        output_dict[column_name] = []
            else:
                raise ValueError(f"Item {idx} missing requested tag column '{column_name}'.")
            
        return output_dict

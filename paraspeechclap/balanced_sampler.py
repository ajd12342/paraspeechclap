"""
Balanced sampling utilities for class-balanced multitask training.

This module provides sampling strategies to address class imbalance in tag-based datasets
by ensuring more uniform distribution of tags across training batches.
"""

import torch
import numpy as np
import random
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Tuple
from torch.utils.data import Sampler
from paraspeechclap.debug_utils import logger


class TagFrequencyAnalyzer:
    """Analyzes tag frequencies in the dataset to compute sampling weights."""
    
    def __init__(self, dataset, tag_column: str = "rich_tags"):
        """
        Initialize the analyzer with a dataset.
        
        Args:
            dataset: The dataset to analyze (should have __len__ and __getitem__)
            tag_column: Name of the column containing tags
        """
        self.dataset = dataset
        self.tag_column = tag_column
        self.tag_frequencies = None
        self.sample_tag_sets = None
        self.tag_to_samples = None
        
    def analyze_tag_distribution(self) -> Dict[str, int]:
        """
        Analyze the distribution of tags in the dataset.
        
        Returns:
            Dictionary mapping tag names to their frequencies
        """
        logger.info(f"Analyzing tag distribution for {len(self.dataset)} samples...")
        
        # Validate that the requested tag column exists in the dataset schema
        try:
            raw_ds = self.dataset.dataset if hasattr(self.dataset, 'dataset') else self.dataset
            if hasattr(raw_ds, 'column_names'):
                if self.tag_column not in raw_ds.column_names:
                    raise ValueError(
                        f"Tag column '{self.tag_column}' not found in dataset columns: {list(raw_ds.column_names)}"
                    )
        except Exception as e:
            # Surface clear error if schema is incompatible
            raise
        
        tag_counter = Counter()
        sample_tag_sets = []
        tag_to_samples = defaultdict(set)
        
        for idx in range(len(self.dataset)):
            try:
                # Get the raw dataset item (not processed through __getitem__)
                if hasattr(self.dataset, 'dataset'):
                    # If it's a wrapper around HuggingFace dataset
                    item = self.dataset.dataset[idx]
                else:
                    # Direct access to dataset
                    item = self.dataset[idx]
                
                # Extract tags from the specified column
                tags = item.get(self.tag_column, [])
                if tags is None:
                    tags = []
                
                # Ensure tags is a list
                if not isinstance(tags, list):
                    tags = [tags] if tags else []
                
                # Normalize tags
                normalized_tags = set()
                for tag in tags:
                    if tag and isinstance(tag, str):
                        normalized_tag = tag.strip().lower()
                        normalized_tags.add(normalized_tag)
                        tag_counter[normalized_tag] += 1
                        tag_to_samples[normalized_tag].add(idx)
                
                sample_tag_sets.append(normalized_tags)
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                sample_tag_sets.append(set())
        
        self.tag_frequencies = dict(tag_counter)
        self.sample_tag_sets = sample_tag_sets
        self.tag_to_samples = {tag: list(samples) for tag, samples in tag_to_samples.items()}
        
        # Log statistics
        total_tags = sum(self.tag_frequencies.values())
        num_unique_tags = len(self.tag_frequencies)
        
        logger.info(f"Found {num_unique_tags} unique tags with {total_tags} total occurrences")
        logger.info(f"Average tags per sample: {total_tags / len(self.dataset):.2f}")
        
        # Log most and least frequent tags
        sorted_tags = sorted(self.tag_frequencies.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Most frequent tags: {sorted_tags[:10]}")
        logger.info(f"Least frequent tags: {sorted_tags[-10:]}")
        
        return self.tag_frequencies
    
    def compute_tag_sampling_weights(self, strategy: str = "inverse_frequency") -> Dict[str, float]:
        """
        Compute sampling weights for tags based on their frequencies.
        
        Args:
            strategy: Strategy for computing weights ("inverse_frequency", "sqrt_inverse", "log_inverse")
            
        Returns:
            Dictionary mapping tag names to their sampling weights
        """
        if self.tag_frequencies is None:
            self.analyze_tag_distribution()
        
        tag_weights = {}
        
        if strategy == "inverse_frequency":
            # Weight inversely proportional to frequency
            for tag, freq in self.tag_frequencies.items():
                tag_weights[tag] = 1.0 / freq
        
        elif strategy == "sqrt_inverse":
            # Square root of inverse frequency (less aggressive)
            for tag, freq in self.tag_frequencies.items():
                tag_weights[tag] = 1.0 / np.sqrt(freq)
        
        elif strategy == "log_inverse":
            # Log inverse frequency (even less aggressive)
            for tag, freq in self.tag_frequencies.items():
                tag_weights[tag] = 1.0 / np.log(freq + 1)
        
        else:
            raise ValueError(f"Unknown weighting strategy: {strategy}")
        
        # Normalize weights to sum to number of tags
        total_weight = sum(tag_weights.values())
        num_tags = len(tag_weights)
        for tag in tag_weights:
            tag_weights[tag] = (tag_weights[tag] / total_weight) * num_tags
        
        logger.info(f"Computed tag sampling weights using strategy '{strategy}'")
        logger.debug(f"Weight range: {min(tag_weights.values()):.4f} - {max(tag_weights.values()):.4f}")
        
        return tag_weights
    
    def compute_sample_weights(self, tag_weights: Dict[str, float]) -> List[float]:
        """
        Compute sampling weights for each sample based on its tags.
        
        Args:
            tag_weights: Dictionary mapping tag names to their weights
            
        Returns:
            List of weights for each sample in the dataset
        """
        if self.sample_tag_sets is None:
            self.analyze_tag_distribution()
        
        sample_weights = []
        
        for sample_tags in self.sample_tag_sets:
            if not sample_tags:
                # Sample has no tags, give it minimum weight
                sample_weights.append(min(tag_weights.values()) if tag_weights else 1.0)
            else:
                # Average weight of all tags in the sample
                tag_weight_sum = sum(tag_weights.get(tag, 1.0) for tag in sample_tags)
                sample_weights.append(tag_weight_sum / len(sample_tags))
        
        logger.info(f"Computed sample weights with range: {min(sample_weights):.4f} - {max(sample_weights):.4f}")
        
        return sample_weights


class BalancedTagSampler(Sampler):
    """
    Sampler that provides more balanced tag distribution across batches.
    
    This sampler works by:
    1. Computing weights for each sample based on the rarity of its tags
    2. Sampling batches using weighted sampling to favor underrepresented tags
    3. Optionally ensuring minimum representation of rare tags in each batch
    """
    
    def __init__(
        self,
        dataset,
        tag_column: str = "rich_tags",
        weighting_strategy: str = "inverse_frequency",
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        min_rare_tags_per_batch: int = 0,
        rare_tag_threshold: float = 0.05,
        generator: Optional[torch.Generator] = None
    ):
        """
        Initialize the balanced tag sampler.
        
        Args:
            dataset: The dataset to sample from
            tag_column: Name of the column containing tags
            weighting_strategy: Strategy for computing tag weights
            batch_size: Size of each batch
            num_samples: Number of samples to draw (defaults to dataset length)
            replacement: Whether to sample with replacement
            min_rare_tags_per_batch: Minimum number of rare tag samples per batch
            rare_tag_threshold: Threshold for considering a tag as rare (fraction of total samples)
            generator: Random number generator for reproducibility
        """
        self.dataset = dataset
        self.tag_column = tag_column
        self.weighting_strategy = weighting_strategy
        self.batch_size = batch_size
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        self.min_rare_tags_per_batch = min_rare_tags_per_batch
        self.rare_tag_threshold = rare_tag_threshold
        self.generator = generator
        
        # Initialize analyzer and compute weights
        self.analyzer = TagFrequencyAnalyzer(dataset, tag_column)
        self.tag_frequencies = self.analyzer.analyze_tag_distribution()
        self.tag_weights = self.analyzer.compute_tag_sampling_weights(weighting_strategy)
        self.sample_weights = self.analyzer.compute_sample_weights(self.tag_weights)
        
        # Identify rare tags and their samples
        total_samples = len(dataset)
        self.rare_tags = {
            tag for tag, freq in self.tag_frequencies.items()
            if freq / total_samples < rare_tag_threshold
        }
        
        self.rare_tag_samples = set()
        if self.analyzer.sample_tag_sets:
            for idx, sample_tags in enumerate(self.analyzer.sample_tag_sets):
                if any(tag in self.rare_tags for tag in sample_tags):
                    self.rare_tag_samples.add(idx)
        
        logger.info(f"BalancedTagSampler initialized:")
        logger.info(f"  - Dataset size: {len(dataset)}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Weighting strategy: {weighting_strategy}")
        logger.info(f"  - Rare tags: {len(self.rare_tags)} (threshold: {rare_tag_threshold})")
        logger.info(f"  - Samples with rare tags: {len(self.rare_tag_samples)}")
        logger.info(f"  - Min rare tags per batch: {min_rare_tags_per_batch}")
    
    def __iter__(self):
        """Generate batches with balanced tag distribution."""
        # Convert sample weights to tensor for PyTorch sampling
        weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float)
        
        # Generate indices for the entire epoch
        if self.replacement:
            indices = torch.multinomial(
                weights_tensor, 
                self.num_samples, 
                replacement=True,
                generator=self.generator
            ).tolist()
        else:
            # For sampling without replacement, we need a different approach
            # Use weighted sampling but track used indices
            indices = []
            available_indices = list(range(len(self.dataset)))
            available_weights = self.sample_weights.copy()
            
            for _ in range(min(self.num_samples, len(self.dataset))):
                if not available_indices:
                    break
                
                # Sample one index
                weights_tensor = torch.tensor(available_weights, dtype=torch.float)
                sampled_idx = torch.multinomial(weights_tensor, 1, generator=self.generator).item()
                
                # Add the actual dataset index
                actual_idx = available_indices[sampled_idx]
                indices.append(actual_idx)
                
                # Remove from available
                available_indices.pop(sampled_idx)
                available_weights.pop(sampled_idx)
        
        # Group indices into batches and optionally ensure rare tag representation
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # If we need to ensure rare tag representation and batch is not full
            if (self.min_rare_tags_per_batch > 0 and 
                len(batch_indices) == self.batch_size and 
                self.rare_tag_samples):
                
                # Count rare tag samples in current batch
                rare_count = sum(1 for idx in batch_indices if idx in self.rare_tag_samples)
                
                # If we don't have enough rare tag samples, replace some regular samples
                if rare_count < self.min_rare_tags_per_batch:
                    needed_rare = self.min_rare_tags_per_batch - rare_count
                    
                    # Find non-rare samples in batch to replace
                    non_rare_in_batch = [
                        (i, idx) for i, idx in enumerate(batch_indices) 
                        if idx not in self.rare_tag_samples
                    ]
                    
                    if len(non_rare_in_batch) >= needed_rare:
                        # Replace some non-rare samples with rare ones
                        rare_samples_list = list(self.rare_tag_samples)
                        replacement_rare = random.sample(rare_samples_list, needed_rare)
                        
                        for i in range(needed_rare):
                            batch_pos, _ = non_rare_in_batch[i]
                            batch_indices[batch_pos] = replacement_rare[i]
            
            batches.extend(batch_indices)
        
        return iter(batches)
    
    def __len__(self):
        """Return the number of samples per epoch."""
        return self.num_samples


class DistributedBalancedTagSampler(BalancedTagSampler):
    """
    Distributed version of BalancedTagSampler for multi-GPU training.
    """
    
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Initialize the distributed balanced tag sampler.
        
        Args:
            dataset: The dataset to sample from
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: Whether to shuffle the data
            seed: Random seed
            drop_last: Whether to drop the last incomplete batch
            **kwargs: Additional arguments passed to BalancedTagSampler
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Initialize the base sampler
        super().__init__(dataset, **kwargs)
        
        # Calculate samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples_per_replica = len(self.dataset) // self.num_replicas
        else:
            self.num_samples_per_replica = (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
        
        self.total_size = self.num_samples_per_replica * self.num_replicas
        
        logger.info(f"DistributedBalancedTagSampler initialized for rank {rank}/{num_replicas}")
        logger.info(f"  - Samples per replica: {self.num_samples_per_replica}")
        logger.info(f"  - Total size: {self.total_size}")
    
    def __iter__(self):
        """Generate indices for this replica."""
        if self.shuffle:
            # Generate seed based on epoch and base seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
        else:
            g = None
        
        # Get balanced indices from parent class
        self.generator = g
        saved_num_samples = self.num_samples
        self.num_samples = self.total_size  # Temporarily generate enough for all replicas

        indices = list(super().__iter__())

        self.num_samples = saved_num_samples  # Restore parent state
        
        # Ensure we have exactly total_size indices
        if len(indices) < self.total_size:
            # Pad with repetition
            indices += indices[:self.total_size - len(indices)]
        indices = indices[:self.total_size]
        
        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        
        return iter(indices)
    
    def __len__(self):
        """Return the number of samples for this replica."""
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int):
        """Set the epoch for this sampler (affects shuffling)."""
        self.epoch = epoch


def analyze_batch_tag_distribution(batch_tags: List[List[str]], tag_vocabulary: List[str]) -> Dict[str, float]:
    """
    Analyze the tag distribution in a batch.

    Args:
        batch_tags: List of tag lists for each sample in the batch
        tag_vocabulary: Complete vocabulary of all possible tags

    Returns:
        Dictionary mapping tag names to their frequencies in the batch
    """
    tag_counts = Counter()
    total_samples = len(batch_tags)
    tag_vocabulary_set = set(tag_vocabulary)

    for sample_tags in batch_tags:
        for tag in sample_tags:
            if tag in tag_vocabulary_set:
                tag_counts[tag] += 1
    
    # Convert to frequencies
    tag_frequencies = {tag: count / total_samples for tag, count in tag_counts.items()}
    
    # Add zero frequencies for missing tags
    for tag in tag_vocabulary:
        if tag not in tag_frequencies:
            tag_frequencies[tag] = 0.0
    
    return tag_frequencies


def log_tag_distribution_stats(tag_frequencies: Dict[str, float], prefix: str = ""):
    """
    Log statistics about tag distribution.
    
    Args:
        tag_frequencies: Dictionary mapping tag names to frequencies
        prefix: Prefix for log messages
    """
    if not tag_frequencies:
        return
    
    frequencies = list(tag_frequencies.values())
    non_zero_frequencies = [f for f in frequencies if f > 0]
    
    logger.info(f"{prefix}Tag distribution statistics:")
    logger.info(f"  - Total tags: {len(tag_frequencies)}")
    logger.info(f"  - Tags present: {len(non_zero_frequencies)}")
    logger.info(f"  - Coverage: {len(non_zero_frequencies) / len(tag_frequencies) * 100:.1f}%")
    
    if non_zero_frequencies:
        logger.info(f"  - Frequency range: {min(non_zero_frequencies):.3f} - {max(non_zero_frequencies):.3f}")
        logger.info(f"  - Mean frequency: {np.mean(non_zero_frequencies):.3f}")
        logger.info(f"  - Std frequency: {np.std(non_zero_frequencies):.3f}")

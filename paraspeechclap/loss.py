import torch
import torch.nn as nn
import torch.nn.functional as F
from paraspeechclap.debug_utils import logger, debug_tensor


class ClipLoss(nn.Module):
    """Calculates the contrastive loss between text and audio embeddings."""

    def __init__(self):
        """Initializes the simplified ClipLoss module."""
        super().__init__()
        # Removed all constructor arguments and attributes like cache_labels, etc.
        logger.debug("ClipLoss initialized (ultra-simplified)")

    def forward(self, text_features, audio_features, logit_scale):
        """Calculates the symmetric contrastive loss.

        Args:
            text_features (torch.Tensor): Text embeddings (batch_size, embed_dim).
            audio_features (torch.Tensor): Audio embeddings (batch_size, embed_dim).
            logit_scale (torch.Tensor): Scalar logit scale parameter.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        batch_size = audio_features.size(0)
        assert text_features.size(0) == batch_size, f"Batch size mismatch: text {text_features.size(0)} vs audio {batch_size}"

        # Normalize embeddings
        text_features = F.normalize(text_features, p=2, dim=1)
        audio_features = F.normalize(audio_features, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = logit_scale * audio_features @ text_features.T

        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size).to(sim_matrix.device)

        # Compute symmetric loss
        loss_a2t = F.cross_entropy(sim_matrix, labels)  # Audio-to-Text
        loss_t2a = F.cross_entropy(sim_matrix.T, labels)  # Text-to-Audio

        total_loss = (loss_a2t + loss_t2a) / 2.0

        return total_loss


class MultiTaskLoss(nn.Module):
    """Calculates combined contrastive and classification losses for multi-task learning."""

    def __init__(self):
        """Initializes the multi-task loss module."""
        super().__init__()
        self.contrastive_loss = ClipLoss()
        logger.debug("MultiTaskLoss initialized")

    def forward(self, text_features, audio_features, logit_scale, 
                tag_embeddings=None, rich_tags_batch=None, tag_to_idx=None,
                classification_heads=None):
        """
        Calculates the combined contrastive and classification loss.

        Args:
            text_features (torch.Tensor): Text embeddings (batch_size, embed_dim).
            audio_features (torch.Tensor): Audio embeddings (batch_size, embed_dim).
            logit_scale (torch.Tensor): Scalar logit scale parameter.
            tag_embeddings (torch.Tensor, optional): Tag embeddings (num_tags, embed_dim) for single-head mode.
            rich_tags_batch (List[List[str]], optional): Ground truth tags for single-head mode.
            tag_to_idx (dict[str,int], optional): Tag-to-index mapping for single-head mode.
            classification_heads (List[dict], optional): Multi-head mode. Each dict must contain:
                - 'tag_embeddings' (Tensor[num_tags, embed_dim])
                - 'rich_tags_batch' (List[List[str]]) per-sample tag lists for this head
                - 'tag_to_idx' (dict[str,int]) mapping for this head
                - 'name' (str) optional, for logging/debugging

        Returns:
            dict: Dictionary containing:
                - 'total_loss': Combined loss
                - 'contrastive_loss': Contrastive loss component
                - 'classification_loss': Classification loss component (0 if not computed)
        """
        # Always compute contrastive loss
        contrastive_loss = self.contrastive_loss(text_features, audio_features, logit_scale)
        
        classification_loss = torch.tensor(0.0, device=audio_features.device)
        
        # Normalize inputs: allow either a single head via tag_embeddings/rich_tags_batch
        # or a list of heads via classification_heads=[{tag_embeddings, rich_tags_batch, tag_to_idx, name}]
        heads_to_use = []
        if classification_heads is not None and isinstance(classification_heads, list) and len(classification_heads) > 0:
            heads_to_use = classification_heads
        elif tag_embeddings is not None and rich_tags_batch is not None and tag_to_idx is not None:
            heads_to_use = [{
                'tag_embeddings': tag_embeddings,
                'rich_tags_batch': rich_tags_batch,
                'tag_to_idx': tag_to_idx,
                'name': 'default'
            }]
        
        # Sum classification loss across heads (if any)
        if heads_to_use:
            per_head_losses = []
            for head in heads_to_use:
                head_loss = self._compute_classification_loss(
                    audio_features,
                    head['tag_embeddings'],
                    head['rich_tags_batch'],
                    head['tag_to_idx']
                )
                per_head_losses.append(head_loss)
            if len(per_head_losses) > 0:
                classification_loss = torch.stack(per_head_losses).mean()
        
        # Combine losses
        total_loss = contrastive_loss + classification_loss
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss
        }

    def _compute_classification_loss(self, audio_features, tag_embeddings, rich_tags_batch, tag_to_idx):
        """
        Computes the multi-label classification loss using tag embeddings as a dynamic classifier.
        
        Args:
            audio_features (torch.Tensor): Audio embeddings (batch_size, embed_dim).
            tag_embeddings (torch.Tensor): Tag embeddings (num_tags, embed_dim).
            rich_tags_batch (List[List[str]]): Ground truth tags for each sample.
            tag_to_idx (dict[str,int]): Mapping from tag string to class index for this head.
            
        Returns:
            torch.Tensor: Classification loss.
        """
        batch_size = audio_features.size(0)
        num_tags = tag_embeddings.size(0)
        
        # Normalize embeddings
        audio_features_norm = F.normalize(audio_features, p=2, dim=1)
        tag_embeddings_norm = F.normalize(tag_embeddings, p=2, dim=1)
        
        # Compute classification logits: (batch_size, num_tags)
        classification_logits = audio_features_norm @ tag_embeddings_norm.T
        
        # Create multi-label targets and track which samples have valid tags
        targets = torch.zeros(batch_size, num_tags, device=audio_features.device)
        valid_indices = []

        for i, sample_tags in enumerate(rich_tags_batch):
            if sample_tags is not None and len(sample_tags) > 0:
                valid_indices.append(i)
                for tag in sample_tags:
                    assert tag in tag_to_idx, f"Tag {tag} not found in tag_to_idx"
                    tag_idx = tag_to_idx[tag]
                    targets[i, tag_idx] = 1.0

        # If no samples in the batch have tags, classification contributes 0 loss
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=audio_features.device)

        # Compute BCE only over valid samples
        classification_loss = F.binary_cross_entropy_with_logits(
            classification_logits[valid_indices], targets[valid_indices], reduction='mean'
        )

        debug_tensor("Classification logits", classification_logits[valid_indices])
        debug_tensor("Classification targets", targets[valid_indices])
        logger.debug(f"Classification loss: {classification_loss.item()}")

        return classification_loss


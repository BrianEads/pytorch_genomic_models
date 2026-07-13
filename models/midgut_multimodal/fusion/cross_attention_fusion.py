"""Multi-modal fusion heads for tower embedding aggregation."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion over per-modality CLS embeddings (Option B).

    Each modality CLS token attends to all other modality tokens, enabling
    per-sample weighting of discriminative signals.

    Args:
        d_model: Input embedding dimension from each tower.
        d_fusion: Output fused embedding dimension.
        n_heads: Number of attention heads.
        dropout: Dropout inside the attention block.

    Input:
        embeddings: List of tensors, each shape ``(B, d_model)``.

    Output:
        Fused embedding ``(B, d_fusion)``.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_fusion: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_fusion),
            nn.ReLU(),
            nn.LayerNorm(d_fusion),
        )

    def forward(
        self,
        embeddings: List[torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse a list of tower CLS embeddings.

        Args:
            embeddings: One embedding tensor per available modality.
            return_attention: If True, also return attention weights.

        Returns:
            Fused embedding, optionally with attention weights ``(B, M, M)``.
        """
        if not embeddings:
            raise ValueError(
                "CrossAttentionFusion requires at least one modality embedding."
            )
        # Stack modalities: (B, M, d_model) where M = number of modalities.
        stacked = torch.stack(embeddings, dim=1)
        attended, attn_weights = self.attention(stacked, stacked, stacked, need_weights=True)
        fused_token = self.norm(attended.mean(dim=1))
        fused = self.projection(fused_token)
        if return_attention:
            return fused, attn_weights
        return fused


class LateFusionHead(nn.Module):
    """Late fusion baseline that concatenates CLS tokens (Option A).

    Supports incremental manifests: only available modality embeddings are
    concatenated. ``max_modalities`` sets the upper bound for parameter sizing
    when using :class:`torch.nn.LazyLinear`.

    Args:
        max_modalities: Maximum expected modalities (for lazy init sizing hint).
        d_model: Per-modality embedding dimension.
        d_fusion: Output fused embedding dimension.
        dropout: Dropout in the MLP head.

    Input:
        embeddings: List of tensors, each shape ``(B, d_model)``.

    Output:
        Fused embedding ``(B, d_fusion)``.
    """

    def __init__(
        self,
        max_modalities: int = 5,
        d_model: int = 256,
        d_fusion: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_modalities = max_modalities
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.LazyLinear(d_fusion * 2)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_fusion * 2, d_fusion),
            nn.LayerNorm(d_fusion),
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate available modality embeddings and project.

        Args:
            embeddings: One embedding tensor per *available* modality. Length
                may be 1–``max_modalities`` for incremental training.

        Returns:
            Fused embedding of shape ``(B, d_fusion)``.

        Raises:
            ValueError: If no embeddings are provided.
        """
        if not embeddings:
            raise ValueError("LateFusionHead requires at least one modality embedding.")
        concatenated = torch.cat(embeddings, dim=-1)
        hidden = self.dropout(self.projection(concatenated).relu())
        return self.output(hidden)

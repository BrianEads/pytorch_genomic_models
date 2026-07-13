"""Bt biochemical screening MLP tower."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BtScreeningMLPTower(nn.Module):
    """Three-layer MLP over protein and compound embeddings.

    Args:
        protein_dim: Dimension of the protein (ESM-2) embedding input.
        compound_dim: Dimension of the compound fingerprint input.
        d_model: Output tower embedding dimension.
        dropout: Dropout probability between MLP layers.

    Input shapes:
        protein_embedding: ``(B, protein_dim)``.
        compound_embedding: ``(B, compound_dim)``.

    Output:
        Tuple of embedding ``(B, d_model)`` and ``None``.
    """

    def __init__(
        self,
        protein_dim: int = 1280,
        compound_dim: int = 2048,
        d_model: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = protein_dim + compound_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        protein_embedding: torch.Tensor,
        compound_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse protein and compound representations.

        Args:
            protein_embedding: Pre-computed protein vector, shape ``(B, protein_dim)``.
            compound_embedding: Compound fingerprint, shape ``(B, compound_dim)``.

        Returns:
            Tower embedding and ``None`` for attention weights.
        """
        fused = torch.cat([protein_embedding, compound_embedding], dim=-1)
        embedding = self.mlp(fused)
        return embedding, None

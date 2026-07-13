"""Resistance locus binary classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResistanceClassifierHead(nn.Module):
    """Classify genomic windows as resistance-associated or neutral.

    Args:
        d_fusion: Input fused embedding dimension.
        label_smoothing: Label smoothing epsilon for BCE loss.

    Input shape:
        fused_embedding: ``(B, d_fusion)``.

    Output shape:
        logits: ``(B, 1)`` unnormalised binary logits.
    """

    def __init__(self, d_fusion: int = 256, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.head = nn.Sequential(
            nn.Linear(d_fusion, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Run binary resistance classification."""
        return self.head(fused_embedding)

    def loss_fn(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss with label smoothing."""
        smoothed = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, smoothed)

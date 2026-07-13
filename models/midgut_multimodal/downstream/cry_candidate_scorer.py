"""Novel CRY protein candidate scoring head."""

from __future__ import annotations

import torch
import torch.nn as nn


class CryCandidateScorerHead(nn.Module):
    """Multi-label head for insecticidal activity and target-order prediction.

    Args:
        d_fusion: Input fused embedding dimension.
        n_labels: Number of output labels (activity + target orders).

    Input shape:
        fused_embedding: ``(B, d_fusion)``.

    Output shape:
        logits: ``(B, n_labels)`` multi-label logits.
    """

    def __init__(self, d_fusion: int = 256, n_labels: int = 4) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_fusion, 128),
            nn.ReLU(),
            nn.Linear(128, n_labels),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Score CRY protein candidates."""
        return self.head(fused_embedding)

    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-label cross-entropy (BCE-with-logits)."""
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

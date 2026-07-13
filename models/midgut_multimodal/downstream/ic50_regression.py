"""IC50 regression downstream head."""

from __future__ import annotations

import torch
import torch.nn as nn


class IC50RegressionHead(nn.Module):
    """Predict log10(IC50) from a fused multi-modal embedding.

    Args:
        d_fusion: Input fused embedding dimension.

    Input shape:
        fused_embedding: ``(B, d_fusion)``.

    Output shape:
        predictions: ``(B, 1)`` log10(IC50) in µg/mL.
    """

    def __init__(self, d_fusion: int = 256) -> None:
        super().__init__()
        self.head = nn.Linear(d_fusion, 1)

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Run IC50 regression."""
        return self.head(fused_embedding)

    @staticmethod
    def loss_fn(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss on log-transformed IC50 values."""
        return nn.functional.mse_loss(predictions, targets)

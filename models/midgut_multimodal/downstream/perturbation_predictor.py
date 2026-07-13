"""Midgut cell-type perturbation prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerturbationPredictorHead(nn.Module):
    """Predict post-perturbation cell-type distribution shifts.

    Args:
        d_fusion: Input fused embedding dimension.
        n_cell_types: Number of cell-type bins in the output distribution.

    Input shape:
        fused_embedding: ``(B, d_fusion)``.

    Output shape:
        log_probs: ``(B, n_cell_types)`` log-probabilities over cell types.
    """

    def __init__(self, d_fusion: int = 256, n_cell_types: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_fusion, 128),
            nn.ReLU(),
            nn.Linear(128, n_cell_types),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Predict perturbed cell-type log-distribution."""
        return nn.functional.log_softmax(self.head(fused_embedding), dim=-1)

    @staticmethod
    def loss_fn(
        log_probs: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL-divergence against an observed cell-type distribution."""
        return nn.functional.kl_div(
            log_probs,
            target_distribution,
            reduction="batchmean",
        )

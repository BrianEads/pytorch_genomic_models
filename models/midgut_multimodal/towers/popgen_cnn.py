"""Population genomics CNN tower for allele-frequency windows."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PopGenCNNTower(nn.Module):
    """1D CNN over allele-frequency windows with summary-statistics fusion.

    Args:
        d_model: Output embedding dimension.
        n_alleles: Number of allele channels per genomic window.
        hidden_channels: Convolution channel width.

    Input shape:
        allele_freqs: ``(B, W_window, N_alleles)`` allele frequencies per window.

    Output:
        Tuple of pooled embedding ``(B, d_model)`` and optional window-level
        attention proxy ``(B, W_window)``.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_alleles: int = 4,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.conv_stack = nn.Sequential(
            nn.Conv1d(n_alleles, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(n_alleles * 3, d_model),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels + d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, allele_freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode population-genomics windows.

        Args:
            allele_freqs: Allele frequency tensor, shape ``(B, W_window, N_alleles)``.

        Returns:
            Window-pooled embedding and optional attention weights over windows.
        """
        # Conv1d expects (B, channels, length).
        conv_input = allele_freqs.transpose(1, 2)
        conv_out = self.conv_stack(conv_input)
        pooled = conv_out.mean(dim=-1)

        mean_stats = allele_freqs.mean(dim=1)
        var_stats = allele_freqs.var(dim=1, unbiased=False)
        max_stats = allele_freqs.max(dim=1).values
        summary = torch.cat([mean_stats, var_stats, max_stats], dim=-1)
        summary_embed = self.summary_mlp(summary)

        embedding = self.fusion(torch.cat([pooled, summary_embed], dim=-1))

        attention_weights = torch.softmax(conv_out.mean(dim=1), dim=-1)
        return embedding, attention_weights

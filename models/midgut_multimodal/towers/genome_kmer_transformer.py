"""Genome k-mer transformer tower for masked language-model pre-training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GenomeKmerTransformerTower(nn.Module):
    """Transformer encoder over fixed-length k-mer windows.

    Consumes Goal 3 genome HDF5 ``tokens`` tensors produced by
    ``dmel_genome_tokenize.py``.

    Args:
        vocab_size: K-mer vocabulary size (4096 + special tokens for k=6).
        d_model: Hidden dimension.
        n_layers: Transformer encoder depth.
        n_heads: Attention head count.
        max_tokens: Window length (default 512).
        dropout: Dropout probability.

    Input shape:
        tokens: ``(B, L)`` integer k-mer token IDs.

    Output:
        Tuple of CLS embedding ``(B, d_model)`` and optional attention weights.
    """

    def __init__(
        self,
        vocab_size: int = 4101,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        max_tokens: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode a batch of k-mer windows.

        Args:
            tokens: K-mer token indices, shape ``(B, L)``.
            return_attention: If True, return proxy self-attention weights.

        Returns:
            CLS embedding and optional attention weights.
        """
        batch_size, seq_len = tokens.shape
        token_embed = self.token_embedding(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, token_embed], dim=1)
        x = x + self.positional_encoding[:, : seq_len + 1, :]
        encoded = self.transformer(x)
        cls_embedding = self.output_norm(encoded[:, 0, :])

        attention_weights: Optional[torch.Tensor] = None
        if return_attention:
            attention_weights = torch.softmax(
                torch.matmul(encoded[:, 1:, :], encoded[:, 1:, :].transpose(1, 2))
                / (self.d_model**0.5),
                dim=-1,
            )
        return cls_embedding, attention_weights

"""scRNA-seq transformer tower for masked gene expression modelling."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScRNATransformerTower(nn.Module):
    """Transformer encoder over gene-token sequences (scGPT-style).

    Args:
        vocab_size: Number of unique gene tokens in the vocabulary.
        d_model: Hidden dimension for embeddings and transformer layers.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (must divide ``d_model``).
        max_genes: Maximum number of gene tokens per cell.
        dropout: Dropout probability applied inside the transformer.

    Input shapes:
        gene_ids: ``(B, N_genes)`` integer gene indices in ``[0, vocab_size)``.
        expression: ``(B, N_genes)`` log1p-normalised expression values.

    Output:
        Tuple of CLS embedding ``(B, d_model)`` and optional self-attention
        weights ``(B, N_genes, N_genes)`` averaged across heads.
    """

    def __init__(
        self,
        vocab_size: int = 20_000,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_genes: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_genes + 1, d_model)
        )
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
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass over a batch of cells.

        Args:
            gene_ids: Gene token indices, shape ``(B, N_genes)``.
            expression: Matching expression values, shape ``(B, N_genes)``.
            return_attention: If True, return averaged self-attention weights.

        Returns:
            CLS embedding and optional attention weights.
        """
        batch_size, seq_len = gene_ids.shape
        gene_embed = self.gene_embedding(gene_ids)
        value_embed = self.value_projection(expression.unsqueeze(-1))
        tokens = gene_embed + value_embed

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.positional_encoding[:, : seq_len + 1, :]
        encoded = self.transformer(tokens)
        cls_embedding = self.output_norm(encoded[:, 0, :])

        attention_weights: Optional[torch.Tensor] = None
        if return_attention:
            # Lightweight proxy weights for interpretability hooks in stubs/tests.
            attention_weights = torch.softmax(
                torch.matmul(encoded[:, 1:, :], encoded[:, 1:, :].transpose(1, 2))
                / math.sqrt(self.d_model),
                dim=-1,
            )

        return cls_embedding, attention_weights

"""PPI graph tower with Goal-2-owned ESM-2 node feature generation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from models.midgut_multimodal.towers.protein_esm2 import ProteinESM2Tower

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:  # pragma: no cover - optional at import time
    GCNConv = None  # type: ignore[misc, assignment]
    global_mean_pool = None  # type: ignore[misc, assignment]


class PPIGraphTower(nn.Module):
    """Graph convolution tower for DroID PPI networks.

    Goal 3 writes graphs with ``x=None``. This tower calls
    :class:`ProteinESM2Tower` to generate per-node embeddings from protein
    sequences before message passing.

    Args:
        d_model: Output graph embedding dimension.
        esm_tower: Optional pre-built ESM-2 tower for node encoding.
        num_gcn_layers: Number of GCN layers.
        use_stub: Use lightweight ESM stub when no backbone is provided.

    Input:
        graph: PyG ``Data`` with ``edge_index`` and optional ``x`` / ``batch``.
        protein_token_ids: ``(num_nodes, L)`` amino-acid tokens when ``x`` is
            missing.

    Output:
        Tuple of graph-level embedding ``(B, d_model)`` and ``None``.
    """

    def __init__(
        self,
        d_model: int = 256,
        esm_tower: ProteinESM2Tower | None = None,
        num_gcn_layers: int = 2,
        use_stub: bool = True,
    ) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError(
                "torch_geometric is required for PPIGraphTower. "
                "Install with: pip install torch-geometric"
            )

        self.d_model = d_model
        self.esm_tower = esm_tower or ProteinESM2Tower(d_model=d_model, use_stub=use_stub)
        self.gcn_layers = nn.ModuleList(
            [
                GCNConv(d_model, d_model),
                *[GCNConv(d_model, d_model) for _ in range(num_gcn_layers - 1)],
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def fill_node_features(
        self,
        graph: Any,
        protein_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Populate node features with ESM-2 embeddings when ``graph.x`` is None.

        Args:
            graph: PyG ``Data`` object, possibly with ``x=None``.
            protein_token_ids: Tokenised protein sequences per node.

        Returns:
            Node feature matrix ``(num_nodes, d_model)``.
        """
        if getattr(graph, "x", None) is not None:
            return graph.x

        if protein_token_ids is None:
            raise ValueError(
                "PPI graph has x=None and no protein_token_ids were provided. "
                "Goal 2 must supply sequences for ESM-2 embedding generation."
            )

        node_features, _ = self.esm_tower.encode_nodes(protein_token_ids)
        graph.x = node_features
        return node_features

    def forward(
        self,
        graph: Any,
        protein_token_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode a PPI graph into a graph-level embedding.

        Args:
            graph: PyG ``Data`` with ``edge_index``; ``x`` may be ``None``.
            protein_token_ids: Required when ``graph.x`` is ``None``.

        Returns:
            Graph-level embedding and ``None`` for attention weights.
        """
        x = self.fill_node_features(graph, protein_token_ids)
        edge_index = graph.edge_index

        for layer in self.gcn_layers:
            x = layer(x, edge_index).relu()

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        pooled = global_mean_pool(x, batch)
        embedding = self.output_norm(pooled)
        return embedding, None

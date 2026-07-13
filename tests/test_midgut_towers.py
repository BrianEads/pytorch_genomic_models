"""Unit tests for midgut multi-modal tower forward passes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.midgut_multimodal.fusion.cross_attention_fusion import (
    CrossAttentionFusion,
    LateFusionHead,
)
from models.midgut_multimodal.towers.bt_screening_mlp import BtScreeningMLPTower
from models.midgut_multimodal.towers.cell_painting_resnet import CellPaintingResNetTower
from models.midgut_multimodal.towers.genome_kmer_transformer import GenomeKmerTransformerTower
from models.midgut_multimodal.towers.popgen_cnn import PopGenCNNTower
from models.midgut_multimodal.towers.protein_esm2 import ProteinESM2Tower
from models.midgut_multimodal.towers.scrna_transformer import ScRNATransformerTower

D_MODEL = 256
BATCH = 4


@pytest.mark.parametrize(
    "tower_factory,inputs",
    [
        (
            lambda: ScRNATransformerTower(d_model=D_MODEL, n_layers=2, n_heads=8),
            lambda: (
                torch.randint(0, 1000, (BATCH, 128)),
                torch.rand(BATCH, 128),
            ),
        ),
        (
            lambda: CellPaintingResNetTower(d_model=D_MODEL, pretrained=False),
            lambda: (torch.rand(BATCH, 5, 224, 224),),
        ),
        (
            lambda: PopGenCNNTower(d_model=D_MODEL, n_alleles=4),
            lambda: (torch.rand(BATCH, 64, 4),),
        ),
        (
            lambda: BtScreeningMLPTower(d_model=D_MODEL),
            lambda: (torch.rand(BATCH, 1280), torch.rand(BATCH, 2048)),
        ),
        (
            lambda: GenomeKmerTransformerTower(d_model=D_MODEL, n_layers=2, n_heads=8),
            lambda: (torch.randint(0, 100, (BATCH, 128)),),
        ),
        (
            lambda: ProteinESM2Tower(d_model=D_MODEL, use_stub=True),
            lambda: (torch.randint(0, 20, (BATCH, 64)),),
        ),
    ],
)
def test_tower_forward_shape(tower_factory, inputs) -> None:
    """Each tower should return an embedding of shape (B, d_model)."""
    tower = tower_factory()
    tower.eval()
    with torch.no_grad():
        embedding, _ = tower(*inputs())
    assert embedding.shape == (BATCH, D_MODEL)


def test_cross_attention_fusion_shape() -> None:
    """Cross-attention fusion should return (B, d_fusion)."""
    fusion = CrossAttentionFusion(d_model=D_MODEL, d_fusion=D_MODEL)
    embeddings = [torch.randn(BATCH, D_MODEL) for _ in range(5)]
    fused = fusion(embeddings)
    assert fused.shape == (BATCH, D_MODEL)


def test_late_fusion_shape() -> None:
    """Late fusion head should return (B, d_fusion)."""
    fusion = LateFusionHead(max_modalities=5, d_model=D_MODEL, d_fusion=D_MODEL)
    embeddings = [torch.randn(BATCH, D_MODEL) for _ in range(5)]
    fused = fusion(embeddings)
    assert fused.shape == (BATCH, D_MODEL)


def test_ppi_graph_tower_with_esm_fill() -> None:
    """PPIGraphTower should fill x=None graphs via ESM-2 node encoding."""
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from models.midgut_multimodal.towers.ppi_graph_tower import PPIGraphTower

    tower = PPIGraphTower(d_model=D_MODEL, use_stub=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graph = Data(edge_index=edge_index, x=None)
    token_ids = torch.randint(0, 20, (3, 32))
    embedding, _ = tower(graph, protein_token_ids=token_ids)
    assert embedding.shape == (1, D_MODEL)

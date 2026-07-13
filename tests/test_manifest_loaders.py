"""Tests for manifest-driven loaders and incremental modality support."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

h5py = pytest.importorskip("h5py")

from models.midgut_multimodal.data.datasets import BulkRNASeqDataset, GenomeKmerDataset
from models.midgut_multimodal.data.factory import build_manifest_loaders
from models.midgut_multimodal.data.manifest import load_manifest
from models.midgut_multimodal.fusion.cross_attention_fusion import CrossAttentionFusion, LateFusionHead
from models.midgut_multimodal.towers.protein_esm2 import ProteinESM2Tower

D_MODEL = 256
BATCH = 4


def _write_manifest(path: Path, modalities: dict) -> None:
    payload = {
        "schema_version": "1.0",
        "project": "test",
        "organism": "dmel",
        "assembly": "dm6",
        "modalities": modalities,
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_genome_h5(path: Path, n: int = 8, window: int = 16) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tokens", data=np.random.randint(0, 100, size=(n, window), dtype=np.int16))
        h5.create_dataset("chromosome", data=np.array(["chr2L"] * n, dtype="S8"))
        h5.create_dataset("start", data=np.arange(n, dtype=np.int32))
        h5.create_dataset("end", data=np.arange(window, window + n, dtype=np.int32))


def _write_rnaseq_h5(
    path: Path,
    tissues: list[str],
    n_genes: int = 32,
) -> None:
    n_samples = len(tissues)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "expression",
            data=np.random.rand(n_samples, n_genes).astype(np.float32),
        )
        h5.create_dataset("gene_ids", data=np.arange(n_genes, dtype=np.int64))
        h5.create_dataset(
            "sample_ids",
            data=np.array([f"sample_{i}" for i in range(n_samples)], dtype="S16"),
        )
        h5.create_dataset(
            "sample_tissues",
            data=np.array(tissues, dtype="S16"),
        )


def test_load_manifest_partial_v1(tmp_path: Path) -> None:
    """Manifest parsing should report missing materialised outputs."""
    genome_h5 = tmp_path / "genome.h5"
    _write_genome_h5(genome_h5)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "genome": {
                "source": "flybase",
                "format": "fasta",
                "output_path": str(genome_h5),
            },
            "rnaseq": {
                "source": "modencode",
                "format": "tsv_tpm",
                "output_path": str(tmp_path / "missing_rnaseq.h5"),
            },
        },
    )
    ctx = load_manifest(manifest_path, repo_root=tmp_path)
    assert ctx.available_modalities() == ["genome"]
    assert "rnaseq" in ctx.missing_modalities(("genome", "rnaseq", "ppi"))


def test_build_loaders_skips_missing_modalities(tmp_path: Path) -> None:
    """Loader factory should skip unavailable modalities without strict mode."""
    genome_h5 = tmp_path / "genome.h5"
    _write_genome_h5(genome_h5)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {
            "genome": {
                "source": "flybase",
                "format": "fasta",
                "output_path": str(genome_h5),
            },
        },
    )
    _, loaders = build_manifest_loaders(str(manifest_path), batch_size=2, tissue_filter=None)
    assert loaders.available() == ["genome"]
    assert loaders.rnaseq is None
    batch = next(iter(loaders.genome))
    assert batch["tokens"].shape == (2, 16)


def test_midgut_tissue_filter(tmp_path: Path) -> None:
    """RNA-seq loader should retain only midgut-labelled samples by default."""
    rnaseq_h5 = tmp_path / "rnaseq.h5"
    _write_rnaseq_h5(rnaseq_h5, ["midgut", "brain", "foregut", "fat_body"])
    dataset = BulkRNASeqDataset(rnaseq_h5, tissue_filter="midgut")
    assert len(dataset) == 2


def test_protein_esm2_fill_graph_node_features() -> None:
    """ProteinESM2Tower should populate graph.x when it is None."""
    tower = ProteinESM2Tower(d_model=D_MODEL, use_stub=True)
    graph = SimpleNamespace(x=None)
    token_ids = torch.randint(0, 20, (8, 32))
    features = tower.fill_graph_node_features(graph, token_ids)
    assert features.shape == (8, D_MODEL)
    assert graph.x is not None
    assert torch.equal(graph.x, features)


def test_cross_attention_partial_modalities() -> None:
    """Fusion should accept 1–N modality embeddings for incremental training."""
    fusion = CrossAttentionFusion(d_model=D_MODEL, d_fusion=D_MODEL)
    for count in (1, 2, 3):
        embeddings = [torch.randn(BATCH, D_MODEL) for _ in range(count)]
        fused = fusion(embeddings)
        assert fused.shape == (BATCH, D_MODEL)


def test_late_fusion_partial_modalities() -> None:
    """LateFusionHead should adapt to variable modality counts."""
    fusion = LateFusionHead(max_modalities=5, d_model=D_MODEL, d_fusion=D_MODEL)
    for count in (1, 3, 5):
        embeddings = [torch.randn(BATCH, D_MODEL) for _ in range(count)]
        fused = fusion(embeddings)
        assert fused.shape == (BATCH, D_MODEL)


def test_genome_dataset_reads_h5(tmp_path: Path) -> None:
    """GenomeKmerDataset should read tokens from Goal 3 HDF5 layout."""
    genome_h5 = tmp_path / "genome.h5"
    _write_genome_h5(genome_h5, n=4)
    dataset = GenomeKmerDataset(genome_h5)
    item = dataset[0]
    assert item["tokens"].dtype == torch.int64
    assert item["tokens"].numel() == 16

"""PyTorch datasets for manifest-backed v1 modalities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MIDGUT_TISSUE_ALIASES: frozenset[str] = frozenset(
    {"midgut", "gut", "intestine", "hindgut", "foregut", "cardia"}
)


class GenomeKmerDataset(Dataset):
    """HDF5 dataset of sliding-window genome k-mer tokens.

    Expects Goal 3 ``dmel_genome_tokenize.py`` output with datasets:
    ``tokens``, ``chromosome``, ``start``, ``end``.

    Args:
        h5_path: Path to tokenised genome HDF5.
        split_indices: Optional index array restricting rows to a split.
    """

    def __init__(self, h5_path: Path, split_indices: np.ndarray | None = None) -> None:
        self.h5_path = h5_path
        self._file: h5py.File | None = None
        with h5py.File(h5_path, "r") as probe:
            self.length = int(probe["tokens"].shape[0])
            self.window = int(probe["tokens"].shape[1])
        self.indices = split_indices if split_indices is not None else np.arange(self.length)

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        h5 = self._ensure_open()
        row = int(self.indices[idx])
        tokens = torch.from_numpy(np.asarray(h5["tokens"][row], dtype=np.int64))
        return {
            "tokens": tokens,
            "chromosome": h5["chromosome"][row].decode("utf-8"),
            "start": int(h5["start"][row]),
            "end": int(h5["end"][row]),
        }


class BulkRNASeqDataset(Dataset):
    """Bulk RNA-seq expression dataset with optional midgut tissue filtering.

    Expects Goal 3 ``modencode_rnaseq_qc.py`` HDF5 with datasets:
    ``expression`` (samples × genes), ``gene_ids``, ``sample_ids`` and optional
    ``sample_tissues`` or ``tissue`` per sample.

    When ``tissue_filter="midgut"``, only samples whose tissue label matches
    midgut aliases are retained (Goal 3 midgut-only subset alignment).

    Args:
        h5_path: Tokenised RNA-seq HDF5 path.
        tissue_filter: If set to ``"midgut"``, retain midgut-labelled samples.
        split: Optional split name filter when an HDF5 ``split`` dataset exists.
        max_genes: Cap active genes per sample (top by mean expression).
    """

    def __init__(
        self,
        h5_path: Path,
        tissue_filter: str | None = "midgut",
        split: str | None = None,
        max_genes: int = 2048,
    ) -> None:
        self.h5_path = h5_path
        self.max_genes = max_genes
        self._file: h5py.File | None = None

        with h5py.File(h5_path, "r") as probe:
            expression = np.asarray(probe["expression"])
            n_samples, n_genes = expression.shape
            sample_ids = _decode_strings(probe["sample_ids"][:])
            tissues = _load_tissues(probe, n_samples)

            sample_mask = np.ones(n_samples, dtype=bool)
            if tissue_filter == "midgut":
                if tissues is None:
                    logger.warning(
                        "No tissue metadata in %s; loading all %d samples.",
                        h5_path,
                        n_samples,
                    )
                else:
                    sample_mask = np.array(
                        [_is_midgut_tissue(t) for t in tissues], dtype=bool
                    )
                    logger.info(
                        "Midgut tissue filter retained %d / %d samples.",
                        int(sample_mask.sum()),
                        n_samples,
                    )

            if split is not None and "split" in probe:
                split_labels = _decode_strings(probe["split"][:])
                sample_mask &= np.array([label == split for label in split_labels], dtype=bool)

            self.sample_indices = np.flatnonzero(sample_mask)
            if self.sample_indices.size == 0:
                raise ValueError(f"No samples remain after filtering for {h5_path}")

            gene_ids = np.asarray(probe["gene_ids"][:], dtype=np.int64)
            mean_expr = expression[sample_mask].mean(axis=0)
            if n_genes > max_genes:
                top_idx = np.argpartition(-mean_expr, max_genes - 1)[:max_genes]
                top_idx.sort()
            else:
                top_idx = np.arange(n_genes)

            self.gene_ids = gene_ids[top_idx]
            self.expression = expression[:, top_idx]
            self.sample_ids = sample_ids

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_row = int(self.sample_indices[idx])
        expr = torch.from_numpy(
            np.asarray(self.expression[sample_row], dtype=np.float32)
        )
        gene_ids = torch.from_numpy(np.asarray(self.gene_ids, dtype=np.int64))
        return {
            "gene_ids": gene_ids,
            "expression": expr,
            "sample_id": self.sample_ids[sample_row],
        }


class PPIGraphDataset(Dataset):
    """Dataset wrapping a PyG PPI graph with ``x=None`` node features.

    Goal 3 writes ``data/tokenised/dmel_ppi_graph.pt`` with ``edge_index`` and
    ``gene_id``; node features are filled at training time by Goal 2 ESM-2.

    Each ``__getitem__`` returns the full graph (batch size 1). For larger graphs,
  use neighbour sampling in a future iteration.

    Args:
        graph_path: Path to the ``.pt`` graph file.
        protein_token_ids: Optional pre-tokenised sequences aligned to nodes.
    """

    def __init__(
        self,
        graph_path: Path,
        protein_token_ids: torch.Tensor | None = None,
    ) -> None:
        self.graph_path = graph_path
        self.graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        self.protein_token_ids = protein_token_ids

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "protein_token_ids": self.protein_token_ids,
        }


def _decode_strings(values: Any) -> list[str]:
    decoded: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _load_tissues(h5: h5py.File, n_samples: int) -> list[str] | None:
    if "sample_tissues" in h5:
        return _decode_strings(h5["sample_tissues"][:])
    if "tissue" in h5:
        return _decode_strings(h5["tissue"][:])
    if "sample_metadata" in h5:
        # JSON-encoded metadata blob per sample (future Goal 3 format).
        meta = h5["sample_metadata"][:]
        return [_extract_tissue(m) for m in meta]
    return None


def _extract_tissue(meta_item: Any) -> str:
    if isinstance(meta_item, bytes):
        meta_item = meta_item.decode("utf-8")
    if isinstance(meta_item, str):
        return meta_item
    if isinstance(meta_item, dict):
        return str(meta_item.get("tissue", ""))
    return ""


def _is_midgut_tissue(label: str) -> bool:
    normalised = label.strip().lower().replace(" ", "_")
    return any(alias in normalised for alias in MIDGUT_TISSUE_ALIASES)

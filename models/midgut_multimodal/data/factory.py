"""Build DataLoaders from a DatasetManifest with partial-modality support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from models.midgut_multimodal.data.datasets import (
    BulkRNASeqDataset,
    GenomeKmerDataset,
    PPIGraphDataset,
)
from models.midgut_multimodal.data.manifest import ManifestContext, V1_MODALITIES, load_manifest

logger = logging.getLogger(__name__)


@dataclass
class ModalityLoaders:
    """Container of optional per-modality DataLoaders."""

    genome: DataLoader | None = None
    rnaseq: DataLoader | None = None
    ppi: DataLoader | None = None
    scrna: DataLoader | None = None
    popgen: DataLoader | None = None
    cell_painting: DataLoader | None = None
    bt_screening: DataLoader | None = None
    protein: DataLoader | None = None

    def available(self) -> list[str]:
        """Return names of modalities with a non-None loader."""
        return [
            name
            for name in (
                "genome",
                "rnaseq",
                "ppi",
                "scrna",
                "popgen",
                "cell_painting",
                "bt_screening",
                "protein",
            )
            if getattr(self, name) is not None
        ]

    def get(self, modality: str) -> DataLoader | None:
        """Return the loader for a modality name, or None."""
        return getattr(self, modality, None)


def build_manifest_loaders(
    manifest_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    tissue_filter: str | None = "midgut",
    split: str | None = "train",
    required_modalities: tuple[str, ...] | None = None,
    strict: bool = False,
) -> tuple[ManifestContext, ModalityLoaders]:
    """Construct DataLoaders for every materialised manifest modality.

    Missing modalities are skipped with an info log unless ``strict=True``,
    in which case a ``FileNotFoundError`` is raised.

    Args:
        manifest_path: Path to DatasetManifest JSON.
        batch_size: Batch size for tabular/image loaders (PPI uses 1).
        num_workers: DataLoader worker count.
        tissue_filter: Tissue filter for bulk RNA-seq (``"midgut"`` or None).
        split: Optional split filter when split indices exist in HDF5.
        required_modalities: If set, these must be materialised or an error is raised.
        strict: Raise when any declared modality lacks tokenised output.

    Returns:
        Tuple of parsed manifest context and :class:`ModalityLoaders`.
    """
    manifest = load_manifest(manifest_path)
    loaders = ModalityLoaders()
    builders: dict[str, Callable[[], Dataset | None]] = {
        "genome": lambda: _maybe_genome(manifest, strict),
        "rnaseq": lambda: _maybe_rnaseq(manifest, tissue_filter, split, strict),
        "ppi": lambda: _maybe_ppi(manifest, strict),
    }

    for modality, builder in builders.items():
        spec = manifest.get(modality)
        if spec is None:
            continue
        if not spec.is_materialised:
            message = f"Modality '{modality}' declared but output missing: {spec.output_path}"
            if strict:
                raise FileNotFoundError(message)
            logger.info("Skipping unavailable modality: %s", modality)
            continue
        dataset = builder()
        if dataset is None:
            continue
        collate_fn = _collate_ppi if modality == "ppi" else None
        effective_batch = 1 if modality == "ppi" else batch_size
        setattr(
            loaders,
            modality,
            DataLoader(
                dataset,
                batch_size=effective_batch,
                shuffle=modality != "ppi",
                num_workers=num_workers,
                collate_fn=collate_fn,
            ),
        )

    if required_modalities:
        missing = [m for m in required_modalities if getattr(loaders, m) is None]
        if missing:
            raise FileNotFoundError(
                f"Required modalities not available: {missing}. "
                f"Materialised: {manifest.available_modalities()}"
            )

    available = loaders.available()
    logger.info(
        "Manifest loaders ready for %s (v1 expects %s; missing %s).",
        available,
        list(V1_MODALITIES),
        manifest.missing_modalities(),
    )
    return manifest, loaders


def _maybe_genome(manifest: ManifestContext, strict: bool) -> Dataset | None:
    spec = manifest.get("genome")
    if spec is None or spec.output_path is None:
        return None
    if not spec.output_path.is_file():
        if strict:
            raise FileNotFoundError(spec.output_path)
        return None
    return GenomeKmerDataset(spec.output_path)


def _maybe_rnaseq(
    manifest: ManifestContext,
    tissue_filter: str | None,
    split: str | None,
    strict: bool,
) -> Dataset | None:
    spec = manifest.get("rnaseq")
    if spec is None or spec.output_path is None:
        return None
    if not spec.output_path.is_file():
        if strict:
            raise FileNotFoundError(spec.output_path)
        return None
    return BulkRNASeqDataset(
        spec.output_path,
        tissue_filter=tissue_filter,
        split=split,
    )


def _maybe_ppi(manifest: ManifestContext, strict: bool) -> Dataset | None:
    spec = manifest.get("ppi")
    if spec is None or spec.output_path is None:
        return None
    if not spec.output_path.is_file():
        if strict:
            raise FileNotFoundError(spec.output_path)
        return None
    return PPIGraphDataset(spec.output_path)


def _collate_ppi(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """PPI graphs are returned one at a time (no stacking)."""
    return batch[0]

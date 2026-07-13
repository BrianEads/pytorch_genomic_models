"""DatasetManifest parsing and modality availability helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Maps manifest modality keys (Goal 3) to Goal 2 tower identifiers.
MANIFEST_TO_TOWER: dict[str, str] = {
    "genome": "genome_kmer_transformer",
    "rnaseq": "scrna_transformer",
    "ppi": "ppi_graph",
    "scrna": "scrna_transformer",
    "popgen": "popgen_cnn",
    "cell_painting": "cell_painting_resnet",
    "bt_screening": "bt_screening_mlp",
    "protein": "protein_esm2",
}

# v1 incremental manifest ships these modalities first.
V1_MODALITIES: tuple[str, ...] = ("genome", "rnaseq", "ppi")


@dataclass(frozen=True)
class ModalitySpec:
    """Resolved modality entry from a DatasetManifest."""

    name: str
    source: str
    format: str
    output_path: Path | None
    raw_path: Path | None
    tokeniser: str | None
    normalise: str | None
    extra: dict[str, Any]

    @property
    def tower_id(self) -> str:
        """Return the Goal 2 tower identifier for this modality."""
        return MANIFEST_TO_TOWER.get(self.name, self.name)

    @property
    def is_materialised(self) -> bool:
        """Return True when tokenised output exists on disk."""
        return self.output_path is not None and self.output_path.is_file()


@dataclass(frozen=True)
class ManifestContext:
    """Parsed DatasetManifest with convenience accessors."""

    path: Path
    schema_version: str
    organism: str
    assembly: str
    modalities: dict[str, ModalitySpec]
    splits: dict[str, float]
    split_strategy: str | None
    fetch_recipe: str | None

    def available_modalities(self) -> list[str]:
        """Return modality names whose tokenised outputs exist."""
        return [name for name, spec in self.modalities.items() if spec.is_materialised]

    def missing_modalities(self, expected: tuple[str, ...] = V1_MODALITIES) -> list[str]:
        """Return expected modalities not yet materialised on disk."""
        available = set(self.available_modalities())
        return [name for name in expected if name not in available]

    def get(self, modality: str) -> ModalitySpec | None:
        """Return a modality spec if declared in the manifest."""
        return self.modalities.get(modality)


def load_manifest(manifest_path: str | Path, repo_root: Path | None = None) -> ManifestContext:
    """Load and parse a DatasetManifest JSON file.

    Args:
        manifest_path: Path to the manifest JSON (absolute or repo-relative).
        repo_root: Optional repository root for resolving relative paths.

    Returns:
        Parsed :class:`ManifestContext`.
    """
    path = Path(manifest_path)
    if repo_root is None:
        repo_root = _find_repo_root(path)

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    modalities: dict[str, ModalitySpec] = {}
    for name, cfg in raw.get("modalities", {}).items():
        output_path = _resolve_path(cfg.get("output_path"), repo_root)
        raw_path = _resolve_path(cfg.get("raw_path"), repo_root)
        extra = {
            k: v
            for k, v in cfg.items()
            if k not in {"source", "format", "tokeniser", "normalise", "raw_path", "output_path"}
        }
        modalities[name] = ModalitySpec(
            name=name,
            source=str(cfg.get("source", "")),
            format=str(cfg.get("format", "")),
            output_path=output_path,
            raw_path=raw_path,
            tokeniser=cfg.get("tokeniser"),
            normalise=cfg.get("normalise"),
            extra=extra,
        )

    return ManifestContext(
        path=path.resolve(),
        schema_version=str(raw.get("schema_version", "1.0")),
        organism=str(raw.get("organism", "")),
        assembly=str(raw.get("assembly", "")),
        modalities=modalities,
        splits={k: float(v) for k, v in raw.get("splits", {}).items()},
        split_strategy=raw.get("split_strategy"),
        fetch_recipe=raw.get("fetch_recipe"),
    )


def _resolve_path(value: str | None, repo_root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _find_repo_root(start: Path) -> Path:
    """Walk parents until a directory containing ``data/schemas`` is found."""
    for parent in [start.resolve(), *start.resolve().parents]:
        if (parent / "data" / "schemas").is_dir():
            return parent
    return start.resolve().parent

"""modENCODE / ENCODE bulk RNA-seq QC and normalisation pipeline.

Reads a gene × sample TPM matrix (plus optional sample metadata), applies
gene/sample QC filters, optionally retains midgut tissue samples only, and
writes a normalised expression matrix to HDF5 for foundation-model training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Tissue labels matched case-insensitively when --midgut-only is set.
MIDGUT_TISSUE_TERMS = frozenset(
    {
        "midgut",
        "mid gut",
        "mid-gut",
        "gut",
        "intestine",
        "digestive",
        "hindgut",
        "foregut",
        "proventriculus",
    }
)

MIN_GENES_DETECTED = 1_000
OUTLIER_ZSCORE_THRESHOLD = 3.0
SPLIT_PLACEHOLDER = "unassigned"

logger = logging.getLogger(__name__)


def is_midgut_tissue(tissue: str) -> bool:
    """Return True if a tissue label corresponds to midgut or gut derivatives."""
    normalized = tissue.lower().strip().replace("-", " ")
    return any(term in normalized for term in MIDGUT_TISSUE_TERMS)


def load_tpm_matrix(tpm_path: Path, gene_column: str) -> pd.DataFrame:
    """Load a TPM matrix with genes as rows and samples as columns."""
    df = pd.read_csv(tpm_path, sep="\t", dtype={gene_column: str})
    if gene_column not in df.columns:
        raise ValueError(f"Gene column '{gene_column}' not found in {tpm_path}")
    df = df.set_index(gene_column)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_sample_metadata(metadata_path: Path, sample_column: str, tissue_column: str) -> pd.DataFrame:
    """Load per-sample metadata used for tissue filtering and stratified splits."""
    meta = pd.read_csv(metadata_path, sep="\t", dtype=str)
    for col in (sample_column, tissue_column):
        if col not in meta.columns:
            raise ValueError(f"Column '{col}' not found in {metadata_path}")
    meta = meta.set_index(sample_column)
    return meta


def filter_midgut_samples(
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
    tissue_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Retain only samples annotated as midgut (or gut derivative) tissues."""
    common = [col for col in tpm.columns if col in metadata.index]
    if not common:
        raise ValueError("No sample IDs overlap between TPM matrix and metadata.")

    midgut_ids = [
        sample_id
        for sample_id in common
        if is_midgut_tissue(str(metadata.loc[sample_id, tissue_column]))
    ]
    dropped = [sample_id for sample_id in common if sample_id not in midgut_ids]
    if not midgut_ids:
        raise ValueError("Midgut filter removed all samples; check tissue annotations.")

    logger.info(
        "Midgut filter: kept %d / %d overlapping samples (%d dropped)",
        len(midgut_ids),
        len(common),
        len(dropped),
    )
    return tpm[midgut_ids], dropped


def count_detected_genes(tpm: pd.DataFrame) -> pd.Series:
    """Count genes with TPM > 0 per sample."""
    return (tpm > 0).sum(axis=0)


def filter_low_coverage_samples(
    tpm: pd.DataFrame,
    min_genes: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove samples with fewer than ``min_genes`` detected genes."""
    detected = count_detected_genes(tpm)
    keep = detected[detected >= min_genes].index.tolist()
    dropped = [sample_id for sample_id in tpm.columns if sample_id not in keep]
    if not keep:
        raise ValueError(f"All samples removed by min_genes={min_genes} filter.")
    if dropped:
        logger.warning("Dropped %d low-coverage samples: %s", len(dropped), dropped)
    return tpm[keep], dropped


def flag_outlier_samples(tpm: pd.DataFrame, z_threshold: float) -> list[str]:
    """Flag outlier samples by z-score on total TPM (logged, excluded from output)."""
    totals = tpm.sum(axis=0)
    mean = totals.mean()
    std = totals.std(ddof=0)
    if std == 0:
        return []
    zscores = (totals - mean) / std
    outliers = zscores[zscores > z_threshold].index.tolist()
    if outliers:
        logger.warning(
            "Flagged %d outlier sample(s) (z > %.1f on total TPM): %s",
            len(outliers),
            z_threshold,
            outliers,
        )
    return outliers


def filter_genes_by_prevalence(
    tpm: pd.DataFrame,
    min_tpm: float,
    min_samples_pct: float,
) -> pd.DataFrame:
    """Keep genes expressed above ``min_tpm`` in at least ``min_samples_pct`` of samples."""
    n_samples = tpm.shape[1]
    min_samples = max(1, int(np.ceil(min_samples_pct * n_samples)))
    expressed = (tpm >= min_tpm).sum(axis=1) >= min_samples
    kept = tpm.loc[expressed]
    logger.info(
        "Gene prevalence filter: kept %d / %d genes (TPM >= %.2f in >= %d samples)",
        kept.shape[0],
        tpm.shape[0],
        min_tpm,
        min_samples,
    )
    return kept


def normalize_expression(tpm: pd.DataFrame) -> np.ndarray:
    """Apply log1p then per-gene z-score across samples.

    Returns:
        Float32 array of shape (n_samples, n_genes).
    """
    log1p = np.log1p(tpm.to_numpy(dtype=np.float64))
    gene_mean = log1p.mean(axis=0, keepdims=True)
    gene_std = log1p.std(axis=0, ddof=0, keepdims=True)
    gene_std = np.where(gene_std == 0, 1.0, gene_std)
    zscore = (log1p - gene_mean) / gene_std
    return zscore.T.astype(np.float32)


def run_qc(
    tpm_path: Path,
    out_path: Path,
    *,
    min_tpm: float,
    min_samples_pct: float,
    min_genes: int,
    outlier_z: float,
    midgut_only: bool,
    metadata_path: Path | None,
    sample_column: str,
    tissue_column: str,
    gene_column: str,
) -> dict[str, object]:
    """Run the full RNA-seq QC pipeline and write HDF5 output."""
    tpm = load_tpm_matrix(tpm_path, gene_column)
    qc_report: dict[str, object] = {
        "input_samples": tpm.shape[1],
        "input_genes": tpm.shape[0],
        "midgut_only": midgut_only,
        "dropped_samples": {},
    }

    if midgut_only:
        if metadata_path is None:
            raise ValueError("--sample-metadata is required when --midgut-only is set.")
        metadata = load_sample_metadata(metadata_path, sample_column, tissue_column)
        tpm, midgut_dropped = filter_midgut_samples(tpm, metadata, tissue_column)
        qc_report["dropped_samples"]["midgut_filter"] = midgut_dropped

    tpm, coverage_dropped = filter_low_coverage_samples(tpm, min_genes)
    qc_report["dropped_samples"]["low_coverage"] = coverage_dropped

    outliers = flag_outlier_samples(tpm, outlier_z)
    qc_report["outlier_samples"] = outliers
    if outliers:
        tpm = tpm.drop(columns=outliers, errors="ignore")

    tpm = filter_genes_by_prevalence(tpm, min_tpm, min_samples_pct)
    expression = normalize_expression(tpm)

    gene_ids = tpm.index.to_numpy()
    sample_ids = tpm.columns.to_numpy()
    str_dtype = h5py.string_dtype(encoding="utf-8")
    splits = np.full(len(sample_ids), SPLIT_PLACEHOLDER, dtype=str_dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("expression", data=expression, dtype="float32")
        h5.create_dataset("gene_ids", data=gene_ids, dtype=str_dtype)
        h5.create_dataset("sample_ids", data=sample_ids, dtype=str_dtype)
        h5.create_dataset("split", data=splits, dtype=str_dtype)
        h5.attrs["normalisation"] = "log1p_then_per_gene_zscore"
        h5.attrs["midgut_only"] = midgut_only
        h5.attrs["n_samples"] = len(sample_ids)
        h5.attrs["n_genes"] = len(gene_ids)
        h5.attrs["qc_report_json"] = json.dumps(qc_report)

    qc_report["output_samples"] = len(sample_ids)
    qc_report["output_genes"] = len(gene_ids)
    logger.info(
        "Wrote %s: %d samples × %d genes",
        out_path,
        len(sample_ids),
        len(gene_ids),
    )
    return qc_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QC and normalise modENCODE bulk RNA-seq TPM matrix to HDF5."
    )
    parser.add_argument("--tpm-tsv", required=True, type=Path, help="Input TPM TSV path.")
    parser.add_argument("--out", required=True, type=Path, help="Output HDF5 path.")
    parser.add_argument(
        "--sample-metadata",
        type=Path,
        default=None,
        help="Optional sample metadata TSV (sample_id, tissue, stage, …).",
    )
    parser.add_argument(
        "--midgut-only",
        action="store_true",
        help="Retain only midgut/gut tissue samples (requires --sample-metadata).",
    )
    parser.add_argument("--min-tpm", type=float, default=1.0, help="Min TPM for gene prevalence.")
    parser.add_argument(
        "--min-samples-pct",
        type=float,
        default=0.05,
        help="Min fraction of samples a gene must exceed min-tpm in.",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=MIN_GENES_DETECTED,
        help="Min detected genes (TPM > 0) per sample.",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=OUTLIER_ZSCORE_THRESHOLD,
        help="Z-score threshold for outlier sample flagging.",
    )
    parser.add_argument("--gene-column", default="gene_id", help="Gene ID column name.")
    parser.add_argument("--sample-column", default="sample_id", help="Sample ID metadata column.")
    parser.add_argument("--tissue-column", default="tissue", help="Tissue metadata column.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        run_qc(
            tpm_path=args.tpm_tsv,
            out_path=args.out,
            min_tpm=args.min_tpm,
            min_samples_pct=args.min_samples_pct,
            min_genes=args.min_genes,
            outlier_z=args.outlier_z,
            midgut_only=args.midgut_only,
            metadata_path=args.sample_metadata,
            sample_column=args.sample_column,
            tissue_column=args.tissue_column,
            gene_column=args.gene_column,
        )
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

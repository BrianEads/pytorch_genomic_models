"""DroID / FlyBase protein-protein interaction graph builder.

Reads an edge-list TSV, filters by confidence score, deduplicates undirected
edges, and writes a PyTorch Geometric ``Data`` object for graph-based training.
Node features (``x``) are left ``None`` — Goal 2 attaches ESM-2 embeddings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

GENE_A_ALIASES = ("gene_a", "interactor_a", "protein_a", "id_a", "source")
GENE_B_ALIASES = ("gene_b", "interactor_b", "protein_b", "id_b", "target")
SCORE_ALIASES = ("score", "confidence", "combined_score", "weight")

logger = logging.getLogger(__name__)


def resolve_column(columns: list[str], aliases: tuple[str, ...], label: str) -> str:
    """Resolve a column name from a list of known aliases."""
    lower_map = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise ValueError(f"Could not find {label} column; tried {aliases}. Found: {columns}")


def load_edges(edges_path: Path) -> pd.DataFrame:
    """Load an interaction edge list TSV."""
    df = pd.read_csv(edges_path, sep="\t", dtype=str)
    gene_a = resolve_column(list(df.columns), GENE_A_ALIASES, "gene A")
    gene_b = resolve_column(list(df.columns), GENE_B_ALIASES, "gene B")
    score_col = resolve_column(list(df.columns), SCORE_ALIASES, "score")

    edges = pd.DataFrame(
        {
            "gene_a": df[gene_a].astype(str).str.strip(),
            "gene_b": df[gene_b].astype(str).str.strip(),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
        }
    )
    return edges.dropna(subset=["gene_a", "gene_b", "score"])


def filter_and_deduplicate_edges(edges: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """Apply confidence, self-loop, and reciprocal-edge filters."""
    filtered = edges[edges["score"] >= min_score].copy()
    n_below = len(edges) - len(filtered)
    if n_below:
        logger.info("Dropped %d edges below min_score=%.2f", n_below, min_score)

    filtered = filtered[filtered["gene_a"] != filtered["gene_b"]]
    n_self = len(edges[edges["score"] >= min_score]) - len(filtered)
    if n_self:
        logger.info("Removed %d self-loop edges", n_self)

    # Canonicalise undirected pairs so A-B == B-A, then keep highest score.
    a = filtered["gene_a"]
    b = filtered["gene_b"]
    swap = a > b
    filtered["gene_a"] = a.where(~swap, b)
    filtered["gene_b"] = b.where(~swap, a)
    filtered = (
        filtered.sort_values("score", ascending=False)
        .drop_duplicates(subset=["gene_a", "gene_b"], keep="first")
        .reset_index(drop=True)
    )
    logger.info("Retained %d unique undirected edges", len(filtered))
    return filtered


def build_pyg_data(edges: pd.DataFrame) -> Data:
    """Build a PyG ``Data`` object with ``edge_index`` and per-node ``gene_id``."""
    genes = pd.Index(
        pd.unique(pd.concat([edges["gene_a"], edges["gene_b"]], ignore_index=True))
    ).sort_values()
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    src = edges["gene_a"].map(gene_to_idx).to_numpy(dtype="int64")
    dst = edges["gene_b"].map(gene_to_idx).to_numpy(dtype="int64")
    edge_index = torch.from_numpy(np.stack([src, dst]))

    data = Data(edge_index=edge_index, num_nodes=len(genes))
    data.gene_id = genes.to_list()
    data.edge_score = torch.tensor(edges["score"].to_numpy(dtype="float32"))
    return data


def build_graph(
    edges_path: Path,
    out_path: Path,
    min_score: float,
) -> Data:
    """Run the full PPI graph pipeline and save a PyG ``Data`` object."""
    edges = load_edges(edges_path)
    logger.info("Loaded %d edges from %s", len(edges), edges_path)

    edges = filter_and_deduplicate_edges(edges, min_score)
    if edges.empty:
        raise ValueError("No edges remaining after filtering.")

    data = build_pyg_data(edges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    logger.info(
        "Wrote %s: %d nodes, %d edges (x=%s)",
        out_path,
        data.num_nodes,
        data.num_edges,
        data.x,
    )
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a DroID/FlyBase PPI graph as a PyG Data object."
    )
    parser.add_argument("--edges-tsv", required=True, type=Path, help="Input edge TSV.")
    parser.add_argument("--out", required=True, type=Path, help="Output .pt path.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum combined confidence score (default: 0.7).",
    )
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
        build_graph(args.edges_tsv, args.out, args.min_score)
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

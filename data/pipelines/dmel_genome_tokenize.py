"""Genome k-mer tokenisation pipeline for D. melanogaster reference FASTA.

Reads a multi-chromosome FASTA, slides fixed-size k-mer windows along each
chromosome, and writes integer token tensors to HDF5 for masked language-model
pre-training (Goal 2 / foundation model).
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Iterator

import h5py
from Bio import SeqIO

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
LOG_INTERVAL = 10_000
WRITE_BATCH_SIZE = 1_024

logger = logging.getLogger(__name__)


def build_kmer_vocabulary(k: int) -> dict[str, int]:
    """Build k-mer vocabulary with special tokens prepended.

    Args:
        k: K-mer length (e.g. 6 for hexamers).

    Returns:
        Mapping from token string to integer ID. Special tokens occupy IDs
        0–4; canonical k-mers follow in lexicographic product order.
    """
    bases = ["A", "C", "G", "T"]
    kmers = ["".join(chars) for chars in product(bases, repeat=k)]
    return {token: idx for idx, token in enumerate(SPECIAL_TOKENS + kmers)}


def seq_to_kmer_ids(sequence: str, k: int, vocab: dict[str, int]) -> list[int]:
    """Convert a DNA sequence to a list of k-mer token IDs.

    Non-ACGT bases produce ``[UNK]`` tokens for overlapping k-mers
    that contain them.

    Args:
        sequence: Uppercase (or mixed-case) DNA string.
        k: K-mer length.
        vocab: Token-to-ID mapping from :func:`build_kmer_vocabulary`.

    Returns:
        List of integer token IDs, length ``max(0, len(sequence) - k + 1)``.
    """
    seq = sequence.upper()
    unk_id = vocab["[UNK]"]
    token_ids: list[int] = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if all(base in "ACGT" for base in kmer):
            token_ids.append(vocab[kmer])
        else:
            token_ids.append(unk_id)
    return token_ids


def iter_kmer_windows(
    kmer_ids: list[int],
    window: int,
    stride: int,
) -> Iterator[tuple[int, list[int]]]:
    """Yield (start_kmer_index, token_window) pairs along a chromosome.

    Args:
        kmer_ids: Full chromosome k-mer ID sequence.
        window: Number of k-mer tokens per window.
        stride: Step size between consecutive window start positions.

    Yields:
        Tuples of start k-mer index and the window token list.
    """
    if len(kmer_ids) < window:
        return
    for start in range(0, len(kmer_ids) - window + 1, stride):
        yield start, kmer_ids[start : start + window]


def count_windows(fasta_path: Path, k: int, window: int, stride: int) -> int:
    """Count total sliding windows across all records in a FASTA file."""
    total = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        n_kmers = max(0, len(record.seq) - k + 1)
        if n_kmers >= window:
            total += (n_kmers - window) // stride + 1
    return total


def tokenize_fasta(
    fasta_path: Path,
    out_path: Path,
    k: int,
    window: int,
    stride: int,
) -> int:
    """Tokenise a FASTA file and write sliding-window k-mer tensors to HDF5.

    Output datasets:
        - ``tokens``: (N, window) int16 — k-mer token IDs per window
        - ``chromosome``: (N,) UTF-8 string — source chromosome / contig name
        - ``start``: (N,) int32 — 0-based genomic start of the window
        - ``end``: (N,) int32 — 0-based exclusive genomic end of the window

    Args:
        fasta_path: Input multi-record FASTA path.
        out_path: Output HDF5 path.
        k: K-mer length.
        window: K-mer tokens per sliding window.
        stride: Stride between window start positions (in k-mer indices).

    Returns:
        Total number of windows written.
    """
    vocab = build_kmer_vocabulary(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_windows = count_windows(fasta_path, k, window, stride)
    if total_windows == 0:
        raise ValueError(
            f"No windows produced from {fasta_path} "
            f"(k={k}, window={window}, stride={stride})."
        )

    logger.info("Writing %d windows to %s", total_windows, out_path)

    windows_written = 0
    with h5py.File(out_path, "w") as h5:
        tokens_ds = h5.create_dataset(
            "tokens",
            shape=(total_windows, window),
            dtype="int16",
            chunks=(min(WRITE_BATCH_SIZE, total_windows), window),
        )
        chrom_ds = h5.create_dataset(
            "chromosome",
            shape=(total_windows,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(min(WRITE_BATCH_SIZE, total_windows),),
        )
        start_ds = h5.create_dataset(
            "start",
            shape=(total_windows,),
            dtype="int32",
            chunks=(min(WRITE_BATCH_SIZE, total_windows),),
        )
        end_ds = h5.create_dataset(
            "end",
            shape=(total_windows,),
            dtype="int32",
            chunks=(min(WRITE_BATCH_SIZE, total_windows),),
        )

        h5.attrs["kmer_size"] = k
        h5.attrs["window"] = window
        h5.attrs["stride"] = stride
        h5.attrs["vocab_size"] = len(vocab)

        batch_tokens: list[list[int]] = []
        batch_chroms: list[str] = []
        batch_starts: list[int] = []
        batch_ends: list[int] = []

        def flush_batch() -> None:
            nonlocal windows_written
            if not batch_tokens:
                return
            n = len(batch_tokens)
            row_slice = slice(windows_written, windows_written + n)
            tokens_ds[row_slice] = batch_tokens
            chrom_ds[row_slice] = batch_chroms
            start_ds[row_slice] = batch_starts
            end_ds[row_slice] = batch_ends
            windows_written += n
            batch_tokens.clear()
            batch_chroms.clear()
            batch_starts.clear()
            batch_ends.clear()

        for record in SeqIO.parse(fasta_path, "fasta"):
            chrom = record.id
            kmer_ids = seq_to_kmer_ids(str(record.seq), k, vocab)
            logger.info("Processing %s (%d k-mers)", chrom, len(kmer_ids))

            for start_kmer, token_window in iter_kmer_windows(kmer_ids, window, stride):
                genomic_start = start_kmer
                genomic_end = start_kmer + window + k - 1
                batch_tokens.append(token_window)
                batch_chroms.append(chrom)
                batch_starts.append(genomic_start)
                batch_ends.append(genomic_end)

                if len(batch_tokens) >= WRITE_BATCH_SIZE:
                    flush_batch()

                if windows_written + len(batch_tokens) > 0:
                    done = windows_written + len(batch_tokens)
                    if done % LOG_INTERVAL == 0:
                        logger.info("Progress: %d / %d windows", done, total_windows)

            flush_batch()

    logger.info("Finished: %d windows written", windows_written)
    return windows_written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenise a dm6 FASTA into k-mer sliding windows (HDF5 output)."
    )
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA path.")
    parser.add_argument("--out", required=True, type=Path, help="Output HDF5 path.")
    parser.add_argument("--kmer-size", type=int, default=6, help="K-mer length (default: 6).")
    parser.add_argument(
        "--window",
        type=int,
        default=512,
        help="K-mer tokens per window (default: 512).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between window starts in k-mer indices (default: 256).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.kmer_size < 1:
        logger.error("--kmer-size must be >= 1")
        sys.exit(1)
    if args.window < 1:
        logger.error("--window must be >= 1")
        sys.exit(1)
    if args.stride < 1:
        logger.error("--stride must be >= 1")
        sys.exit(1)
    if not args.fasta.is_file():
        logger.error("FASTA not found: %s", args.fasta)
        sys.exit(1)

    tokenize_fasta(
        fasta_path=args.fasta,
        out_path=args.out,
        k=args.kmer_size,
        window=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()

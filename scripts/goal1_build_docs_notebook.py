#!/usr/bin/env python3
"""Goal 1: deduplicate docs, split e2e explorer, build annotated notebook."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEDUP_MARKERS = {
    "docs/1_masked_lang_model.md": [
        "In the burgeoning field of genomics, the application of deep learning models",
        "Here is a comprehensive explanation of the concepts of Masked Language Modeling",
    ],
    "docs/2_kmer_pretraining.md": [
        "In the pre-training of genomic foundational models like DNABERT, the k-mer tokenization strategy is a crucial step that adapts principles",
        "The Blueprint of Genomic Language Models",
    ],
    "docs/3_fine_tuning.md": [
        "Fine-Tuning and Interpreting Genomic Foundational Models",
        "Fine-Tuning Pre-trained Genomic Foundational Models: A Deep Dive",
    ],
    "docs/4_attention_deep_dive.md": [
        "1. The Scaled Dot-Product Attention Mechanism\nThe scaled dot-product attention mechanism is a key component",
        "A Deep Dive into Transformer-Based Models in Genomics",
    ],
}

DOC_TITLES = {
    "docs/1_masked_lang_model.md": "Masked Language Modeling for Genomic Sequences",
    "docs/2_kmer_pretraining.md": "K-mer Pre-training for Genomic Foundation Models",
    "docs/3_fine_tuning.md": "Fine-Tuning Pre-trained Genomic Foundation Models",
    "docs/4_attention_deep_dive.md": "Scaled Dot-Product Attention in Genomic Transformers",
}

SECTION_HEADING_PATTERNS = [
    (r"^Masked Language Modeling \(MLM\) for Genomic Sequences$", "## Masked Language Modeling (MLM) for Genomic Sequences"),
    (r"^The Role and Necessity of Positional Encodings$", "## The Role and Necessity of Positional Encodings"),
    (r"^PyTorch Example: Sinusoidal Positional Encodings$", "## PyTorch Example: Sinusoidal Positional Encodings"),
    (r"^End-to-End PyTorch Model for Genomic Sequence Classification$", "## End-to-End PyTorch Model for Genomic Sequence Classification"),
    (r"^K-mer Tokenization Strategy: Rationale and Methodology$", "## K-mer Tokenization Strategy: Rationale and Methodology"),
    (r"^The Data Processing Pipeline$", "## The Data Processing Pipeline"),
    (r"^Masked Language Modeling \(MLM\) and the 15% Masking Strategy$", "## Masked Language Modeling (MLM) and the 15% Masking Strategy"),
    (r"^Python Code Example$", "## Python Code Example"),
    (r"^Fine-Tuning Pre-trained Genomic Foundational Models$", "## Fine-Tuning Pre-trained Genomic Foundational Models"),
    (r"^Architectural Modifications$", "### Architectural Modifications"),
    (r"^Training Strategies$", "### Training Strategies"),
    (r"^Conceptual PyTorch Code Example$", "## Conceptual PyTorch Code Example"),
    (r"^Interpreting 1D Convolutional Layers as Motif Detectors$", "## Interpreting 1D Convolutional Layers as Motif Detectors"),
    (r"^Function of a 1D Convolutional Layer$", "### Function of a 1D Convolutional Layer"),
    (r"^Interpreting and Visualizing Learned Features$", "### Interpreting and Visualizing Learned Features"),
    (r"^The Scaled Dot-Product Attention Mechanism: A Deep Dive$", "## The Scaled Dot-Product Attention Mechanism"),
    (r"^Roles of Query, Key, and Value Vectors$", "### Roles of Query, Key, and Value Vectors"),
    (r"^Sequence of Matrix Operations$", "### Sequence of Matrix Operations"),
    (r"^The Purpose of the Scaling Factor and Softmax Function$", "### The Purpose of the Scaling Factor and Softmax Function"),
    (r"^Permutation-Invariance and the Need for Positional Encodings$", "## Permutation-Invariance and Positional Encodings"),
    (r"^Interpreting and Visualizing Attention Weights for Biological Insights$", "## Interpreting and Visualizing Attention Weights"),
    (r"^PyTorch Code Example: Extracting and Visualizing Attention Weights$", "## PyTorch Code Example: Attention Heatmaps"),
    (r"^Adapting Models: Parameter-Efficient Fine-Tuning \(PEFT\) and LoRA$", "## Parameter-Efficient Fine-Tuning (PEFT) and LoRA"),
    (r"^Low-Rank Adaptation \(LoRA\)$", "### Low-Rank Adaptation (LoRA)"),
    (r"^Advantages of LoRA over Full Fine-Tuning$", "### Advantages of LoRA over Full Fine-Tuning"),
    (r"^Conceptual Code Example with the peft Library$", "## Conceptual LoRA Code Example"),
    (r"^Evaluating Model Performance on Imbalanced Genomic Datasets$", "## Evaluating Model Performance on Imbalanced Datasets"),
    (r"^AUC-ROC vs\. PR-AUC$", "### AUC-ROC vs. PR-AUC"),
    (r"^Scikit-learn Code Example: Computing and Visualizing ROC and Precision-Recall Curves$", "## Scikit-learn ROC / PR-AUC Example"),
]

KMER_DIAGRAM = """```text
Sequence:  A T G C A G T T A C G A
           |-------|               k=4, step=1
               |-------|
                   |-------|
                       |-------|
                           |-------|
                               |-------|
                                   |-------|
                                       |-------|
                                           |-------|

Tokens:  ATGC → id:42
         TGCA → id:17
         GCAG → id:83
         CAGT → id:29
         ...
```"""

SEQUENCE_DIAGRAM = """```text
pos:  1   2   3   4   5   6   7   8   9  10  11  12
seq:  A   T   G   C   A   G   T   T   A   C   G   A
```"""

NOTEBOOK_SECTIONS = [
    {
        "title": "Embeddings: From Nucleotides to Numbers",
        "script": "embedding_ex.py",
        "biology": "Maps DNA nucleotide tokens to dense vectors so neural networks can learn similarity between bases and motifs.",
        "io": "Input: integer indices `(seq_len,)`. Output: embedded tensor `(seq_len, embedding_dim)`.",
        "compute": [("Embedding lookup (len 4, dim 16)", "~0.01 ms / ~0.01 ms", "~1 KB", "Negligible")],
    },
    {
        "title": "Convolution: Finding Motifs in Sequences",
        "script": "convolution.py",
        "biology": "1D convolutions scan the sequence for short motif patterns such as transcription-factor binding sites.",
        "io": "Input: `(batch, channels, length)`. Output: activation map `(batch, out_channels, length_out)`.",
        "compute": [("Conv1d motif scan (batch 1, len 100)", "~5 ms / ~0.5 ms", "~20 MB", "Memory bandwidth")],
    },
    {
        "title": "Attention: Focusing on What Matters",
        "script": "attention.py",
        "biology": "Self-attention lets each position weigh other positions, capturing long-range regulatory dependencies.",
        "io": "Input: `(seq_len, batch, embed_dim)`. Output: context vectors plus `(batch, seq, seq)` weights.",
        "compute": [("Multi-head self-attention (len 100)", "~50 ms / ~2 ms", "~40 MB", "Attention O(n²)")],
    },
    {
        "title": "Attention Heatmap Visualization",
        "script": "heatmap.py",
        "biology": "Visualizing attention weights reveals which bases the model treats as interacting partners.",
        "io": "Input: self-attention on random embeddings. Output: heatmap figure.",
        "compute": [("Heatmap render (100×100)", "~200 ms / ~50 ms", "~5 MB", "CPU plotting")],
    },
    {
        "title": "Transformers: Positional Encodings and Encoder Stack",
        "script": "positional_encoding.py",
        "biology": "Positional encodings restore sequence order; stacked encoder layers build contextual representations.",
        "io": "Input: `(seq_len, batch, embed_dim)`. Output: encoded tensor of same shape.",
        "compute": [("Transformer encoder (2 layers, len 100)", "~200 ms / ~5 ms", "~500 MB", "Attention O(n²)")],
    },
    {
        "title": "Pre-training: K-mer Tokenisation and MLM Masking",
        "script": "pretrain_token_mask",
        "biology": "Overlapping k-mers tokenise DNA; masked language modeling teaches the model genomic grammar.",
        "io": "Input: DNA string. Output: masked token IDs and label tensor with `-100` on unmasked positions.",
        "compute": [("K-mer tokenisation (1 M bp seq)", "~2 s / ~0.3 s", "~50 MB", "CPU string ops")],
    },
    {
        "title": "Fine-Tuning: Classification Head and LoRA",
        "script": "pretrained_finetuning.py",
        "biology": "Adapts a pre-trained foundation model to a downstream task with a new head and optional LoRA adapters.",
        "io": "Input: token IDs `(batch, seq_len)`. Output: classification logits `(batch, num_labels)`.",
        "compute": [("LoRA fine-tune step (batch 8, len 512)", "~500 ms / ~20 ms", "~2 GB", "GPU compute")],
        "optional": True,
    },
    {
        "title": "Optimization: AdamW Training Step",
        "script": "optimizer.py",
        "biology": "Optimizers update model weights to minimise prediction error on labeled genomic data.",
        "io": "Input: model, batch, labels. Output: scalar loss after one `optimizer.step()`.",
        "compute": [("AdamW step (64×10 linear)", "~1 ms / ~0.1 ms", "~1 MB", "Negligible")],
    },
    {
        "title": "Regularization: Dropout and Weight Decay",
        "script": "regularization_methods.py",
        "biology": "Dropout and weight decay reduce overfitting when training data are limited.",
        "io": "Input: activation tensor. Output: dropped activations; optimizer with `weight_decay`.",
        "compute": [("Dropout forward (batch 20)", "~0.1 ms", "Negligible", "Negligible")],
    },
    {
        "title": "Batching and Padding Variable-Length Sequences",
        "script": "batch_pad_loader.py",
        "biology": "Real genomic batches mix sequence lengths; padding aligns them for efficient GPU training.",
        "io": "Input: list of variable-length tensors. Output: padded batch `(batch, max_len)`.",
        "compute": [("pad_sequence collate (batch 32)", "~1 ms", "~8 MB", "CPU memory copy")],
    },
    {
        "title": "Loss Functions for Genomic Classification",
        "script": "loss_functions.py",
        "biology": "Binary and multi-class losses score how well predictions match binding or structure labels.",
        "io": "Input: logits and labels. Output: scalar loss.",
        "compute": [("BCE / CE loss (batch 4)", "~0.05 ms", "Negligible", "Negligible")],
    },
    {
        "title": "Mixed Precision Training",
        "script": "amp_training.py",
        "biology": "FP16 autocast speeds GPU training and reduces memory without changing the learning objective.",
        "io": "Input: model batch on CUDA. Output: loss after scaled backward pass.",
        "compute": [("AMP training step (batch 64)", "~2 ms / ~0.5 ms", "~half vs FP32", "GPU tensor cores")],
        "optional": True,
    },
    {
        "title": "Distributed Data Parallel Training",
        "script": "DD_parallel.py",
        "biology": "DDP shards data across GPUs so large genomic corpora train in parallel.",
        "io": "Input: per-rank data shard. Output: synchronised gradients across processes.",
        "compute": [("DDP all-reduce (4 GPUs)", "~5 ms / ~1 ms", "4× model replicas", "Network sync")],
        "optional": True,
    },
    {
        "title": "End-to-End Genomic Classifier",
        "script": "e2e_ex.py",
        "biology": "Combines embedding, convolution, pooling, and a linear head to detect a simple ACG motif.",
        "io": "Input: `(batch, seq_len)` token IDs. Output: binding probability `(batch, 1)`.",
        "compute": [("E2E training epoch (2k seqs, CPU)", "~30 s / ~3 s", "~200 MB", "Conv + linear")],
    },
    {
        "title": "Evaluation Metrics for Imbalanced Genomic Data",
        "script": "model_eval.py",
        "biology": "ROC and precision-recall curves assess rare-class performance beyond misleading accuracy.",
        "io": "Input: `y_true`, `y_scores`. Output: ROC/PR plots and AUC metrics.",
        "compute": [("ROC + PR plot (20 samples)", "~300 ms", "~5 MB", "CPU plotting")],
    },
]


def deduplicate_text(text: str, markers: list[str]) -> str:
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    return text[:cut].rstrip() + "\n"


def promote_headings(line: str) -> str:
    stripped = line.strip()
    for pattern, replacement in SECTION_HEADING_PATTERNS:
        if re.match(pattern, stripped):
            return replacement
    return line


def fence_code_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "python":
            out.append("```python")
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() in {"python", "text"}:
                    break
                if promote_headings(nxt) != nxt and not nxt.startswith(" "):
                    break
                if re.match(r"^[A-Z][^\n]{10,}$", nxt.strip()) and not nxt.strip().startswith("#"):
                    if i > 0 and lines[i - 1].strip() == "":
                        break
                out.append(nxt)
                i += 1
            out.append("```")
            continue
        out.append(promote_headings(line))
        i += 1
    return "\n".join(out)


def clean_artifacts(text: str) -> str:
    text = re.sub(r"\[\.\.\.\]\(asc_slot://[^)]+\)", "...", text)
    text = text.replace("asc_slot://start-slot-14", "0")
    text = text.replace("asc_slot://start-slot-94", "0")
    text = text.replace("asc_slot://start-slot-222", "0, 1, 2, 3")
    text = text.replace("asc_slot://start-slot-224", "0, 1, 2")
    text = text.replace("asc_slot://start-slot-226", "0, 1, 2, 3, 2, 1")
    text = text.replace("asc_slot://start-slot-228", "0, 1")
    text = text.replace("asc_slot://start-slot-232", "2")
    text = text.replace("asc_slot://start-slot-234", "1")
    text = text.replace("asc_slot://start-slot-236", "2")
    text = text.replace("asc_slot://start-slot-238", "1, 2, 3")
    text = text.replace("asc_slot://start-slot-262", "0, 1")
    text = re.sub(r"\bm\nMotif", "Motif", text)
    text = text.replace("torch.tensor()", "torch.tensor([0, 1, 2, 3], dtype=torch.long)")
    text = text.replace("np.array()", "np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])")
    text = text.replace("plt.plot(, ,", "plt.plot([0, 1], [0, 1],")
    text = text.replace("import matplotlib.pyplot to plt", "import matplotlib.pyplot as plt")
    return text


def format_doc(rel_path: str) -> None:
    path = ROOT / rel_path
    raw = path.read_text()
    deduped = deduplicate_text(raw, DEDUP_MARKERS[rel_path])
    cleaned = clean_artifacts(deduped)
    fenced = fence_code_blocks(cleaned)
    title = DOC_TITLES[rel_path]
    body = fenced.strip()
    if not body.startswith("# "):
        body = f"# {title}\n\n{body}"
    if rel_path == "docs/2_kmer_pretraining.md" and "k=4, step=1" not in body:
        insert_at = body.find("## K-mer Tokenization Strategy")
        if insert_at != -1:
            end = body.find("\n\n", insert_at)
            body = body[:end] + f"\n\n### Sliding-window example (k=4)\n\n{KMER_DIAGRAM}\n" + body[end:]
    if rel_path == "docs/1_masked_lang_model.md" and "pos:  1" not in body:
        body += f"\n\n## Sequence position diagram\n\n{SEQUENCE_DIAGRAM}\n"
    path.write_text(body + "\n")


def compute_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [
        "> **Compute note**",
        "> | Step | Typical wall-clock (CPU / single GPU) | Memory footprint | Bottleneck |",
        "> |------|---------------------------------------|------------------|------------|",
    ]
    for step, wall, mem, bottleneck in rows:
        lines.append(f"> | {step} | {wall} | {mem} | {bottleneck} |")
    return "\n".join(lines)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
        "outputs": [],
        "execution_count": None,
    }


def build_notebook() -> None:
    cells: list[dict] = [
        md_cell(
            "# E2E Genomic Model Explorer (Notebook)\n\n"
            "Runnable companion to [`e2e_explorer.md`](../e2e_explorer.md). "
            "Each section runs a focused PyTorch example from the repo root scripts."
        )
    ]
    for section in NOTEBOOK_SECTIONS:
        script_path = ROOT / section["script"]
        if not script_path.exists():
            continue
        source = script_path.read_text()
        explainer = (
            f"## {section['title']}\n\n"
            f"**What this step does:** {section['biology']}\n\n"
            f"**Input / Output shapes:** {section['io']}\n\n"
            f"{compute_table(section['compute'])}"
        )
        if section.get("depends"):
            explainer += f"\n\n> **Prerequisite:** run `{section['depends']}` first (e.g. `{section['depends']}` sets `attn_weights`)."
        if section.get("optional"):
            explainer += "\n\n> **Optional:** requires extra packages (`transformers`, `peft`, or CUDA for AMP/DDP)."
        cells.append(md_cell(explainer))
        cells.append(code_cell(source))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    out = ROOT / "e2e_explorer.ipynb"
    try:
        import nbformat

        nb_node = nbformat.from_dict(nb)
        nbformat.validate(nb_node)
        nbformat.write(nb_node, out)
    except ImportError:
        out.write_text(json.dumps(nb, indent=1) + "\n")


def split_explorer_md() -> None:
    src = ROOT / "e2e_explorer.ipynb"
    if not src.exists():
        return
    # If already JSON notebook from a prior run, read companion narrative from old plain text backup
    text = src.read_text()
    if text.lstrip().startswith("{"):
        return
    cleaned = clean_artifacts(text)
    # Convert numbered sections to markdown headings
    cleaned = re.sub(r"^(\d+(?:\.\d+)?)\. (.+)$", r"## \1. \2", cleaned, flags=re.MULTILINE)
    cleaned = fence_code_blocks(cleaned)
    header = (
        "# E2E Genomic Model Explorer\n\n"
        "Readable walkthrough of PyTorch building blocks for genomic deep learning. "
        "For runnable cells with compute notes, open [`e2e_explorer.ipynb`](e2e_explorer.ipynb).\n\n"
    )
    (ROOT / "e2e_explorer.md").write_text(header + cleaned.strip() + "\n")


def main() -> None:
    for rel in DEDUP_MARKERS:
        format_doc(rel)
    split_explorer_md()
    build_notebook()
    print("Goal1 doc/notebook build complete.")


if __name__ == "__main__":
    main()

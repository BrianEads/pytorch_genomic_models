# Goal 1 — Docs & Notebook Reformatting

**Branch:** `feat/goal-1-docs-reformat`
**Agent:** `docs-reformatter`

---

## Goal Summary

The repository's existing documentation and notebook files require reformatting before they can serve as reliable learning or reference material. `docs/1_masked_lang_model.mdmd` uses a non-standard file extension and raw prose without Markdown structure. The end-to-end explorer notebook (`e2e_explorer.ipynb`) contains broken tensor literals, unresolved rendering artefacts, and no explanatory narrative between code sections. `README.md` lacks headers and a Quick Start guide. This goal delivers clean, consistently formatted docs and a fully annotated notebook that an agent or human reader can follow from start to finish.

---

## Current State Audit

- **`docs/1_masked_lang_model.mdmd`**
  - Non-standard `.mdmd` extension; should be renamed to `1_masked_lang_model.md`
  - No triple-backtick fences on code blocks (bare `python` inline text, not fenced)
  - Narrative text runs together with code; no section breaks or headers
  - Sequence chunk examples are not highlighted or annotated with position numbers

- **`e2e_explorer.ipynb`**
  - `torch.tensor()` calls contain empty arguments — broken, non-functional examples
  - Unresolved `asc_slot://` render artefacts (slot references from notebook generation that were never resolved) left in Markdown cells
  - No Markdown explainer cells between code sections — reader cannot follow the biological or technical reasoning
  - No compute-load or memory footprint notes for any cell

- **`README.md`**
  - No Markdown headings — entire document is a wall of plain prose
  - No "Quick Start" section with installation steps or a minimal run example
  - No reference to what the repo contains (docs/, notebooks, scripts)

- **`docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, `docs/4_attention_deep_dive.md`**
  - Need formatting audit: check for missing fences, inconsistent heading levels, absent sequence examples

---

## Acceptance Criteria

1. All docs use proper `.md` extension — rename `docs/1_masked_lang_model.mdmd` → `docs/1_masked_lang_model.md`.
2. Every code block is properly fenced with triple backticks and a language specifier (` ```python ` … ` ``` `).
3. DNA/RNA sequence examples use a `text` fence with position numbers clearly annotated (see formatting standard below).
4. k-mer chunking diagrams show the sliding window visually (see formatting standard below).
5. `e2e_explorer.ipynb` has a **Markdown cell before every code cell** explaining: (a) what the step does biologically, (b) expected input/output shapes, (c) compute load notes (see template below).
6. All broken `torch.tensor()` calls are replaced with correct, minimal working examples.
7. All `asc_slot://` render artefacts are removed or replaced with proper explanatory text.
8. `README.md` is reformatted as proper Markdown with an H1 title, H2 section headings, bullet lists, and a "Quick Start" section (install + minimal run command).
9. `docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, and `docs/4_attention_deep_dive.md` pass the same fencing and heading standards.
10. All `.ipynb` outputs are stripped before commit (`nbstripout` hook active).

---

## Compute-Load Explainer Template

Every logical section of `e2e_explorer.ipynb` must include the following callout as a Markdown cell immediately preceding the relevant code cell:

```markdown
> **⚙️ Compute note**
> | Step | Typical wall-clock (CPU / single GPU) | Memory footprint | Bottleneck |
> |------|---------------------------------------|------------------|------------|
> | k-mer tokenisation (1 M bp seq) | ~2 s / ~0.3 s | ~50 MB | CPU-bound string ops |
> | Embedding lookup (batch 32, len 512) | ~1 ms / ~0.1 ms | ~8 MB | Negligible |
> | Conv1d motif scan | ~5 ms / ~0.5 ms | ~20 MB | Memory bandwidth |
> | Transformer encoder (2 layers) | ~200 ms / ~5 ms | ~500 MB | Attention O(n²) |
> | MLM pre-training epoch (100 k seqs) | ~4 h / ~12 min (A100) | ~8 GB | GPU compute |
```

Adjust the table rows to match the specific step being described. Every section must have at least one row; multi-step sections should have multiple rows.

---

## Sequence Chunk Formatting Standard

### Position/sequence diagram

Sequence examples must use a `text` fence with explicit position numbers on the line above the bases:

````text
```text
pos:  1   2   3   4   5   6   7   8   9  10  11  12
seq:  A   T   G   C   A   G   T   T   A   C   G   A
```
````

### k-mer sliding window ASCII example

k-mer tokenisation diagrams must show the sliding window explicitly:

````text
```text
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
```
````

---

## Work Breakdown Table

| Task | Files touched | Effort |
|------|---------------|--------|
| Rename + fix `1_masked_lang_model.mdmd` | `docs/1_masked_lang_model.mdmd` → `docs/1_masked_lang_model.md` | S |
| Audit & fix docs 2–4 for fencing and headings | `docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, `docs/4_attention_deep_dive.md` | M |
| Fix broken `torch.tensor()` calls in notebook | `e2e_explorer.ipynb` | S |
| Remove `asc_slot://` artefacts from notebook | `e2e_explorer.ipynb` | S |
| Add Markdown explainer cells before every code cell | `e2e_explorer.ipynb` | L |
| Add compute-load callout tables to notebook | `e2e_explorer.ipynb` | M |
| Reformat `README.md` with headers and Quick Start | `README.md` | S |
| Verify `nbstripout` hook is active | `.pre-commit-config.yaml` (create if absent) | S |

---

## Tooling & Stack

| Tool / Library | Purpose |
|----------------|---------|
| `nbformat` | Read, edit, and write `.ipynb` JSON programmatically |
| `nbstripout` | Strip cell outputs before commit |
| `pre-commit` | Hook runner for `nbstripout` and any linters |
| `markdownlint` (optional) | Lint Markdown files for heading and fence consistency |
| Python `json` stdlib | Manual inspection of notebook cell sources if needed |

Install:
```bash
pip install nbformat nbstripout pre-commit
```

---

## Agent Instructions — `docs-reformatter`

Execute these steps in order. Do not skip a step even if it appears to be already done — verify each one.

### Step 1 — Set up branch and tools

```bash
git checkout -b feat/goal-1-docs-reformat
pip install nbformat nbstripout pre-commit
```

### Step 2 — Rename and fix `1_masked_lang_model.mdmd`

1. Rename the file:
   ```bash
   git mv docs/1_masked_lang_model.mdmd docs/1_masked_lang_model.md
   ```
2. Open `docs/1_masked_lang_model.md` and:
   - Add an H1 title at the top if absent.
   - Find every code block not wrapped in triple backticks and wrap it with ` ```python ` … ` ``` `.
   - Add H2 section headings to separate narrative, code, and sequence example sections.
   - Replace any bare sequence strings with the position-annotated `text` fenced format (see formatting standard above).
   - Add the k-mer sliding window ASCII diagram wherever k-mer tokenisation is discussed.

### Step 3 — Audit and fix docs 2–4

For each of `docs/2_kmer_pretraining.md`, `docs/3_fine_tuning.md`, `docs/4_attention_deep_dive.md`:
1. Check that every code block uses triple-backtick fencing with a language specifier.
2. Check that heading levels are consistent (H1 title → H2 sections → H3 subsections).
3. Check that any DNA/RNA sequence examples use the `text` fenced format.
4. Fix any issues found in-place.

### Step 4 — Fix `e2e_explorer.ipynb` artefacts

Using `nbformat` (or direct JSON editing):
1. Find all code cells containing `torch.tensor()` with empty or missing arguments. Replace with the minimal working equivalent, e.g.:
   ```python
   import torch
   x = torch.tensor([1.0, 2.0, 3.0])
   ```
2. Find all Markdown cells containing `asc_slot://` strings. Remove the slot reference and replace with a plain-text description of what was intended.

### Step 5 — Add explainer Markdown cells to `e2e_explorer.ipynb`

For every existing code cell in the notebook, insert a new Markdown cell immediately before it containing:
- **What this step does** — one or two sentences explaining the biological or technical purpose.
- **Input / Output shapes** — what tensor shapes or data structures enter and leave this cell.
- **Compute-load callout** — the `⚙️ Compute note` table using the template above, filled in for this specific step.

### Step 6 — Reformat `README.md`

Rewrite `README.md` with proper Markdown structure:
```markdown
# PyTorch Genomic Models

One-paragraph description of the project.

## Quick Start
\`\`\`bash
pip install -r requirements.txt
python e2e_ex.py
\`\`\`

## Repository Layout
(bullet list of top-level dirs and files with one-line descriptions)

## Goals & Plans
(link table pointing to PLAN_GOAL1–4 and SKILLS.md)

## Contributing
(coding standards: Black, type hints, Google docstrings, nbstripout)
```

### Step 7 — Enable `nbstripout` hook

1. Create or update `.pre-commit-config.yaml` to include:
   ```yaml
   repos:
     - repo: https://github.com/kynan/nbstripout
       rev: 0.7.1
       hooks:
         - id: nbstripout
   ```
2. Run:
   ```bash
   pre-commit install
   ```

### Step 8 — Verify and commit

```bash
# Verify notebook parses correctly
python -c "import nbformat; nbformat.read('e2e_explorer.ipynb', as_version=4)"

# Strip outputs
nbstripout e2e_explorer.ipynb

git add docs/ e2e_explorer.ipynb README.md .pre-commit-config.yaml
git commit -m "feat(goal-1): reformat docs, fix notebook artefacts, add explainer cells"
git push origin feat/goal-1-docs-reformat
```

Open a pull request targeting `main` with the title: `[Goal 1] Docs & notebook reformat`.

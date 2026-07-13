---
name: goal-1-docs-reformatter
description: >-
  Reformat docs and the e2e explorer notebook for Goal 1. Use when working on
  docs/, e2e_explorer.ipynb, README Quick Start, or pre-commit nbstripout setup.
---

# Goal 1 — Docs Reformatter

**Branch:** `feat/goal-1-docs-reformat`
**Plan:** [PLAN_GOAL1_docs_reformat.md](../../PLAN_GOAL1_docs_reformat.md)
**Agent name:** `docs-reformatter`

## Scope

- Markdown docs under `docs/`
- `e2e_explorer.ipynb` and companion `e2e_explorer.md`
- `README.md` Quick Start and layout table
- `.pre-commit-config.yaml` (`nbstripout`)
- `scripts/goal1_build_docs_notebook.py` (notebook regeneration helper)

**Out of scope:** model code, data pipelines, infra, DFW integration.

## Acceptance Criteria Checklist

- [x] All docs use `.md` extension (no `.mdmd`)
- [x] Code blocks fenced with language specifiers
- [x] DNA/RNA examples use position-annotated `text` fences
- [x] `e2e_explorer.ipynb`: Markdown + compute note before every code cell
- [x] Broken `torch.tensor()` calls fixed
- [x] All `asc_slot://` artefacts removed
- [x] `README.md`: H1, H2 sections, Quick Start (`uv sync`)
- [x] `nbstripout` pre-commit hook configured
- [ ] PR opened to `main` (user-triggered)
- [ ] Optional: `markdownlint` clean on `docs/`

## Coordination Rules

- Notebook is a **separable concern**: readable narrative lives in `e2e_explorer.md`; notebook is the runnable companion.
- Do not modify `models/`, `data/pipelines/`, or `infra/` unless user explicitly expands scope.
- Regenerate notebook from script when bulk-editing structure: `uv run python scripts/goal1_build_docs_notebook.py`.

## When to Pause / Escalate

| Trigger | Action |
|---------|--------|
| Need to change training/model code to fix examples | Escalate to Goal 2 or user |
| User wants notebook in external env only | Update D13 in `DECISIONS.md`; conductor decides packaging |
| Pre-commit hook conflicts with other goals | Report to conductor |

## Status Board Update (after session)

Report to conductor:

```
Goal 1 | branch | completed: [...] | blocked: [...] | next: [...]
```

## Dependencies

- None blocking. Goal 1 improves onboarding for all other goals.
- **Program gate P0** does not block Goal 1 work.

## Primary Skills

`code-formatting`, `notebook-editing`, `documentation-writing` — see [SKILLS.md](../../SKILLS.md).

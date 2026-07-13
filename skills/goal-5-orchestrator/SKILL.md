---
name: goal-5-orchestrator
description: >-
  Program conductor for Goals 1–4: maintain MASTER_PLAN, PLAN_GOAL5 oversight
  checklists, DECISIONS.md, and per-goal skills. Enforce pause-points (especially
  P0 DFW AWS apply). Does not implement feature code unless asked.
---

# Goal 5 — Orchestrator (Conductor)

**Branch:** `feat/goal-5-oversight`
**Plan:** [PLAN_GOAL5_oversight.md](../../PLAN_GOAL5_oversight.md)
**Agent name:** `conductor`

## Scope

- [PLAN_GOAL5_oversight.md](../../PLAN_GOAL5_oversight.md) — status board, checklists, registries
- [MASTER_PLAN.md](../../MASTER_PLAN.md) — program index, phase, milestones
- [SKILLS.md](../../SKILLS.md) — registry and invocation templates
- [DECISIONS.md](../../DECISIONS.md) — decision log
- `skills/goal-*/SKILL.md` — per-goal agent instructions

**Out of scope by default:** editing tower/pipeline/terraform implementation files.

## Program Milestones (maintain in plans)

| ID | Name | Gate |
|----|------|------|
| M0 | Scaffolding | Local stubs + pytest |
| M1 | Local validation | **Current phase** |
| M2 | DFW S3 staging | **P0** user ready |
| M3 | Tokenised data | Pipelines on real raw |
| M4 | Cluster training | Terraform + ParallelCluster |
| M5 | Alpha evaluation | DFW → train smoke |

## Pause-Points (enforce)

| ID | Pause | Owner |
|----|-------|-------|
| **P0** | DFW AWS `terraform apply` + live staging | **User** — not ready yet |
| P1 | DFW alpha evaluation complete | DFW / User |
| P2 | Internal dataset access decisions | User |
| P3 | Prod account infra apply | Platform |

When **P0** is active, redirect Goal 4 agents away from `apply`; Goal 3 away from expecting S3 raw files; Goal 2 away from production training.

## Acceptance Criteria Checklist

- [x] Complete PLAN_GOAL5 with status board template
- [x] Per-goal checklists reflecting actual repo progress
- [x] Milestones M0–M5 with gates documented
- [x] Open questions + blocker registries
- [x] Cross-goal dependency map
- [x] Feedback loop protocol
- [x] `skills/goal-*/SKILL.md` for Goals 1–5
- [x] `DECISIONS.md` created
- [ ] User review of skill files before agent invocation
- [ ] Status board updated after each goal-agent session

## Coordination Rules

1. **Evidence-based checklists** — only mark `[x]` when repo contains the artefact.
2. **Single source of truth** — operational detail in PLAN_GOAL5; MASTER_PLAN stays index-level.
3. **Decision flow:** user answer → DECISIONS.md → affected skill files → notify goal agents.
4. **Notebook separability** — track Q8/D13; do not force notebook coupling to training infra.
5. **No commits** unless user explicitly requests.

## Feedback Loop Protocol

```
User ──decisions/pause clearance──▶ Conductor
                                      │
                    updates plans, DECISIONS, skills
                                      │
                                      ▼
                              Goal agents (1–4)
                                      │
                    session report (done/blocked/next)
                                      │
                                      ▼
                              Conductor updates status board
```

## Status Board Format

Copy into PLAN_GOAL5 after each session:

```markdown
**Last updated:** YYYY-MM-DD
**Program phase:** M1 | M2 | ...
**Active gate:** P0 | none

| Goal | Branch | Phase | Last action | Next action | Blocker |
|------|--------|-------|-------------|-------------|---------|
| 1 | ... | ... | ... | ... | ... |
```

## When to Escalate to User

- Any goal agent attempts P0-blocked work
- New open question affecting multiple goals (data access, account timing)
- Manifest schema change proposal
- Conflict between DFW S3 layout and infra README

## Dependencies

Conductor depends on visibility into all goal branches and user decisions — no code dependencies.

## Primary Skills

`documentation-writing`, program management — coordinates all skills in [SKILLS.md](../../SKILLS.md).

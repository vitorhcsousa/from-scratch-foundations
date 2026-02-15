# from-scratch-foundations

A public **from-scratch foundations** monorepo/workbench for my AI research transition: **notes, first-principles implementations, experiments, and Anki exports** — all tracked as **GitHub Issues/Projects**.

This repo is intentionally **not** a reusable ML template. It’s where I:
- implement core ideas from scratch (clean APIs + tests)
- run small, reproducible experiments (configs → results → short report)
- write structured notes (interview-level clarity)
- export spaced repetition decks (TSV)

---

## How work is tracked

- **Everything is tracked as GitHub Issues** (one “done” = one measurable artifact).
- Weekly execution uses a **parent “Weekly” issue** + **sub-issues per area** (Math / DL / RL / NLP / DSA / Writing).
- The canonical view lives in a **GitHub Project** with an **Iteration (weekly) field** and per-area views.

---

## Repository layout

```text
from-scratch-foundations/
├── src/foundations/           # packaged helpers + CLI
│   ├── cli/                   # Typer command groups
│   ├── projects/              # importable “project modules” (small + tested)
│   └── utils/
├── projects/                  # non-packaged work (WIP, notebooks, scratch)
│   ├── transformer/
│   ├── rl/
│   └── nlp/
├── notes/                     # structured notes (markdown)
│   ├── math/
│   ├── deep-learning/
│   ├── reinforcement-learning/
│   ├── nlp-llms/
│   └── dsa/
├── experiments/               # configs + results + short reports
│   ├── configs/
│   ├── results/
│   └── reports/
├── anki/                      # TSV exports + deck metadata
│   ├── exports/
│   └── decks/
├── artifacts/                 # local-only (gitignored)
│   ├── data/
│   └── outputs/
├── scripts/                   # small utilities
└── tests/                     # tests (package + CLI smoke)
```

**Conventions**
- `src/foundations/projects/`: small importable modules (keep tests close)
- top-level `projects/`: heavier WIP that may graduate into its own repo
- `artifacts/`: anything large or local-only (ignored by git)

---

## Tooling

This repo keeps a consistent dev UX:
- **uv** for env + lock/sync
- **Typer** for the CLI
- **ruff** for lint/format
- **ty** for type checks
- **pytest** for tests
- **pre-commit** hooks (optional)

---

## How to run

```bash
# install deps
uv sync

# run full local CI
make ci

# CLI help
uv run foundations --help

# list packaged project modules
uv run foundations projects list

# list note categories (from repo checkout)
uv run foundations notes list
```

---

## License

See `LICENSE`.

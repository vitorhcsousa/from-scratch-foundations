"""from-scratch-foundations.

A public monorepo/workbench for *first-principles* implementations, experiments, notes,
and Anki exports.

The Python package (`foundations`) contains small reusable utilities and a Typer CLI.
Most heavy work lives in top-level folders like `projects/`, `notes/`, and `experiments/`.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Keep a single source of truth for the package version.
# (Avoid importing importlib.metadata at import time to keep this package lightweight.)
__version__ = "0.1.0"

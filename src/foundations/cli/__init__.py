"""Typer CLI entrypoint.

The CLI is intentionally minimal and oriented around *navigation* of this monorepo:
- listing packaged project modules
- listing note categories

Add new command groups as the repo grows.
"""

from __future__ import annotations

import typer

from foundations.cli.notes import app as notes_app
from foundations.cli.projects import app as projects_app

app = typer.Typer(
    name="foundations",
    help="from-scratch-foundations CLI",
    add_completion=False,
    rich_markup_mode="rich",
)

app.add_typer(projects_app, name="projects", help="Project helpers")
app.add_typer(notes_app, name="notes", help="Notes helpers")

if __name__ == "__main__":
    app()

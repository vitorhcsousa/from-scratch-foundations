"""Notes-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()
NOTES_DIR = Path("docs")


def _find_repo_root() -> Path:
    """Best-effort repository root discovery.

    We prefer the *current working directory* so the CLI behaves correctly in a
    dev checkout. We walk upwards until we find a `pyproject.toml`.
    """
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


@app.command("list")
def list_note_categories() -> None:
    """List note categories under the repo `notes/` folder."""
    notes_dir = _find_repo_root() / "notes"

    if not notes_dir.exists():
        console.print("[red]No notes directory found:[/red]", str(notes_dir))
        raise typer.Exit(code=1)

    cats: list[str] = []
    for child in sorted(notes_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            cats.append(child.name)

    if not cats:
        console.print("(no note categories yet)")
        raise typer.Exit(code=0)

    console.print("\n".join(cats))

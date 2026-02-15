"""Project-related CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()


def _project_pkg_dir() -> Path:
    """Return the directory for packaged project modules."""
    # src/foundations/cli/projects.py -> src/foundations/projects
    return Path(__file__).resolve().parents[1] / "projects"


@app.command("list")
def list_projects() -> None:
    """List available packaged project modules under `src/foundations/projects/`."""
    base = _project_pkg_dir()

    if not base.exists():
        console.print("[red]No packaged projects directory found:[/red]", str(base))
        raise typer.Exit(code=1)

    modules: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if (child / "__init__.py").exists():
            modules.append(child.name)

    if not modules:
        console.print("(no packaged project modules yet)")
        raise typer.Exit(code=0)

    console.print("\n".join(modules))

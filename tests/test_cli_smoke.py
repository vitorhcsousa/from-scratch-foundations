from __future__ import annotations

from typer.testing import CliRunner

from foundations.cli import app


def test_import_foundations() -> None:
    import foundations  # noqa: F401


def test_cli_help_exit_code_0() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_projects_list_exit_code_0() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["projects", "list"])
    assert result.exit_code == 0


def test_notes_list_exit_code_0() -> None:
    runner = CliRunner()
    # This test expects to run from the repo root (pytest default).
    result = runner.invoke(app, ["notes", "list"])
    assert result.exit_code == 0

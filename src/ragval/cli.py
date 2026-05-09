"""Placeholder CLI. Filled out in week 3."""

from __future__ import annotations

import typer

app = typer.Typer(help="ragval — rigorous RAG evaluation.")


@app.command()
def smoke():
    """Run the smoke test."""
    from ragval.smoke_test import main

    main()


@app.command()
def version():
    """Print version."""
    from ragval import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()

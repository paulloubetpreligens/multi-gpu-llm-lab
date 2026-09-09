"""The Typer application every entrypoint module registers a command on."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Multi-GPU LLM lab."""

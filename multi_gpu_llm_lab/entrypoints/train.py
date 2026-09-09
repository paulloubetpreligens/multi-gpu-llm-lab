"""The `train` command."""

from pathlib import Path
from typing import Annotated

import typer

from multi_gpu_llm_lab.configs import config_paths, resolve_trainer_config
from multi_gpu_llm_lab.entrypoints.app import app
from multi_gpu_llm_lab.train import train as run_training

CONFIG_HELP = "YAML config file; omitted keys keep their in-code default"
OVERRIDE_HELP = f"key=value, repeatable, wins over the file. Keys: {', '.join(config_paths())}"


@app.command()
def train(
    config: Annotated[Path | None, typer.Option(help=CONFIG_HELP)] = None,
    overrides: Annotated[list[str] | None, typer.Option("--set", help=OVERRIDE_HELP)] = None,
) -> None:
    """Train a GPT-2 on a single GPU."""
    run_training(resolve_trainer_config(config, overrides or []))

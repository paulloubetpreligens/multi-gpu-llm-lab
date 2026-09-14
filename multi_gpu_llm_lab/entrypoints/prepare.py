"""The `prepare` command."""

from pathlib import Path
from typing import Annotated

import typer

from multi_gpu_llm_lab.data import prepare_shards
from multi_gpu_llm_lab.entrypoints.app import app

TEXT_HELP = "plain-text file to tokenize with the GPT-2 BPE"
OUT_DIR_HELP = "directory the train.bin and val.bin shards are written to"
VAL_FRACTION_HELP = "share of the tokens held out as the validation shard"


@app.command()
def prepare(
    text: Annotated[Path, typer.Argument(help=TEXT_HELP)],
    out_dir: Annotated[Path, typer.Option(help=OUT_DIR_HELP)] = Path("data"),
    val_fraction: Annotated[float, typer.Option(help=VAL_FRACTION_HELP)] = 0.1,
) -> None:
    """Tokenize a text file into the uint16 shards a training run reads."""
    train_path, val_path = prepare_shards(text, out_dir, val_fraction)

    print(f"wrote {train_path} and {val_path}")

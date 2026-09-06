"""Train a model."""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import islice
from typing import Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from multi_gpu_llm_lab.configs import TrainerConfig, build_model_config
from multi_gpu_llm_lab.data import build_dataset
from multi_gpu_llm_lab.device import select_device
from multi_gpu_llm_lab.model import GPT

OPTIMIZERS: dict[str, bool] = {"adamw": False, "adamw_fused": True}


@dataclass(frozen=True)
class Precision:
    """The two orthogonal axes a precision mode sets."""

    autocast_dtype: torch.dtype | None
    matmul: Literal["highest", "high"]


PRECISIONS: dict[str, Precision] = {
    "fp32": Precision(autocast_dtype=None, matmul="highest"),
    "tf32": Precision(autocast_dtype=None, matmul="high"),
    "bf16": Precision(autocast_dtype=torch.bfloat16, matmul="high"),
}


@dataclass
class Trainer:
    """Owns the model, the optimizer and the data pipeline."""

    model: nn.Module
    optimizer: Optimizer
    dataloader: DataLoader
    val_dataloader: DataLoader
    device: str
    autocast: AbstractContextManager[None]


def build_optimizer(name: str, model: nn.Module, learning_rate: float) -> Optimizer:
    """Return the optimizer registered under `name`."""
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}, expected one of {sorted(OPTIMIZERS)}")

    return torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=OPTIMIZERS[name])


def apply_precision(device: str, name: str) -> AbstractContextManager[None]:
    """Set the global fp32 matmul precision and return the autocast context `name` asks for."""
    if name not in PRECISIONS:
        raise KeyError(f"unknown precision {name!r}, expected one of {sorted(PRECISIONS)}")

    precision = PRECISIONS[name]
    torch.set_float32_matmul_precision(precision.matmul)
    dtype = precision.autocast_dtype

    # No GradScaler: bf16 keeps fp32's exponent range, only the mantissa shrinks.
    return nullcontext() if dtype is None else torch.autocast(device_type=device, dtype=dtype)


def build_dataloader(dataset: Dataset, batch_size: int, workers: int, *, train: bool) -> DataLoader:
    """Return a dataloader over `dataset`, shuffling and dropping the tail only when training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        drop_last=train,
    )


def get_trainer(config: TrainerConfig | None = None) -> Trainer:
    """Build every part described by `config` and hand them to a trainer."""
    config = config or TrainerConfig()
    device = select_device() if config.runtime.device == "auto" else config.runtime.device

    model_config = build_model_config(config.model)
    model: nn.Module = GPT(model_config).to(device)
    optimizer = build_optimizer(config.optim.name, model, config.optim.learning_rate)
    if config.runtime.compile:
        model = cast(nn.Module, torch.compile(model))
    shape = {"block_size": model_config.block_size, "vocab_size": model_config.vocab_size}
    # Seeds differ so the synthetic validation blocks never overlap the training ones.
    dataloader = build_dataloader(
        build_dataset(config.data.train, seed=0, **shape),
        batch_size=config.optim.micro_batch,
        workers=config.data.workers,
        train=True,
    )
    val_dataloader = build_dataloader(
        build_dataset(config.data.val, seed=1, **shape),
        batch_size=config.eval.batch or config.optim.micro_batch,
        workers=config.data.workers,
        train=False,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        device=device,
        autocast=apply_precision(device, config.runtime.precision),
    )


@torch.no_grad()
def evaluate(trainer: Trainer, eval_iters: int) -> float:
    """Return the mean validation loss over at most `eval_iters` batches."""
    trainer.model.eval()

    with trainer.autocast:
        losses = [
            trainer.model(inputs.to(trainer.device), targets.to(trainer.device))[1]
            for inputs, targets in islice(trainer.val_dataloader, eval_iters)
        ]

    trainer.model.train()

    return torch.stack(losses).mean().item()


def train(config: TrainerConfig | None = None, log_interval: int = 10) -> None:
    """Run a training loop on a single GPU."""
    config = config or TrainerConfig()
    trainer = get_trainer(config)

    for step, (x, y) in enumerate(trainer.dataloader):
        x, y = x.to(trainer.device), y.to(trainer.device)

        trainer.optimizer.zero_grad(set_to_none=True)
        with trainer.autocast:
            loss = trainer.model(x, y)[1]

        loss.backward()
        trainer.optimizer.step()

        # can be optimized by accumulating loss.detach()
        if step % log_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")

        if step % config.eval.interval == 0:
            print(f"step {step}: val loss {evaluate(trainer, config.eval.iters):.4f}")

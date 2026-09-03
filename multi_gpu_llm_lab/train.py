"""Train a model."""

from itertools import islice

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from multi_gpu_llm_lab.configs import TrainerConfig, build_model_config
from multi_gpu_llm_lab.data import build_dataset
from multi_gpu_llm_lab.device import select_device
from multi_gpu_llm_lab.model import GPT

OPTIMIZERS: dict[str, bool] = {"adamw": False, "adamw_fused": True}


class Trainer:
    """Owns the model, the optimizer and the data pipeline."""

    def __init__(
        self,
        model: GPT,
        optimizer: Optimizer,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device


def build_optimizer(name: str, model: GPT, learning_rate: float) -> Optimizer:
    """Return the optimizer registered under `name`."""
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}, expected one of {sorted(OPTIMIZERS)}")

    return torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=OPTIMIZERS[name])


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
    model = GPT(model_config).to(device)
    optimizer = build_optimizer(config.optim.name, model, config.optim.learning_rate)
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
    )


@torch.inference_mode()
def evaluate(trainer: Trainer, eval_iters: int) -> float:
    """Return the mean validation loss over at most `eval_iters` batches."""
    trainer.model.eval()

    # Losses stay on the device until the final mean: one sync per eval instead of one per batch.
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
        loss = trainer.model(x, y)[1]
        loss.backward()
        trainer.optimizer.step()

        # can be optimized by accumulating loss.detach()
        if step % log_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")

        if step % config.eval.interval == 0:
            print(f"step {step}: val loss {evaluate(trainer, config.eval.iters):.4f}")

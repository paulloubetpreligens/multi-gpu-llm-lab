"""Joins one process per GPU into a data-parallel group."""

import os
from dataclasses import dataclass

import torch
from torch import nn

from multi_gpu_llm_lab.configs import RuntimeConfig


@dataclass(frozen=True)
class Distributed:
    """This process' place in the group. the defaults describe a single-process run."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    ddp: bool = False


SINGLE_PROCESS = Distributed()


def resolve_distributed(runtime: RuntimeConfig, device: str) -> Distributed:
    """Read this process' rank from the environment torchrun set, rejecting a config that contradicts it."""
    if not runtime.ddp:
        return SINGLE_PROCESS

    if device != "cuda":
        raise ValueError(f"ddp needs a cuda device, got {device!r}; nccl is the only backend wired up")

    if "WORLD_SIZE" not in os.environ:
        raise ValueError("ddp needs the ranks torchrun spawns.")

    world_size = int(os.environ["WORLD_SIZE"])
    # world_size stays at its default when it is only a projection knob, so only a stated value can contradict.
    if runtime.world_size != 1 and runtime.world_size != world_size:
        raise ValueError(f"config world_size {runtime.world_size} contradicts the {world_size} ranks torchrun spawned")

    return Distributed(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=world_size,
        ddp=True,
    )


def init_distributed(runtime: RuntimeConfig, device: str) -> Distributed:
    """Join the process group torchrun set up, or stay single-process when DDP is off."""
    distributed = resolve_distributed(runtime, device)
    if not distributed.ddp:
        return distributed

    # set_device before any allocation, so plain "cuda" means this rank's GPU everywhere downstream.
    torch.cuda.set_device(distributed.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    return distributed


def shutdown_distributed() -> None:
    """Leave the process group, if this run ever joined one."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def unwrap(model: nn.Module) -> nn.Module:
    """Return the model under any DDP or compile wrapper, for the attributes only it carries."""
    return getattr(model, "module", model)

"""Assembles the recorder that measures a run and the writer that persists it."""

import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import nn

from multi_gpu_llm_lab.configs import TrainerConfig
from multi_gpu_llm_lab.metrics.probes import build_barrier, build_memory_probe
from multi_gpu_llm_lab.metrics.recorder import MetricsRecorder, model_flops_per_token
from multi_gpu_llm_lab.metrics.records import Hardware, RunSpec, Workload
from multi_gpu_llm_lab.metrics.writers import MetricsWriter, NullMetricsWriter, Writer
from multi_gpu_llm_lab.model import GPTConfig


def resolve_world_size(configured: int) -> int:
    """Prefer the live process group over the configured value, which cannot be verified."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return configured


def is_rank_zero() -> bool:
    """Only one rank may write, or the ranks interleave their rows into one corrupt table."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    return True


def git_revision(cwd: str | Path = Path(__file__).parent) -> str | None:
    """Return the revision under measurement, `-dirty` when the tree carries uncommitted work."""
    git = shutil.which("git")
    if git is None:
        return None

    try:
        # Every argument is a literal and git is resolved from PATH: no shell, no caller input.
        completed = subprocess.run(  # noqa: S603
            [git, "describe", "--always", "--dirty"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip()


def device_name(device: str) -> str:
    """Return the GPU a run measured, or the host architecture when it ran without one."""
    if device == "cuda":
        return torch.cuda.get_device_name()

    return platform.machine()


def build_run_id(config: TrainerConfig, device: str, world_size: int, started_at: datetime) -> str:
    """Return a sortable, self-describing identifier for this run."""
    if config.metrics.run_id:
        return config.metrics.run_id

    compiled = "-compile" if config.runtime.compile else ""
    slug = f"{config.model}-{device}-{config.runtime.precision}{compiled}-mb{config.optim.micro_batch}-ws{world_size}"

    return f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{slug}"


def build_metrics(config: TrainerConfig, model: nn.Module, device: str) -> tuple[MetricsRecorder, Writer]:
    """Assemble the recorder that measures a run and the writer that persists it."""
    world_size = resolve_world_size(config.runtime.world_size)
    model_config = cast(GPTConfig, model.config)
    count_params = cast(Callable[[], int], model.get_num_params)
    n_params = count_params()
    started_at = datetime.now(UTC)

    recorder = MetricsRecorder(
        workload=Workload(
            micro_batch=config.optim.micro_batch,
            block_size=model_config.block_size,
            world_size=world_size,
            flops_per_token=model_flops_per_token(model_config, n_params),
        ),
        hardware=Hardware(peak_flops=config.metrics.peak_flops, world_size=world_size),
        warmup_steps=config.metrics.warmup_steps,
        memory=build_memory_probe(device),
        clock=time.perf_counter,
        barrier=build_barrier(device),
    )

    if not config.metrics.enabled or not is_rank_zero():
        return recorder, NullMetricsWriter()

    spec = RunSpec(
        run_id=build_run_id(config, device, world_size, started_at),
        started_at=started_at.isoformat(),
        git_revision=git_revision(),
        torch_version=torch.__version__,
        model=config.model,
        device=device,
        device_name=device_name(device),
        precision=config.runtime.precision,
        compiled=config.runtime.compile,
        micro_batch=config.optim.micro_batch,
        block_size=model_config.block_size,
        world_size=world_size,
        n_params=n_params,
        flops_per_token=recorder.workload.flops_per_token,
        warmup_steps=config.metrics.warmup_steps,
        peak_flops=config.metrics.peak_flops,
        peak_flops_label=config.metrics.peak_flops_label,
    )

    return recorder, MetricsWriter(config.metrics.out_dir, spec)

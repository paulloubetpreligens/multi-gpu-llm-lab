"""Times each step and folds the steady-state ones into a run summary."""

import math
import statistics
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from multi_gpu_llm_lab.metrics.probes import Barrier, MemoryProbe, no_barrier
from multi_gpu_llm_lab.metrics.records import Hardware, RunSummary, StepRecord, StepScope, Workload
from multi_gpu_llm_lab.model import GPTConfig

Clock = Callable[[], float]


def model_flops_per_token(config: GPTConfig, n_params: int) -> int:
    """Return the matmul FLOPs one token costs in a forward and backward pass."""
    head_dim = config.n_embd // config.n_head

    return 6 * n_params + 12 * config.n_layer * config.n_head * head_dim * config.block_size


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile, always a value that was actually observed."""
    ordered = sorted(values)

    return ordered[math.ceil(fraction * len(ordered)) - 1]


@dataclass
class MetricsRecorder:
    """Times each step and folds the steady-state ones into a run summary."""

    workload: Workload
    hardware: Hardware
    warmup_steps: int
    memory: MemoryProbe
    clock: Clock = time.perf_counter
    barrier: Barrier = no_barrier
    records: list[StepRecord] = field(default_factory=list)
    _wall_start: float = field(default=0.0, init=False, repr=False)
    _wall_seconds: float = field(default=0.0, init=False, repr=False)

    @contextmanager
    def step(self, index: int) -> Generator[StepScope]:
        """Measure one training step and append its record."""
        scope = StepScope()

        self.barrier()
        start = self.clock()

        yield scope

        self.barrier()
        end = self.clock()
        seconds = end - start

        if not self.records or index == self.warmup_steps:
            self._wall_start = start
        self._wall_seconds = end - self._wall_start

        self.records.append(self._derive(index, seconds, scope.loss))

        if index + 1 == self.warmup_steps:
            self.memory.reset()

    def summarize(self, status: str = "completed") -> RunSummary:
        """Fold the steady-state steps into the run's single comparable row."""
        measured = [record for record in self.records if not record.warmup]
        if not measured:
            return self._empty_summary(status)

        step_ms = [record.step_ms for record in measured]
        seconds = sum(step_ms) / 1000
        tokens = sum(record.tokens for record in measured)
        tokens_per_second = tokens / seconds

        return RunSummary(
            status=status,
            total_steps=len(self.records),
            measured_steps=len(measured),
            measured_seconds=seconds,
            wall_seconds=self._wall_seconds,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
            mfu=tokens_per_second * self.workload.flops_per_token / self.hardware.peak_total,
            step_ms_median=statistics.median(step_ms),
            step_ms_p90=percentile(step_ms, 0.9),
            peak_allocated=measured[-1].peak_allocated,
            peak_reserved=measured[-1].peak_reserved,
            final_loss=measured[-1].loss,
        )

    def _empty_summary(self, status: str) -> RunSummary:
        return RunSummary(
            status=status,
            total_steps=len(self.records),
            measured_steps=0,
            measured_seconds=0.0,
            wall_seconds=self._wall_seconds,
            tokens=0,
            tokens_per_second=None,
            mfu=None,
            step_ms_median=None,
            step_ms_p90=None,
            peak_allocated=None,
            peak_reserved=None,
            final_loss=None,
        )

    def _derive(self, index: int, seconds: float, loss: Any) -> StepRecord:
        tokens = self.workload.tokens_per_step
        tokens_per_second = tokens / seconds if seconds > 0 else 0.0
        peaks = self.memory.peaks()

        return StepRecord(
            step=index,
            warmup=index < self.warmup_steps,
            step_ms=seconds * 1000,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
            mfu=tokens_per_second * self.workload.flops_per_token / self.hardware.peak_total,
            loss=None if loss is None else float(loss),
            peak_allocated=peaks.allocated,
            peak_reserved=peaks.reserved,
        )

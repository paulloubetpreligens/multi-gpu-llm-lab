"""CSV persistence for the per-run step rows and the shared run row."""

import csv
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from multi_gpu_llm_lab.metrics.records import RunSpec, RunSummary, StepRecord

StepCell = Callable[[StepRecord, RunSpec], object]
RunCell = Callable[[RunSummary, RunSpec], object]


class Writer(Protocol):
    """Where step rows and the run row are persisted."""

    def write_step(self, record: StepRecord) -> None:
        """Persist one step row."""
        ...

    def write_run(self, summary: RunSummary) -> None:
        """Persist the run's summary row."""
        ...

    def close(self) -> None:
        """Release whatever the writer holds open."""
        ...


# One literal per table: the header text and the value filling it cannot drift apart.
STEP_ROW: Mapping[str, StepCell] = {
    "run_id": lambda record, spec: spec.run_id,
    "step": lambda record, spec: record.step,
    "warmup": lambda record, spec: record.warmup,
    "step_ms": lambda record, spec: record.step_ms,
    "tokens": lambda record, spec: record.tokens,
    "tokens_per_second": lambda record, spec: record.tokens_per_second,
    "mfu": lambda record, spec: record.mfu,
    "loss": lambda record, spec: record.loss,
    "peak_allocated": lambda record, spec: record.peak_allocated,
    "peak_reserved": lambda record, spec: record.peak_reserved,
    "world_size": lambda record, spec: spec.world_size,
    "flops_per_token": lambda record, spec: spec.flops_per_token,
    "peak_flops": lambda record, spec: spec.peak_flops,
    "peak_flops_label": lambda record, spec: spec.peak_flops_label,
}

RUN_ROW: Mapping[str, RunCell] = {
    "run_id": lambda summary, spec: spec.run_id,
    "started_at": lambda summary, spec: spec.started_at,
    "status": lambda summary, spec: summary.status,
    "git_revision": lambda summary, spec: spec.git_revision,
    "torch_version": lambda summary, spec: spec.torch_version,
    "model": lambda summary, spec: spec.model,
    "device": lambda summary, spec: spec.device,
    "device_name": lambda summary, spec: spec.device_name,
    "precision": lambda summary, spec: spec.precision,
    "compiled": lambda summary, spec: spec.compiled,
    "micro_batch": lambda summary, spec: spec.micro_batch,
    "block_size": lambda summary, spec: spec.block_size,
    "world_size": lambda summary, spec: spec.world_size,
    "n_params": lambda summary, spec: spec.n_params,
    "flops_per_token": lambda summary, spec: spec.flops_per_token,
    "peak_flops": lambda summary, spec: spec.peak_flops,
    "peak_flops_label": lambda summary, spec: spec.peak_flops_label,
    "warmup_steps": lambda summary, spec: spec.warmup_steps,
    "total_steps": lambda summary, spec: summary.total_steps,
    "measured_steps": lambda summary, spec: summary.measured_steps,
    "measured_seconds": lambda summary, spec: summary.measured_seconds,
    "wall_seconds": lambda summary, spec: summary.wall_seconds,
    "tokens": lambda summary, spec: summary.tokens,
    "tokens_per_second": lambda summary, spec: summary.tokens_per_second,
    "mfu": lambda summary, spec: summary.mfu,
    "step_ms_median": lambda summary, spec: summary.step_ms_median,
    "step_ms_p90": lambda summary, spec: summary.step_ms_p90,
    "peak_allocated": lambda summary, spec: summary.peak_allocated,
    "peak_reserved": lambda summary, spec: summary.peak_reserved,
    "final_loss": lambda summary, spec: summary.final_loss,
}

STEP_COLUMNS = tuple(STEP_ROW)
RUN_COLUMNS = tuple(RUN_ROW)


class MetricsWriter(Writer):
    """Appends step rows to a per-run steps.csv and one summary row to a shared runs.csv."""

    def __init__(self, out_dir: Path | str, spec: RunSpec) -> None:
        self.spec = spec
        self.out_dir = Path(out_dir)
        self.runs_path = self.out_dir / "runs.csv"
        run_dir = self.out_dir / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._steps_file = (run_dir / "steps.csv").open("w", newline="", encoding="utf-8")
        self._steps_writer = csv.writer(self._steps_file)
        self._steps_writer.writerow(STEP_COLUMNS)
        self._steps_file.flush()

    def write_step(self, record: StepRecord) -> None:
        """Append one step row and flush it, so a crashed run keeps what it measured."""
        self._steps_writer.writerow([cell(record, self.spec) for cell in STEP_ROW.values()])
        self._steps_file.flush()

    def write_run(self, summary: RunSummary) -> None:
        """Append this run's single comparable row to the shared table."""
        needs_header = not self.runs_path.exists() or self.runs_path.stat().st_size == 0

        with self.runs_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(RUN_COLUMNS)

            writer.writerow([cell(summary, self.spec) for cell in RUN_ROW.values()])

    def close(self) -> None:
        """Close the per-run step file."""
        self._steps_file.close()


class NullMetricsWriter(Writer):
    """Discards everything; used off rank zero and when metrics are switched off."""

    def write_step(self, record: StepRecord) -> None:
        """Discard the step row."""

    def write_run(self, summary: RunSummary) -> None:
        """Discard the run row."""

    def close(self) -> None:
        """Nothing to close."""

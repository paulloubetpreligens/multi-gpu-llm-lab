"""Value objects describing what a run costs and what it measured."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Hardware:
    """Reference peak throughput a run's MFU is scored against."""

    peak_flops: float
    """Datasheet FLOP/s of one device at the run's dtype."""

    world_size: int = 1
    """Ranks taking part in the run."""

    @property
    def peak_total(self) -> float:
        """Peak FLOPs of every rank taken together."""
        return self.peak_flops * self.world_size


@dataclass(frozen=True)
class Workload:
    """What one step costs (in tokens and in model FLOPs)."""

    micro_batch: int
    """Sequences one rank processes per step."""

    block_size: int
    """Tokens per sequence."""

    world_size: int
    """Ranks taking part in the run."""

    flops_per_token: int
    """Model FLOPs one token costs, forward and backward together."""

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one step across every rank."""
        return self.micro_batch * self.block_size * self.world_size


@dataclass(frozen=True)
class StepRecord:
    """One training step, as it lands in steps.csv."""

    step: int
    """Index in the run, counting warmup."""

    warmup: bool
    """True while the step is excluded from the summary."""

    step_ms: float
    """Wall time of the step, device barrier included."""

    tokens: int
    """Tokens the step consumed across every rank."""

    tokens_per_second: float
    """This step's throughput."""

    mfu: float
    """Share of the hardware's total peak FLOPs this step actually used."""

    loss: float | None
    """Training loss, None when the step reported none."""

    peak_allocated: int | None
    """Bytes in live tensors at the allocator's high-water mark."""

    peak_reserved: int | None
    """Bytes the allocator held from the driver at its high-water mark."""


@dataclass(frozen=True)
class RunSummary:
    """The steady-state verdict on a run, as it lands in runs.csv."""

    status: str
    """How the run ended: completed, or crashed."""

    total_steps: int
    """Steps recorded, warmup included."""

    measured_steps: int
    """Steps after warmup, the only ones scored below."""

    measured_seconds: float
    """Sum of the measured steps' own times, excluding everything between them."""

    wall_seconds: float
    """First measured step to the last, dataloading and evaluation included."""

    tokens: int
    """Tokens consumed over the measured steps."""

    tokens_per_second: float | None
    """The run's headline throughput; None when nothing was measured."""

    mfu: float | None
    """Share of the hardware's total peak FLOPs the run sustained."""

    step_ms_median: float | None
    """The typical step, immune to a few slow ones."""

    step_ms_p90: float | None
    """The tail step: an observed value, never an interpolated one."""

    peak_allocated: int | None
    """Live-tensor high-water mark at the last measured step."""

    peak_reserved: int | None
    """Allocator-reserved high-water mark at the last measured step."""

    final_loss: float | None
    """Loss at the last measured step; a sanity check, not a convergence claim."""


@dataclass
class StepScope:
    """Carries the step's loss tensor out of the timed region."""

    loss: Any = None


@dataclass(frozen=True)
class RunSpec:
    """Everything a run row needs to describe itself without joining another table."""

    run_id: str
    """Identifies the run and names the directory holding its steps.csv."""

    started_at: str
    """UTC start time, ISO 8601."""

    git_revision: str | None
    """Code under measurement, `-dirty` when uncommitted; None outside a checkout."""

    torch_version: str
    """Torch the run executed on, since throughput moves between releases."""

    model: str
    """Name of the architecture preset, a control variable across runs."""

    device: str
    """Backend the run used: cpu, cuda or mps."""

    device_name: str
    """The GPU the run measured, or the host architecture when it ran without one."""

    precision: str
    """Autocast dtype: fp32, tf32 or bf16."""

    compiled: bool
    """Whether torch.compile was applied to the model."""

    micro_batch: int
    """Sequences one rank processes per step."""

    block_size: int
    """Tokens per sequence."""

    world_size: int
    """Ranks taking part in the run, read from the live process group when there is one."""

    n_params: int
    """Parameter count of the model, embeddings excluded per get_num_params."""

    flops_per_token: int
    """Model FLOPs one token costs, forward and backward together."""

    warmup_steps: int
    """Leading steps excluded from the summary, where compilation and caches settle."""

    peak_flops: float
    """Denominator behind every MFU in this run."""

    peak_flops_label: str
    """What that denominator refers to, so a stale peak is visible rather than silent."""

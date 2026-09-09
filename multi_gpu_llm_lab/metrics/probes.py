"""Device seam: allocator high-water marks and the barrier that drains queued work."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

Barrier = Callable[[], None]


@dataclass(frozen=True)
class MemoryPeaks:
    """Allocator high-water marks in bytes; None where the backend cannot report them."""

    allocated: int | None
    reserved: int | None


NO_PEAKS = MemoryPeaks(allocated=None, reserved=None)


class MemoryProbe(Protocol):
    """Allocator high-water marks for the device a run trains on."""

    def reset(self) -> None:
        """Restart the high-water marks from the current occupancy."""
        ...

    def peaks(self) -> MemoryPeaks:
        """Return the current high-water marks."""
        ...


class NullMemoryProbe(MemoryProbe):
    """Peaks for a backend that cannot report them."""

    def reset(self) -> None:
        """Nothing to restart."""

    def peaks(self) -> MemoryPeaks:
        """Return no peaks at all."""
        return NO_PEAKS


class CudaMemoryProbe(MemoryProbe):
    """Peaks straight from the CUDA caching allocator."""

    def reset(self) -> None:
        """Restart the high-water marks; already-reserved blocks are not released."""
        torch.cuda.reset_peak_memory_stats()

    def peaks(self) -> MemoryPeaks:
        """Return the CUDA allocator's high-water marks."""
        return MemoryPeaks(
            allocated=torch.cuda.max_memory_allocated(),
            reserved=torch.cuda.max_memory_reserved(),
        )


def build_memory_probe(device: str) -> MemoryProbe:
    """Return the probe that can speak for `device`."""
    return CudaMemoryProbe() if device == "cuda" else NullMemoryProbe()


def build_barrier(device: str) -> Barrier:
    """Return a callable draining the queued work of `device`."""
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize

    return no_barrier


def no_barrier() -> None:
    """Drain nothing, for a backend that runs synchronously."""

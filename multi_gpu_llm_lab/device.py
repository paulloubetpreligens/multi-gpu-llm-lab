"""Select the training device."""

import torch


def select_device() -> str:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""Token stream datasets."""

from pathlib import Path

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.uint16
TOKEN_NBYTES = np.dtype(TOKEN_DTYPE).itemsize


class TokenDataset(Dataset):
    """Fixed-length blocks over a memmapped uint16 token shard."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self._tokens: np.memmap | None = None

        n_tokens = self.path.stat().st_size // TOKEN_NBYTES
        self._length = (n_tokens - 1) // block_size

    @property
    def tokens(self) -> np.memmap:
        """Memmap of the shard, opened on first access."""
        # Opened on first access so each dataloader worker maps the file itself.
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")

        return self._tokens

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        window = self.tokens[start : start + self.block_size + 1].astype(np.int64)

        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


class SyntheticDataset(Dataset):
    """Random tokens, for wiring the loop without data on disk."""

    def __init__(self, block_size: int, n_blocks: int, vocab_size: int, seed: int = 0) -> None:
        generator = np.random.default_rng(seed)
        self._windows = generator.integers(0, vocab_size, size=(n_blocks, block_size + 1), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._windows[index]

        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


def encode_text(text: str) -> np.ndarray:
    """Encode text to GPT-2 BPE ids, the tokenizer the model presets are sized for."""
    # np.array rejects an out-of-range id; astype would wrap it silently into a valid token.
    return np.array(tiktoken.get_encoding("gpt2").encode_ordinary(text), dtype=TOKEN_DTYPE)


def write_shards(tokens: np.ndarray, out_dir: str | Path, val_fraction: float = 0.1) -> tuple[Path, Path]:
    """Write `tokens` as the train and val shards under `out_dir`, returning both paths."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie between 0 and 1, got {val_fraction}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = int(len(tokens) * (1.0 - val_fraction))
    paths = (out_dir / "train.bin", out_dir / "val.bin")

    for path, shard in zip(paths, (tokens[:split], tokens[split:]), strict=True):
        shard.tofile(path)

    return paths


def prepare_shards(text_path: str | Path, out_dir: str | Path, val_fraction: float = 0.1) -> tuple[Path, Path]:
    """Tokenize a text file into the shards `build_dataset` reads."""
    # ponytail: the whole corpus is held in memory, tokenize in chunks once one stops fitting.
    return write_shards(encode_text(Path(text_path).read_text()), out_dir, val_fraction)


def build_dataset(name: str, block_size: int, vocab_size: int, n_blocks: int = 64, seed: int = 0) -> Dataset:
    """Resolve a dataset name or shard path to a dataset."""
    if name == "synthetic":
        return SyntheticDataset(block_size=block_size, n_blocks=n_blocks, vocab_size=vocab_size, seed=seed)

    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no token shard at {name!r}; use 'synthetic' to run without data")

    return TokenDataset(path, block_size=block_size)

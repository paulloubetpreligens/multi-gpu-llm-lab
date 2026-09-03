"""Named model presets.

The architecture is a control variable: locked once, never tuned for MFU.
"""

from dataclasses import replace

from multi_gpu_llm_lab.model import GPTConfig

# 50304 = GPT-2's 50257 padded to a multiple of 64 so the vocabulary GEMM aligns
# on tensor cores. Available only because we train from random init.
MODEL_CONFIGS: dict[str, GPTConfig] = {
    "tiny": GPTConfig(n_layer=2, n_head=2, n_embd=128, block_size=128, vocab_size=50304),
    "350m": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=50304),
}


def build_model_config(name: str) -> GPTConfig:
    """Return a fresh copy of the config registered under `name`."""
    if name not in MODEL_CONFIGS:
        raise KeyError(f"unknown model preset {name!r}, expected one of {sorted(MODEL_CONFIGS)}")

    return replace(MODEL_CONFIGS[name])

"""Named model presets and the trainer configuration schema.

The architecture is a control variable: locked once, never tuned for MFU.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

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


@dataclass
class RuntimeConfig:
    """Where the run executes."""

    device: str = "auto"


@dataclass
class DataConfig:
    """Token shards, or 'synthetic' to run without data on disk."""

    train: str = "synthetic"
    val: str = "synthetic"
    workers: int = 0


@dataclass
class OptimConfig:
    """Optimizer name and the knobs it reads."""

    name: str = "adamw"
    learning_rate: float = 6e-4
    micro_batch: int = 4


@dataclass
class EvalConfig:
    """Validation cadence and cost; `batch` falls back to `optim.micro_batch`."""

    interval: int = 100
    iters: int = 20
    batch: int | None = None


@dataclass
class TrainerConfig:
    """Names resolved to objects by `get_trainer`.

    Every optimization defaults OFF: this is the unoptimized baseline.
    """

    model: str = "tiny"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_document(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a nested mapping."""
    # safe_load, never load: a config file must not be able to construct Python objects.
    return yaml.safe_load(Path(path).read_text()) or {}


def build_trainer_config(document: Mapping[str, Any]) -> TrainerConfig:
    """Build a config from a nested mapping, rejecting unknown keys."""
    return _build(TrainerConfig, document)


def config_paths(config: Any = None, prefix: str = "") -> Iterator[str]:
    """Yield the dotted path of every leaf of the config tree."""
    config = TrainerConfig() if config is None else config

    for config_field in fields(config):
        value = getattr(config, config_field.name)
        path = f"{prefix}{config_field.name}"

        if is_dataclass(value):
            yield from config_paths(value, prefix=f"{path}.")
        else:
            yield path


def parse_overrides(assignments: Sequence[str]) -> dict[str, Any]:
    """Fold `key.path=value` assignments into a nested document, reading values as YAML scalars."""
    document: dict[str, Any] = {}

    for assignment in assignments:
        path, separator, value = assignment.partition("=")
        if not separator:
            raise ValueError(f"override {assignment!r} is not of the form key=value")

        *sections, name = path.split(".")
        target = document
        for section in sections:
            target = target.setdefault(section, {})

        target[name] = yaml.safe_load(value)

    return document


def merge(document: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` over `document`, section by section."""
    merged = dict(document)

    for key, value in override.items():
        current = merged.get(key)
        merged[key] = merge(current, value) if isinstance(current, Mapping) and isinstance(value, Mapping) else value

    return merged


def resolve_trainer_config(path: str | Path | None = None, assignments: Sequence[str] = ()) -> TrainerConfig:
    """Build the config from an optional YAML file and the assignments that override it."""
    document = load_document(path) if path else {}

    return build_trainer_config(merge(document, parse_overrides(assignments)))


def _build(config_class: type, document: Mapping[str, Any]) -> Any:
    annotations = get_type_hints(config_class)
    unknown = sorted(set(document) - set(annotations))
    if unknown:
        raise KeyError(f"unknown key(s) {unknown} in {config_class.__name__}, expected {sorted(annotations)}")

    return config_class(
        **{
            name: _build(annotations[name], value)
            if is_dataclass(annotations[name])
            else _coerce(annotations[name], value)
            for name, value in document.items()
        }
    )


def _coerce(annotation: Any, value: Any) -> Any:
    # YAML resolves `1e-3` to a string, so a leaf is cast to its annotated type rather than trusted.
    target = next((arg for arg in get_args(annotation) if arg is not type(None)), annotation)

    return value if value is None or isinstance(value, target) else target(value)

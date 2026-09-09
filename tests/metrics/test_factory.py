import platform
import re

import torch

from multi_gpu_llm_lab.configs import MetricsConfig, RuntimeConfig, TrainerConfig
from multi_gpu_llm_lab.metrics.factory import build_metrics, device_name, git_revision
from multi_gpu_llm_lab.model import GPT, GPTConfig

REVISION = re.compile(r"^[0-9a-f]{7,40}(-dirty)?$")


def build_config(tmp_path, **metrics):
    return TrainerConfig(
        runtime=RuntimeConfig(device="cpu"),
        metrics=MetricsConfig(out_dir=str(tmp_path), **metrics),
    )


def build_model():
    return GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=16))


def test_build_metrics_scores_mfu_against_the_configured_peak_flops(tmp_path):
    config = build_config(tmp_path, peak_flops=1.5e13)

    recorder, _ = build_metrics(config, build_model(), "cpu")

    assert recorder.hardware.peak_flops == 1.5e13


def test_build_metrics_records_the_configured_peak_flops_label(tmp_path):
    config = build_config(tmp_path, peak_flops_label="RTX 4090 bf16 dense")

    _, writer = build_metrics(config, build_model(), "cpu")

    assert writer.spec.peak_flops_label == "RTX 4090 bf16 dense"


def test_build_metrics_records_the_torch_version(tmp_path):
    config = build_config(tmp_path)

    _, writer = build_metrics(config, build_model(), "cpu")

    assert writer.spec.torch_version == torch.__version__


def test_build_metrics_records_the_hardware_the_run_measured(tmp_path):
    config = build_config(tmp_path)

    _, writer = build_metrics(config, build_model(), "cpu")

    assert writer.spec.device_name == platform.machine()


def test_build_metrics_records_the_revision_of_the_code_under_measurement(tmp_path):
    config = build_config(tmp_path)

    _, writer = build_metrics(config, build_model(), "cpu")

    assert REVISION.match(writer.spec.git_revision)


def test_git_revision_outside_a_checkout_returns_none(tmp_path):
    assert git_revision(tmp_path) is None


def test_device_name_without_a_gpu_returns_the_host_architecture():
    assert device_name("cpu") == platform.machine()

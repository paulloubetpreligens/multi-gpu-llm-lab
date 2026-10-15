import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from multi_gpu_llm_lab.configs import RuntimeConfig, build_model_config
from multi_gpu_llm_lab.distributed import (
    Distributed,
    init_distributed,
    resolve_distributed,
    shutdown_distributed,
    unwrap,
)
from multi_gpu_llm_lab.model import GPT


@pytest.fixture
def two_ranks(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")


def test_init_distributed_without_ddp_returns_a_single_process_group():
    runtime = RuntimeConfig(device="cpu")

    distributed = init_distributed(runtime, device="cpu")

    assert distributed == Distributed(rank=0, local_rank=0, world_size=1)


def test_init_distributed_with_ddp_on_a_non_cuda_device_names_the_device_in_the_error():
    runtime = RuntimeConfig(device="auto", ddp=True)

    with pytest.raises(ValueError, match="mps"):
        init_distributed(runtime, device="mps")


def test_init_distributed_with_ddp_outside_torchrun_points_at_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    runtime = RuntimeConfig(device="cuda", ddp=True)

    with pytest.raises(ValueError, match="torchrun"):
        init_distributed(runtime, device="cuda")


def test_init_distributed_with_a_world_size_torchrun_contradicts_raises_error(two_ranks):
    runtime = RuntimeConfig(device="cuda", ddp=True, world_size=4)

    with pytest.raises(ValueError, match="world_size"):
        init_distributed(runtime, device="cuda")


def test_resolve_distributed_with_ddp_adopts_the_ranks_torchrun_assigned(two_ranks):
    runtime = RuntimeConfig(device="cuda", ddp=True)

    distributed = resolve_distributed(runtime, device="cuda")

    assert distributed == Distributed(rank=1, local_rank=1, world_size=2, ddp=True)


@pytest.fixture
def one_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")


def test_resolve_distributed_under_a_single_rank_launcher_still_reports_ddp(one_rank):
    runtime = RuntimeConfig(device="cuda", ddp=True)

    distributed = resolve_distributed(runtime, device="cuda")

    assert distributed.ddp


def test_resolve_distributed_without_ddp_reports_no_ddp():
    runtime = RuntimeConfig(device="cpu")

    distributed = resolve_distributed(runtime, device="cpu")

    assert not distributed.ddp


@pytest.fixture
def gloo_group(tmp_path):
    # A one-rank gloo group over a file store: the only way to build a real DDP wrapper without CUDA.
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=0,
        world_size=1,
    )

    yield

    shutdown_distributed()


@pytest.fixture
def model():
    return GPT(build_model_config("tiny"))


def test_unwrap_on_a_bare_model_returns_the_model_itself(model):
    assert unwrap(model) is model


def test_unwrap_on_a_ddp_wrapped_model_returns_the_model_it_replicates(gloo_group, model):
    wrapped = DistributedDataParallel(model)

    assert unwrap(wrapped) is model


def test_unwrap_on_a_compiled_ddp_wrapped_model_reaches_the_model_config(gloo_group, model):
    wrapped = torch.compile(DistributedDataParallel(model))

    assert unwrap(wrapped).config is model.config


def test_shutdown_distributed_leaves_the_process_group(gloo_group):
    shutdown_distributed()

    assert not torch.distributed.is_initialized()


def test_shutdown_distributed_without_a_group_does_nothing():
    shutdown_distributed()

    assert not torch.distributed.is_initialized()

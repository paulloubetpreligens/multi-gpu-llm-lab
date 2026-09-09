import csv
import math
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from multi_gpu_llm_lab.configs import DataConfig, MetricsConfig, OptimConfig, RuntimeConfig, TrainerConfig
from multi_gpu_llm_lab.data import SyntheticDataset
from multi_gpu_llm_lab.model import GPT
from multi_gpu_llm_lab.train import Trainer, apply_precision, evaluate, get_trainer, train


@pytest.fixture(autouse=True)
def restore_matmul_precision():
    previous = torch.get_float32_matmul_precision()

    yield

    torch.set_float32_matmul_precision(previous)


@pytest.fixture
def config():
    return TrainerConfig(
        model="tiny",
        runtime=RuntimeConfig(device="cpu"),
        data=DataConfig(train="synthetic", val="synthetic"),
        optim=OptimConfig(name="adamw"),
    )


def test_trainer_holds_the_parts_it_is_given():
    dataloader = DataLoader(SyntheticDataset(block_size=4, n_blocks=2, vocab_size=8))
    val_dataloader = DataLoader(SyntheticDataset(block_size=4, n_blocks=2, vocab_size=8, seed=1))

    trainer = Trainer(
        model="m",
        optimizer="o",
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        device="cpu",
        autocast=nullcontext(),
    )

    assert (trainer.model, trainer.optimizer, trainer.dataloader, trainer.val_dataloader) == (
        "m",
        "o",
        dataloader,
        val_dataloader,
    )


def test_get_trainer_builds_the_named_model_preset(config):
    trainer = get_trainer(config)

    assert trainer.model.config.n_layer == 2


def test_get_trainer_with_a_fused_optimizer_name_sets_the_fused_flag(config):
    config.optim.name = "adamw_fused"

    trainer = get_trainer(config)

    assert trainer.optimizer.param_groups[0]["fused"] is True


def test_get_trainer_defaults_to_an_unfused_optimizer(config):
    trainer = get_trainer(config)

    assert trainer.optimizer.param_groups[0]["fused"] is not True


def test_get_trainer_with_unknown_optimizer_raises_error(config):
    config.optim.name = "shampoo"

    with pytest.raises(KeyError):
        get_trainer(config)


def test_get_trainer_yields_batches_shaped_for_the_model(config):
    trainer = get_trainer(config)

    inputs, _ = next(iter(trainer.dataloader))

    assert inputs.shape == (config.optim.micro_batch, trainer.model.config.block_size)


def test_get_trainer_yields_long_tensors_for_embedding_lookup(config):
    trainer = get_trainer(config)

    inputs, _ = next(iter(trainer.dataloader))

    assert inputs.dtype == torch.long


def test_get_trainer_places_the_model_on_the_requested_device(config):
    trainer = get_trainer(config)

    assert next(trainer.model.parameters()).device.type == "cpu"


def test_get_trainer_without_a_config_uses_the_defaults():
    trainer = get_trainer()

    assert trainer.model.config.n_layer == 2


def test_get_trainer_leaves_the_model_uncompiled(config):
    trainer = get_trainer(config)

    assert isinstance(trainer.model, GPT)


def test_get_trainer_builds_a_validation_dataloader_separate_from_the_training_one(config):
    trainer = get_trainer(config)

    assert trainer.val_dataloader is not trainer.dataloader


def test_get_trainer_does_not_shuffle_the_validation_dataloader(config):
    trainer = get_trainer(config)

    assert isinstance(trainer.val_dataloader.sampler, SequentialSampler)


def test_get_trainer_keeps_every_validation_batch(config):
    trainer = get_trainer(config)

    assert trainer.val_dataloader.drop_last is False


def test_get_trainer_builds_disjoint_train_and_validation_blocks(config):
    trainer = get_trainer(config)

    train_inputs, _ = next(iter(trainer.dataloader))
    val_inputs, _ = next(iter(trainer.val_dataloader))

    assert not torch.equal(train_inputs, val_inputs)


def test_get_trainer_without_an_eval_batch_reuses_the_micro_batch(config):
    trainer = get_trainer(config)

    assert trainer.val_dataloader.batch_size == config.optim.micro_batch


def test_get_trainer_with_an_eval_batch_overrides_the_validation_batch_size(config):
    config.eval.batch = 2

    trainer = get_trainer(config)

    assert trainer.val_dataloader.batch_size == 2


def test_evaluate_returns_a_finite_loss(config):
    trainer = get_trainer(config)

    loss = evaluate(trainer, eval_iters=2)

    assert math.isfinite(loss)


def test_evaluate_with_one_eval_iter_returns_the_loss_of_the_first_batch(config):
    trainer = get_trainer(config)
    inputs, targets = next(iter(trainer.val_dataloader))

    loss = evaluate(trainer, eval_iters=1)

    assert loss == pytest.approx(trainer.model(inputs, targets)[1].item())


def test_evaluate_with_more_eval_iters_than_batches_averages_what_it_has(config):
    trainer = get_trainer(config)

    loss = evaluate(trainer, eval_iters=10_000)

    assert math.isfinite(loss)


def test_evaluate_runs_the_forward_pass_without_recording_gradients(config):
    trainer = get_trainer(config)
    grad_enabled: list[bool] = []
    trainer.model.register_forward_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))

    evaluate(trainer, eval_iters=1)

    assert grad_enabled == [False]


def test_evaluate_leaves_the_model_in_training_mode(config):
    trainer = get_trainer(config)

    evaluate(trainer, eval_iters=1)

    assert trainer.model.training is True


def test_apply_precision_with_bf16_enables_autocast_on_the_device():
    context = apply_precision("cpu", "bf16")

    with context:
        assert torch.is_autocast_enabled("cpu")


def test_apply_precision_with_fp32_leaves_autocast_disabled():
    context = apply_precision("cpu", "fp32")

    with context:
        assert not torch.is_autocast_enabled("cpu")


def test_apply_precision_with_tf32_leaves_autocast_disabled():
    context = apply_precision("cpu", "tf32")

    with context:
        assert not torch.is_autocast_enabled("cpu")


def test_apply_precision_with_fp32_keeps_the_matmuls_exact():
    apply_precision("cpu", "fp32")

    assert torch.get_float32_matmul_precision() == "highest"


def test_apply_precision_with_tf32_lets_the_matmuls_use_tensor_cores():
    apply_precision("cpu", "tf32")

    assert torch.get_float32_matmul_precision() == "high"


def test_apply_precision_with_bf16_lets_the_matmuls_use_tensor_cores():
    apply_precision("cpu", "bf16")

    assert torch.get_float32_matmul_precision() == "high"


def test_apply_precision_with_an_unknown_name_raises_error():
    with pytest.raises(KeyError):
        apply_precision("cpu", "fp8")


def test_get_trainer_with_bf16_runs_the_model_activations_in_bf16(config):
    config.runtime.precision = "bf16"
    trainer = get_trainer(config)
    inputs, targets = next(iter(trainer.val_dataloader))
    dtypes: list[torch.dtype] = []
    trainer.model.transformer.h[0].attn.c_attn.register_forward_hook(
        lambda module, args, output: dtypes.append(output.dtype)
    )

    with trainer.autocast:
        trainer.model(inputs, targets)

    assert dtypes == [torch.bfloat16]


def test_get_trainer_with_bf16_keeps_the_parameters_in_fp32(config):
    config.runtime.precision = "bf16"

    trainer = get_trainer(config)

    assert next(trainer.model.parameters()).dtype == torch.float32


def test_get_trainer_defaults_to_fp32_activations(config):
    trainer = get_trainer(config)
    inputs, targets = next(iter(trainer.val_dataloader))
    dtypes: list[torch.dtype] = []
    trainer.model.transformer.h[0].attn.c_attn.register_forward_hook(
        lambda module, args, output: dtypes.append(output.dtype)
    )

    with trainer.autocast:
        trainer.model(inputs, targets)

    assert dtypes == [torch.float32]


def test_get_trainer_with_compile_wraps_the_model(config):
    config.runtime.compile = True

    trainer = get_trainer(config)

    assert isinstance(trainer.model, torch._dynamo.OptimizedModule)


def test_get_trainer_with_tf32_keeps_the_activations_in_fp32(config):
    config.runtime.precision = "tf32"
    trainer = get_trainer(config)
    inputs, targets = next(iter(trainer.val_dataloader))
    dtypes: list[torch.dtype] = []
    trainer.model.transformer.h[0].attn.c_attn.register_forward_hook(
        lambda module, args, output: dtypes.append(output.dtype)
    )

    with trainer.autocast:
        trainer.model(inputs, targets)

    assert dtypes == [torch.float32]


@pytest.fixture
def metrics_config(config, tmp_path):
    config.metrics = MetricsConfig(out_dir=str(tmp_path), warmup_steps=1, run_id="run-under-test")
    config.eval.interval = 10_000

    return config


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_train_writes_one_step_row_per_optimizer_step(metrics_config, tmp_path):
    train(metrics_config)

    assert len(read_rows(tmp_path / "run-under-test" / "steps.csv")) == 16


def test_train_appends_exactly_one_summary_row_per_run(metrics_config, tmp_path):
    train(metrics_config)

    assert len(read_rows(tmp_path / "runs.csv")) == 1


def test_train_reports_a_positive_tokens_per_second(metrics_config, tmp_path):
    train(metrics_config)

    assert float(read_rows(tmp_path / "runs.csv")[0]["tokens_per_second"]) > 0


def test_train_records_the_configured_world_size_in_the_run_row(metrics_config, tmp_path):
    metrics_config.runtime.world_size = 4

    train(metrics_config)

    assert read_rows(tmp_path / "runs.csv")[0]["world_size"] == "4"


def test_train_marks_only_the_warmup_steps_in_the_step_rows(metrics_config, tmp_path):
    train(metrics_config)

    rows = read_rows(tmp_path / "run-under-test" / "steps.csv")

    assert [row["warmup"] for row in rows].count("True") == 1


def test_train_with_metrics_disabled_writes_nothing(metrics_config, tmp_path):
    metrics_config.metrics.enabled = False

    train(metrics_config)

    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def poisoned_shard(tmp_path):
    path = tmp_path / "out_of_vocab.bin"
    np.full(2048, 60_000, dtype=np.uint16).tofile(path)

    return path


def test_train_that_crashes_mid_run_still_records_the_run(metrics_config, poisoned_shard, tmp_path):
    metrics_config.data.train = str(poisoned_shard)

    with pytest.raises(IndexError):
        train(metrics_config)

    assert read_rows(tmp_path / "runs.csv")[0]["status"] == "crashed"


def test_train_that_completes_records_a_completed_run(metrics_config, tmp_path):
    train(metrics_config)

    assert read_rows(tmp_path / "runs.csv")[0]["status"] == "completed"

import math

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from multi_gpu_llm_lab.data import SyntheticDataset
from multi_gpu_llm_lab.model import GPT
from multi_gpu_llm_lab.train import Trainer, TrainerConfig, evaluate, get_trainer


@pytest.fixture
def config():
    return TrainerConfig(model="tiny", optimizer="adamw", dataset="synthetic", val_dataset="synthetic", device="cpu")


def test_trainer_holds_the_parts_it_is_given():
    dataloader = DataLoader(SyntheticDataset(block_size=4, n_blocks=2, vocab_size=8))
    val_dataloader = DataLoader(SyntheticDataset(block_size=4, n_blocks=2, vocab_size=8, seed=1))

    trainer = Trainer(model="m", optimizer="o", dataloader=dataloader, val_dataloader=val_dataloader, device="cpu")

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
    config.optimizer = "adamw_fused"

    trainer = get_trainer(config)

    assert trainer.optimizer.param_groups[0]["fused"] is True


def test_get_trainer_defaults_to_an_unfused_optimizer(config):
    trainer = get_trainer(config)

    assert trainer.optimizer.param_groups[0]["fused"] is not True


def test_get_trainer_with_unknown_optimizer_raises_error(config):
    config.optimizer = "shampoo"

    with pytest.raises(KeyError):
        get_trainer(config)


def test_get_trainer_yields_batches_shaped_for_the_model(config):
    trainer = get_trainer(config)

    inputs, _ = next(iter(trainer.dataloader))

    assert inputs.shape == (config.micro_batch, trainer.model.config.block_size)


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

    assert trainer.val_dataloader.batch_size == config.micro_batch


def test_get_trainer_with_an_eval_batch_overrides_the_validation_batch_size(config):
    config.eval_batch = 2

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


def test_evaluate_runs_the_forward_pass_in_inference_mode(config):
    trainer = get_trainer(config)
    modes: list[bool] = []
    trainer.model.register_forward_hook(lambda *_: modes.append(torch.is_inference_mode_enabled()))

    evaluate(trainer, eval_iters=1)

    assert modes == [True]


def test_evaluate_leaves_the_model_in_training_mode(config):
    trainer = get_trainer(config)

    evaluate(trainer, eval_iters=1)

    assert trainer.model.training is True

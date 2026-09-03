import pytest

from multi_gpu_llm_lab.configs import (
    OptimConfig,
    TrainerConfig,
    build_model_config,
    build_trainer_config,
    config_paths,
    load_document,
    parse_overrides,
    resolve_trainer_config,
)


def test_build_model_config_with_350m_returns_gpt2_medium_depth():
    config = build_model_config("350m")

    assert config.n_layer == 24


def test_build_model_config_with_tiny_returns_a_locally_runnable_depth():
    config = build_model_config("tiny")

    assert config.n_layer == 2


def test_build_model_config_pads_the_vocabulary_for_gemm_alignment():
    config = build_model_config("350m")

    assert config.vocab_size % 64 == 0


def test_build_model_config_with_unknown_name_raises_error():
    with pytest.raises(KeyError):
        build_model_config("gpt5")


def test_build_trainer_config_with_an_empty_document_returns_the_in_code_defaults():
    config = build_trainer_config({})

    assert config == TrainerConfig()


def test_build_trainer_config_overrides_only_the_key_it_is_given():
    config = build_trainer_config({"optim": {"micro_batch": 8}})

    assert (config.optim.micro_batch, config.optim.learning_rate) == (8, OptimConfig().learning_rate)


def test_build_trainer_config_reads_every_section():
    document = {
        "model": "350m",
        "runtime": {"device": "cpu"},
        "data": {"train": "train.bin", "val": "val.bin", "workers": 4},
        "optim": {"name": "adamw_fused"},
        "eval": {"batch": 16},
    }

    config = build_trainer_config(document)

    assert (config.model, config.runtime.device, config.data.workers, config.optim.name, config.eval.batch) == (
        "350m",
        "cpu",
        4,
        "adamw_fused",
        16,
    )


def test_build_trainer_config_with_an_unknown_section_raises_error():
    with pytest.raises(KeyError):
        build_trainer_config({"optimiser": {"name": "adamw"}})


def test_build_trainer_config_with_an_unknown_key_inside_a_section_raises_error():
    with pytest.raises(KeyError):
        build_trainer_config({"optim": {"lr": 1e-3}})


def test_load_document_reads_the_nested_sections_of_a_yaml_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("model: 350m\noptim:\n  micro_batch: 8\n")

    document = load_document(path)

    assert document == {"model": "350m", "optim": {"micro_batch": 8}}


def test_load_document_with_an_empty_file_returns_an_empty_document(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    document = load_document(path)

    assert document == {}


def test_config_paths_lists_every_leaf_of_the_tree():
    paths = list(config_paths())

    assert paths == [
        "model",
        "runtime.device",
        "data.train",
        "data.val",
        "data.workers",
        "optim.name",
        "optim.learning_rate",
        "optim.micro_batch",
        "eval.interval",
        "eval.iters",
        "eval.batch",
    ]


def test_parse_overrides_folds_a_dotted_assignment_into_a_nested_document():
    document = parse_overrides(["optim.micro_batch=8"])

    assert document == {"optim": {"micro_batch": 8}}


def test_parse_overrides_reads_the_value_with_yaml_scalar_rules():
    document = parse_overrides(["optim.micro_batch=8", "eval.batch=null", "model=350m"])

    assert document == {"optim": {"micro_batch": 8}, "eval": {"batch": None}, "model": "350m"}


def test_parse_overrides_merges_two_assignments_in_the_same_section():
    document = parse_overrides(["optim.micro_batch=8", "optim.name=adamw_fused"])

    assert document == {"optim": {"micro_batch": 8, "name": "adamw_fused"}}


def test_parse_overrides_without_an_equals_sign_raises_error():
    with pytest.raises(ValueError, match="micro_batch"):
        parse_overrides(["optim.micro_batch"])


def test_resolve_trainer_config_without_anything_returns_the_in_code_defaults():
    config = resolve_trainer_config()

    assert config == TrainerConfig()


def test_resolve_trainer_config_reads_the_config_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("optim:\n  micro_batch: 8\n")

    config = resolve_trainer_config(path)

    assert config.optim.micro_batch == 8


def test_resolve_trainer_config_lets_an_override_win_over_the_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("optim:\n  micro_batch: 8\n")

    config = resolve_trainer_config(path, ["optim.micro_batch=2"])

    assert config.optim.micro_batch == 2


def test_resolve_trainer_config_keeps_the_sibling_keys_of_an_overridden_one(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("optim:\n  micro_batch: 8\n  name: adamw_fused\n")

    config = resolve_trainer_config(path, ["optim.micro_batch=2"])

    assert config.optim.name == "adamw_fused"


def test_resolve_trainer_config_with_an_unknown_override_key_raises_error():
    with pytest.raises(KeyError):
        resolve_trainer_config(None, ["optim.lr=1e-3"])


def test_build_trainer_config_coerces_a_value_to_the_type_of_its_field():
    config = build_trainer_config({"optim": {"learning_rate": "1e-3"}})

    assert config.optim.learning_rate == pytest.approx(1e-3)


def test_build_trainer_config_keeps_an_optional_field_none():
    config = build_trainer_config({"eval": {"batch": None}})

    assert config.eval.batch is None


def test_build_trainer_config_with_an_uncoercible_value_raises_error():
    with pytest.raises(ValueError, match="huit"):
        build_trainer_config({"optim": {"micro_batch": "huit"}})


def test_resolve_trainer_config_types_an_override_the_yaml_scalar_rules_leave_a_string():
    config = resolve_trainer_config(None, ["optim.learning_rate=1e-3"])

    assert config.optim.learning_rate == pytest.approx(1e-3)

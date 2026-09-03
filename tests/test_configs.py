import pytest

from multi_gpu_llm_lab.configs import build_model_config


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

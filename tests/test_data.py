import numpy as np
import pytest
import torch

from multi_gpu_llm_lab.data import SyntheticDataset, TokenDataset, build_dataset

BLOCK_SIZE = 8


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "shard.bin"
    np.arange(33, dtype=np.uint16).tofile(path)

    return path


def test_token_dataset_returns_an_input_block_of_the_requested_size(shard):
    dataset = TokenDataset(shard, block_size=BLOCK_SIZE)

    inputs, _ = dataset[0]

    assert inputs.shape == (BLOCK_SIZE,)


def test_token_dataset_target_is_the_input_shifted_by_one(shard):
    dataset = TokenDataset(shard, block_size=BLOCK_SIZE)

    inputs, targets = dataset[0]

    assert torch.equal(targets[:-1], inputs[1:])


def test_token_dataset_counts_whole_blocks_only(shard):
    dataset = TokenDataset(shard, block_size=BLOCK_SIZE)

    assert len(dataset) == 4


def test_token_dataset_yields_long_tensors_for_embedding_lookup(shard):
    dataset = TokenDataset(shard, block_size=BLOCK_SIZE)

    inputs, _ = dataset[0]

    assert inputs.dtype == torch.long


def test_synthetic_dataset_stays_inside_the_vocabulary():
    dataset = SyntheticDataset(block_size=BLOCK_SIZE, n_blocks=4, vocab_size=17)

    inputs, targets = dataset[0]

    assert int(torch.cat([inputs, targets]).max()) < 17


def test_build_dataset_with_synthetic_name_needs_no_file_on_disk():
    dataset = build_dataset("synthetic", block_size=BLOCK_SIZE, vocab_size=17)

    assert isinstance(dataset, SyntheticDataset)


def test_build_dataset_with_a_missing_path_raises_error():
    with pytest.raises(FileNotFoundError):
        build_dataset("/nope/missing.bin", block_size=BLOCK_SIZE, vocab_size=17)


def test_build_dataset_with_synthetic_name_and_a_different_seed_yields_different_blocks():
    default_seed = build_dataset("synthetic", block_size=BLOCK_SIZE, vocab_size=17)
    other_seed = build_dataset("synthetic", block_size=BLOCK_SIZE, vocab_size=17, seed=1)

    assert not torch.equal(default_seed[0][0], other_seed[0][0])

import pytest
import torch

from multi_gpu_llm_lab.configs import build_model_config
from multi_gpu_llm_lab.metrics.probes import MemoryPeaks, NullMemoryProbe
from multi_gpu_llm_lab.metrics.recorder import MetricsRecorder, model_flops_per_token, percentile
from multi_gpu_llm_lab.metrics.records import Hardware, Workload

TINY_PARAMS = 6_835_712


def clock_of(*times):
    values = iter(times)

    return lambda: next(values)


def build_recorder(clock, barrier=lambda: None, warmup_steps=0, memory=None):
    return MetricsRecorder(
        workload=Workload(micro_batch=4, block_size=128, world_size=1, flops_per_token=1_000_000),
        hardware=Hardware(peak_flops=1e12, world_size=1),
        warmup_steps=warmup_steps,
        memory=memory or NullMemoryProbe(),
        clock=clock,
        barrier=barrier,
    )


class SpyMemoryProbe:
    def __init__(self):
        self.resets_at = []
        self.records_seen = []

    def reset(self):
        self.resets_at.append(len(self.records_seen))

    def peaks(self):
        self.records_seen.append(1)
        return MemoryPeaks(allocated=1024, reserved=2048)


def test_model_flops_per_token_counts_six_flops_per_parameter_plus_attention():
    config = build_model_config("tiny")

    flops = model_flops_per_token(config, TINY_PARAMS)

    assert flops == 41_407_488


def test_percentile_at_p90_returns_a_step_time_that_actually_happened():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]

    result = percentile(values, 0.9)

    assert result == 100.0


def test_percentile_with_a_single_value_returns_that_value():
    result = percentile([7.0], 0.9)

    assert result == 7.0


def test_recorder_step_measures_the_elapsed_milliseconds_between_the_two_clock_reads():
    recorder = build_recorder(clock=clock_of(0.0, 0.010))

    with recorder.step(0):
        pass

    assert recorder.records[0].step_ms == 10.0


def test_recorder_step_drains_the_device_before_each_clock_read():
    calls = []

    recorder = build_recorder(
        clock=lambda: calls.append("clock") or 0.0,
        barrier=lambda: calls.append("barrier"),
    )
    with recorder.step(0):
        pass

    assert calls == ["barrier", "clock", "barrier", "clock"]


def test_recorder_flags_the_steps_below_the_warmup_threshold():
    recorder = build_recorder(clock=clock_of(0.0, 0.01, 0.01, 0.02, 0.02, 0.03), warmup_steps=2)

    for index in range(3):
        with recorder.step(index):
            pass

    assert [record.warmup for record in recorder.records] == [True, True, False]


def test_recorder_resets_the_memory_peaks_once_after_the_last_warmup_step():
    memory = SpyMemoryProbe()
    recorder = build_recorder(clock=clock_of(0.0, 0.01, 0.01, 0.02, 0.02, 0.03), warmup_steps=2, memory=memory)

    for index in range(3):
        with recorder.step(index):
            pass

    assert memory.resets_at == [2]


def test_recorder_records_the_peaks_the_probe_reports():
    recorder = build_recorder(clock=clock_of(0.0, 0.01), memory=SpyMemoryProbe())

    with recorder.step(0):
        pass

    assert (recorder.records[0].peak_allocated, recorder.records[0].peak_reserved) == (1024, 2048)


def test_recorder_without_cuda_records_no_peaks():
    recorder = build_recorder(clock=clock_of(0.0, 0.01))

    with recorder.step(0):
        pass

    assert (recorder.records[0].peak_allocated, recorder.records[0].peak_reserved) == (None, None)


def test_summarize_excludes_the_warmup_steps_from_the_median():
    recorder = build_recorder(clock=clock_of(0.0, 0.1, 0.1, 0.2, 0.2, 0.21, 0.21, 0.23, 0.23, 0.26), warmup_steps=2)

    for index in range(5):
        with recorder.step(index):
            pass

    assert recorder.summarize().step_ms_median == pytest.approx(20.0)


def test_summarize_divides_the_measured_tokens_by_the_measured_seconds():
    recorder = build_recorder(clock=clock_of(0.0, 0.1, 0.1, 0.2), warmup_steps=1)

    for index in range(2):
        with recorder.step(index):
            pass

    assert recorder.summarize().tokens_per_second == pytest.approx(5120.0)


def test_summarize_scores_the_mfu_against_the_peak_of_every_rank():
    recorder = build_recorder(clock=clock_of(0.0, 0.1, 0.1, 0.2), warmup_steps=1)

    for index in range(2):
        with recorder.step(index):
            pass

    assert recorder.summarize().mfu == pytest.approx(5120.0 * 1_000_000 / 1e12)


def test_summarize_of_a_run_shorter_than_the_warmup_leaves_the_aggregates_empty():
    recorder = build_recorder(clock=clock_of(0.0, 0.1), warmup_steps=5)

    with recorder.step(0):
        pass

    assert recorder.summarize().step_ms_median is None


def test_summarize_of_a_run_shorter_than_the_warmup_still_reports_the_steps_it_ran():
    recorder = build_recorder(clock=clock_of(0.0, 0.1), warmup_steps=5)

    with recorder.step(0):
        pass

    assert recorder.summarize().total_steps == 1


def test_summarize_carries_the_status_it_is_given():
    recorder = build_recorder(clock=clock_of(0.0, 0.1))

    with recorder.step(0):
        pass

    assert recorder.summarize("crashed").status == "crashed"


def test_summarize_reports_the_loss_of_the_last_measured_step():
    recorder = build_recorder(clock=clock_of(0.0, 0.1, 0.1, 0.2))

    for index in range(2):
        with recorder.step(index) as scope:
            scope.loss = torch.tensor(0.5 * index)

    assert recorder.summarize().final_loss == pytest.approx(0.5)

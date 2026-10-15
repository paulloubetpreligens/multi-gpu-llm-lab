from multi_gpu_llm_lab.metrics.records import Hardware, Workload


def test_hardware_peak_total_scales_the_single_gpu_peak_by_the_world_size():
    hardware = Hardware(peak_flops=312e12, world_size=8)

    assert hardware.peak_total == 2.496e15


def test_workload_tokens_per_step_counts_every_rank():
    workload = Workload(micro_batch=4, block_size=128, world_size=8, flops_per_token=1)

    assert workload.tokens_per_step == 4096

from multi_gpu_llm_lab.metrics.probes import MemoryPeaks, NullMemoryProbe, build_memory_probe


def test_build_memory_probe_off_cuda_cannot_report_peaks():
    assert isinstance(build_memory_probe("mps"), NullMemoryProbe)


def test_null_memory_probe_reports_named_absent_peaks():
    peaks = NullMemoryProbe().peaks()

    assert peaks == MemoryPeaks(allocated=None, reserved=None)

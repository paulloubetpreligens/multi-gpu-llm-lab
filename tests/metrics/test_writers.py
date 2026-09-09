import csv

from multi_gpu_llm_lab.metrics.records import RunSpec, RunSummary, StepRecord
from multi_gpu_llm_lab.metrics.writers import MetricsWriter

RUN_DIR = "20260906T120000Z-tiny"


def build_spec(run_id=RUN_DIR):
    return RunSpec(
        run_id=run_id,
        started_at="2026-09-06T12:00:00+00:00",
        git_revision="abc1234",
        torch_version="2.13.0",
        model="tiny",
        device="cpu",
        device_name="arm64",
        precision="fp32",
        compiled=False,
        micro_batch=4,
        block_size=128,
        world_size=1,
        n_params=6_835_712,
        flops_per_token=41_407_488,
        warmup_steps=2,
        peak_flops=312e12,
        peak_flops_label="A100 bf16 dense",
    )


def build_record(step=0, peak_allocated=None, peak_reserved=None):
    return StepRecord(
        step=step,
        warmup=False,
        step_ms=100.0,
        tokens=512,
        tokens_per_second=5120.0,
        mfu=0.00512,
        loss=0.5,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )


def build_summary(status="completed"):
    return RunSummary(
        status=status,
        total_steps=1,
        measured_steps=1,
        measured_seconds=0.1,
        wall_seconds=0.1,
        tokens=512,
        tokens_per_second=5120.0,
        mfu=0.00512,
        step_ms_median=100.0,
        step_ms_p90=100.0,
        peak_allocated=None,
        peak_reserved=None,
        final_loss=0.5,
    )


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(path):
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def test_metrics_writer_writes_one_step_row_per_recorded_step(tmp_path):
    writer = MetricsWriter(tmp_path, build_spec())

    for step in range(2):
        writer.write_step(build_record(step=step))

    assert len(read_rows(tmp_path / RUN_DIR / "steps.csv")) == 2


def test_metrics_writer_flushes_each_step_row_before_the_run_ends(tmp_path):
    writer = MetricsWriter(tmp_path, build_spec())

    writer.write_step(build_record())

    assert len(read_rows(tmp_path / RUN_DIR / "steps.csv")) == 1


def test_metrics_writer_leaves_an_empty_cell_where_the_backend_reports_no_peak(tmp_path):
    writer = MetricsWriter(tmp_path, build_spec())

    writer.write_step(build_record(peak_reserved=None))

    assert read_rows(tmp_path / RUN_DIR / "steps.csv")[0]["peak_reserved"] == ""


def test_metrics_writer_appends_a_second_run_without_repeating_the_run_header(tmp_path):
    for run_id in ("run-a", "run-b"):
        MetricsWriter(tmp_path, build_spec(run_id)).write_run(build_summary())

    assert len(read_rows(tmp_path / "runs.csv")) == 2


def test_metrics_writer_records_the_reference_peak_on_every_run_row(tmp_path):
    MetricsWriter(tmp_path, build_spec()).write_run(build_summary())

    assert read_rows(tmp_path / "runs.csv")[0]["peak_flops_label"] == "A100 bf16 dense"


def test_metrics_writer_creates_a_missing_output_directory(tmp_path):
    target = tmp_path / "deep" / "nested"

    MetricsWriter(target, build_spec())

    assert target.is_dir()


def test_metrics_writer_writes_the_step_header_in_a_fixed_column_order(tmp_path):
    MetricsWriter(tmp_path, build_spec())

    assert read_header(tmp_path / RUN_DIR / "steps.csv") == [
        "run_id",
        "step",
        "warmup",
        "step_ms",
        "tokens",
        "tokens_per_second",
        "mfu",
        "loss",
        "peak_allocated",
        "peak_reserved",
        "world_size",
        "flops_per_token",
        "peak_flops",
        "peak_flops_label",
    ]


def test_metrics_writer_writes_the_step_cells_under_the_column_they_belong_to(tmp_path):
    writer = MetricsWriter(tmp_path, build_spec())

    writer.write_step(build_record(peak_allocated=1024, peak_reserved=2048))

    assert read_rows(tmp_path / RUN_DIR / "steps.csv")[0] == {
        "run_id": RUN_DIR,
        "step": "0",
        "warmup": "False",
        "step_ms": "100.0",
        "tokens": "512",
        "tokens_per_second": "5120.0",
        "mfu": "0.00512",
        "loss": "0.5",
        "peak_allocated": "1024",
        "peak_reserved": "2048",
        "world_size": "1",
        "flops_per_token": "41407488",
        "peak_flops": "312000000000000.0",
        "peak_flops_label": "A100 bf16 dense",
    }


def test_metrics_writer_writes_the_run_header_in_a_fixed_column_order(tmp_path):
    MetricsWriter(tmp_path, build_spec()).write_run(build_summary())

    assert read_header(tmp_path / "runs.csv") == [
        "run_id",
        "started_at",
        "status",
        "git_revision",
        "torch_version",
        "model",
        "device",
        "device_name",
        "precision",
        "compiled",
        "micro_batch",
        "block_size",
        "world_size",
        "n_params",
        "flops_per_token",
        "peak_flops",
        "peak_flops_label",
        "warmup_steps",
        "total_steps",
        "measured_steps",
        "measured_seconds",
        "wall_seconds",
        "tokens",
        "tokens_per_second",
        "mfu",
        "step_ms_median",
        "step_ms_p90",
        "peak_allocated",
        "peak_reserved",
        "final_loss",
    ]


def test_metrics_writer_writes_the_run_cells_under_the_column_they_belong_to(tmp_path):
    MetricsWriter(tmp_path, build_spec()).write_run(build_summary())

    assert read_rows(tmp_path / "runs.csv")[0] == {
        "run_id": RUN_DIR,
        "started_at": "2026-09-06T12:00:00+00:00",
        "status": "completed",
        "git_revision": "abc1234",
        "torch_version": "2.13.0",
        "model": "tiny",
        "device": "cpu",
        "device_name": "arm64",
        "precision": "fp32",
        "compiled": "False",
        "micro_batch": "4",
        "block_size": "128",
        "world_size": "1",
        "n_params": "6835712",
        "flops_per_token": "41407488",
        "peak_flops": "312000000000000.0",
        "peak_flops_label": "A100 bf16 dense",
        "warmup_steps": "2",
        "total_steps": "1",
        "measured_steps": "1",
        "measured_seconds": "0.1",
        "wall_seconds": "0.1",
        "tokens": "512",
        "tokens_per_second": "5120.0",
        "mfu": "0.00512",
        "step_ms_median": "100.0",
        "step_ms_p90": "100.0",
        "peak_allocated": "",
        "peak_reserved": "",
        "final_loss": "0.5",
    }

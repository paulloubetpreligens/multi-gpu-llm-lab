from typer.testing import CliRunner

from multi_gpu_llm_lab.entrypoints.app import app

runner = CliRunner()


def test_app_exposes_a_train_command():
    result = runner.invoke(app, ["--help"])

    assert "train" in result.stdout


def test_train_help_documents_the_config_option():
    result = runner.invoke(app, ["train", "--help"])

    assert "--config" in result.stdout


def test_train_help_lists_the_overridable_config_keys():
    result = runner.invoke(app, ["train", "--help"])

    assert "optim.micro_batch" in result.stdout.replace("\n", "")


def test_train_with_an_unknown_option_exits_with_an_error():
    result = runner.invoke(app, ["train", "--nope", "1"])

    assert result.exit_code != 0


def test_train_with_an_unknown_override_key_exits_with_an_error():
    result = runner.invoke(app, ["train", "--set", "optim.lr=1e-3"])

    assert result.exit_code != 0

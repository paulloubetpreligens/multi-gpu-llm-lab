"""Command-line entrypoints, one module per command."""

from multi_gpu_llm_lab.entrypoints import prepare, train  # noqa: F401  imported for their @app.command registration

setup: # create the virtualenv and install every dependency
	uv sync

format: # apply formatting and import sorting
	uv run ruff format .
	uv run ruff check --fix .

check: # run quality checks
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy --exclude 'multi_gpu_llm_lab/model\.py$$' multi_gpu_llm_lab/

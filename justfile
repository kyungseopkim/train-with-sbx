# default target
default:
	@just --list

# run code formatting and linting
format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# run the test suite
test:
	uv run pytest

# run the test suite with coverage
test-cov:
	uv run pytest --cov=src --cov-report=term-missing

# sync dependencies
sync:
	uv sync

# run optuna hyperparameter tuning for a given ticker and date
tune ticker date n_trials="50":
	uv run rl-model tune {{ticker}} {{date}} --n-trials {{n_trials}}

# train a model on a single day
train-day ticker date:
	uv run rl-model train-day {{ticker}} {{date}}

# run walk-forward training over a date range
train-walk ticker start_date end_date:
	uv run rl-model train-walk {{ticker}} {{start_date}} {{end_date}}

# resume walk-forward training from the latest checkpoint
resume-walk ticker start_date end_date:
	uv run rl-model train-walk {{ticker}} {{start_date}} {{end_date}} --resume

# evaluate a trained model over a date range
evaluate ticker start_date end_date model_path:
	uv run rl-model evaluate {{ticker}} {{start_date}} {{end_date}} --model {{model_path}}

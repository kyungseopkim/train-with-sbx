# rl-model

Reinforcement learning stock trading agent using PPO (Proximal Policy Optimization) via [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). The agent learns intraday trading decisions on a single stock per episode, using technical indicators as observations.

## Requirements

- Python >= 3.14
- MySQL database with daily and intraday bar tables
- NVIDIA GPU (CUDA 13.0) for training
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Install dependencies
uv sync

# Configure database connection
cp .env.example .env  # then edit with your DB credentials
```

The `.env` file should contain:

```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

The database must have two tables per ticker:

- `{ticker}_daily` — daily OHLCV bars with columns: `date`, `open`, `high`, `low`, `close`, `volume`, `vwap`
- `{ticker}_historical` — intraday minute bars with columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`

## Usage

### Single-day training

Train a PPO agent on one day of intraday data:

```bash
uv run python examples/one_day_training.py AAPL 2026-01-15
uv run python examples/one_day_training.py AAPL 2026-01-15 --timesteps 2000000
```

### Walk-forward training

Train across a date range with daily walk-forward evaluation. Uses an actor-based pipeline with parallel data prefetching:

```bash
uv run python examples/walk_forward_training.py AAPL 2025-01-01 2025-12-31
uv run python examples/walk_forward_training.py AAPL 2025-01-01 2025-12-31 \
    --timesteps 200000 --n-envs 8 --save-every 50
```

To resume training from a previously saved model on a new date range, use `--model`:

```bash
uv run python examples/walk_forward_training.py AAPL 2026-01-01 2026-06-30 \
    --model results/model_aapl_2025-01-01_2025-12-31.zip
```

Without `--model`, the script auto-detects a model matching the exact ticker/start/end dates. Use `--from-scratch` to skip all resume logic and train a fresh model.

Results are saved to `results/` as CSV files and model checkpoints. Training can be interrupted with Ctrl+C and the current model will be saved.

## Architecture

```
db.py → indicators.py → env.py
                           ↑
                       pipeline.py (walk-forward orchestration)
```

### `src/rl_model/db.py`

MySQL queries via pymysql. `get_date_window(ticker, date)` is the main entry point: fetches 90 prior daily bars from `{ticker}_daily` and intraday minute bars from `{ticker}_historical`, then concatenates them into a unified Polars DataFrame.

`DailyBarCache` bulk-fetches all daily bars for a ticker in one query, eliminating redundant DB round-trips during walk-forward training (250 days share ~99% of daily bar data).

### `src/rl_model/indicators.py`

Pure Polars transformations. `add_all_indicators()` chains SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, and Stochastic oscillator onto the DataFrame. `normalize()` z-score normalizes all numeric columns.

### `src/rl_model/env.py`

Gymnasium environment (`StockTradingEnv`). Discrete action space of 5 actions mapping to target portfolio allocations (0%, 25%, 50%, 75%, 100% in stock). Observation is 23 normalized feature columns. Reward is step-over-step portfolio value change. One episode = one trading day of minute bars.

`StockTradingEnv.from_arrays()` creates an environment from pre-computed numpy arrays, skipping DB and indicator work (used by the pipeline for vectorized training).

### `src/rl_model/pipeline.py`

Actor-based concurrent training pipeline for walk-forward evaluation:

- **DataPool** — N worker threads prefetch and process data in parallel via a bounded queue. Uses `DailyBarCache` to avoid redundant daily bar queries.
- **TrainingPipeline** — Main-thread GPU training. Builds `DummyVecEnv` from prefetched data, trains PPO, evaluates on the next day.
- **LoggerActor** — Asynchronous result collection.

## Testing

```bash
# Run all tests (no DB connection needed — tests use mocks)
uv run pytest

# Run a specific test file
uv run pytest tests/test_env.py

# Run a specific test
uv run pytest tests/test_env.py::test_buy_updates_portfolio
```

## Key Design Decisions

- Data is loaded once per environment construction, not per step — one episode per env instance
- Tests mock DB functions so no database connection is needed
- Polars is used throughout instead of pandas
- Walk-forward pipeline uses `DailyBarCache` to reduce ~250 daily-bar queries to 1
- Unified CPU-GPU memory (NVIDIA GB10) means numpy arrays from worker threads need no explicit transfer to GPU

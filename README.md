# RL-Model (train-with-sbx)

A high-performance Reinforcement Learning stock trading framework using JAX-accelerated algorithms via [SBX (Stable Baselines JAX)](https://github.com/araffin/sbx). This project implements intraday trading agents that learn optimal portfolio allocation on a single stock per episode using technical indicators and JAX-based environments.

## Features

- **JAX Acceleration**: Uses `sbx-rl` for lightning-fast training on NVIDIA GPUs (CUDA 13.0).
- **Polars Integration**: Efficient data processing and technical indicator calculation using [Polars](https://pola.rs/).
- **Vectorized Environments**: Support for high-throughput training via `jax_env.py` and `vec_env.py`.
- **MySQL Backend**: Integrated with MySQL for historical daily and intraday bar data.
- **Walk-forward Pipeline**: Robust training and evaluation pipeline with concurrent data prefetching.
- **Modern Python Stack**: Built with Python 3.14+, `uv` for dependency management, and `Gymnasium` for environment standards.

## Requirements

- **Python**: >= 3.14
- **Hardware**: NVIDIA GPU (optimized for CUDA 13.0/GB10)
- **Database**: MySQL server with `daily` and `historical` bar tables.
- **Tools**: [uv](https://docs.astral.sh/uv/) package manager.

## Setup

```bash
# Clone the repository
git clone git@github.com:kyungseopkim/train-with-sbx.git
cd train-with-sbx

# Install dependencies using uv
uv sync

# Configure environment variables
cp .env.example .env  # Update with your MySQL credentials
```

### Database Schema

Ensure your MySQL database has the following tables for each ticker (e.g., `AAPL`):

- **`{ticker}_daily`**: Columns: `date`, `open`, `high`, `low`, `close`, `volume`, `vwap`.
- **`{ticker}_historical`**: Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`.

## Usage

The project uses `rich-click` for a polished CLI experience. You can run the main entry point via:

```bash
uv run rl-model --help
```

### Training

To start the training pipeline:

```bash
uv run rl-model train AAPL --start-date 2025-01-01 --end-date 2025-12-31
```

### Running Tests

```bash
# Run all tests (utilizes mocks, no DB required)
uv run pytest

# Run with coverage
uv run pytest --cov=src/rl_model
```

## Project Structure

- `src/rl_model/`
  - `main.py`: CLI entry point.
  - `pipeline.py`: Orchestrates the training/evaluation workflow.
  - `jax_env.py`: JAX-compatible environment implementation.
  - `env.py`: Standard Gymnasium environment.
  - `db.py`: Database interface and caching logic.
  - `indicators.py`: Technical indicator calculations using Polars.
  - `vec_env.py`: Vectorized environment wrappers.
- `tests/`: Comprehensive test suite for all modules.

## Architecture

```text
Database (MySQL) ──> Polars (Indicators) ──> Gymnasium/JAX Env ──> SBX (PPO/DQN/etc)
      ^                    ^                       ^                      |
      └────────────────────┴────────── Pipeline ───┴──────────────────────┘
```

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'feat: add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

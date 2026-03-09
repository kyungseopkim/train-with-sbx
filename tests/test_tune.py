import json
import os
from datetime import date
from unittest.mock import patch

import numpy as np
import optuna
from click.testing import CliRunner

from rl_model.tune import main, objective, suggest_params


def test_suggest_params_returns_expected_keys():
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(trial)

    expected_keys = {
        "learning_rate", "ent_coef", "gamma", "clip_range",
        "n_epochs", "gae_lambda", "vf_coef", "net_arch", "batch_size",
    }
    assert set(params.keys()) == expected_keys

    assert 1e-5 <= params["learning_rate"] <= 1e-3
    assert 0.0 <= params["ent_coef"] <= 0.05
    assert 0.95 <= params["gamma"] <= 0.999
    assert 0.1 <= params["clip_range"] <= 0.3
    assert 3 <= params["n_epochs"] <= 15
    assert 0.9 <= params["gae_lambda"] <= 0.99
    assert 0.25 <= params["vf_coef"] <= 1.0
    assert params["net_arch"] in ["64,64", "128,128", "256,256", "512,512"]
    assert params["batch_size"] in [256, 512, 1024, 2048]


def test_objective_runs_with_synthetic_data():
    rng = np.random.RandomState(42)
    n_rows = 100
    n_features = 23
    train_data = rng.randn(n_rows, n_features).astype(np.float32)
    train_prices = np.linspace(100.0, 110.0, n_rows).astype(np.float64)
    eval_data = rng.randn(n_rows, n_features).astype(np.float32)
    eval_prices = np.linspace(110.0, 115.0, n_rows).astype(np.float64)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            train_data=train_data,
            train_prices=train_prices,
            eval_data=eval_data,
            eval_prices=eval_prices,
            timesteps=256,
            n_envs=2,
            n_steps=128,
            initial_cash=100_000.0,
            device="cpu",
        ),
        n_trials=1,
    )

    assert len(study.trials) == 1
    assert isinstance(study.best_value, float)


@patch("rl_model.tune.get_next_trading_date")
@patch("rl_model.tune.prepare_env_data")
def test_cli_invokes_study(mock_prepare, mock_next_date, tmp_path):
    rng = np.random.RandomState(99)
    n_rows = 100
    n_features = 23
    train_data = rng.randn(n_rows, n_features).astype(np.float32)
    train_prices = np.linspace(100.0, 110.0, n_rows).astype(np.float64)
    eval_data = rng.randn(n_rows, n_features).astype(np.float32)
    eval_prices = np.linspace(110.0, 115.0, n_rows).astype(np.float64)

    mock_next_date.return_value = date(2026, 2, 14)
    mock_prepare.side_effect = [
        (train_data, train_prices),
        (eval_data, eval_prices),
    ]

    output_dir = str(tmp_path / "results")
    runner = CliRunner()
    result = runner.invoke(main, [
        "AAPL", "2026-02-13",
        "--n-trials", "1",
        "--timesteps", "256",
        "--n-envs", "2",
        "--n-steps", "128",
        "--initial-cash", "100000",
        "--output-dir", output_dir,
        "--device", "cpu",
    ])

    assert result.exit_code == 0, result.output
    params_path = os.path.join(output_dir, "best_params.json")
    assert os.path.exists(params_path)
    with open(params_path) as f:
        params = json.load(f)
    assert "learning_rate" in params
    assert "best_return_pct" in params

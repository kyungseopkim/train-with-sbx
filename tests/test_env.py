import random
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from rl_model.env import FEATURE_COLS, StockTradingEnv, prepare_env_data


def make_date_window(n_daily=90, n_intraday=390, base_price=100.0):
    """Synthetic data matching get_date_window output schema."""
    rng = random.Random(42)
    rows = []
    price = base_price

    # Daily rows: dates before the target date
    target = datetime(2026, 2, 13)
    for i in range(n_daily, 0, -1):
        ts = target - timedelta(days=i)
        change = rng.uniform(-2, 2)
        o = price + rng.uniform(-1, 1)
        c = price + change
        h = max(o, c) + rng.uniform(0, 1)
        l = min(o, c) - rng.uniform(0, 1)
        v = rng.randint(500_000, 3_000_000)
        vw = (h + l + c) / 3
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l,
                      "close": c, "volume": v, "vwap": vw})
        price = c

    # Intraday rows: 1-min bars on target date
    start = target.replace(hour=9, minute=30)
    for i in range(n_intraday):
        ts = start + timedelta(minutes=i)
        change = rng.uniform(-0.5, 0.5)
        o = price + rng.uniform(-0.2, 0.2)
        c = price + change
        h = max(o, c) + rng.uniform(0, 0.3)
        l = min(o, c) - rng.uniform(0, 0.3)
        v = rng.randint(10_000, 100_000)
        vw = (h + l + c) / 3
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l,
                      "close": c, "volume": v, "vwap": vw})
        price = c

    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime)
    )


@pytest.fixture
def env():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window()
        e = StockTradingEnv("aapl", "2026-02-13")
        e.reset()
        yield e


def test_env_spaces(env):
    assert env.action_space.n == 5
    assert env.observation_space.shape == (len(FEATURE_COLS),)
    assert env.observation_space.dtype == np.float32


def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert obs.shape == (len(FEATURE_COLS),)
    assert obs.dtype == np.float32
    assert info["step"] == 0
    assert info["cash"] == 100_000.0
    assert info["shares"] == 0.0


def test_reset_restores_state(env):
    env.step(4)  # buy
    env.step(4)
    obs, info = env.reset()
    assert info["cash"] == 100_000.0
    assert info["shares"] == 0.0
    assert info["step"] == 0


def test_step_returns_tuple(env):
    result = env.step(2)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (len(FEATURE_COLS),)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False


def test_no_trade_when_already_at_target(env):
    """Action 0 (0% stock) when already at 0% = no trade, no cost."""
    _, info_before = env.reset()
    _, _, _, _, info_after = env.step(0)
    # No shares traded, no cost deducted, only price movement affects value
    assert info_after["shares"] == 0.0


def test_buy_updates_portfolio(env):
    env.reset()
    _, _, _, _, info = env.step(4)  # strong buy = 100%
    assert info["shares"] > 0
    assert info["cash"] < 0.01  # almost all in stock, minus cost


def test_sell_after_buy(env):
    env.reset()
    env.step(4)  # buy 100%
    _, _, _, _, info = env.step(0)  # sell to 0%
    assert abs(info["shares"]) < 1e-10
    assert info["cash"] > 0


def test_reward_is_pnl_change(env):
    env.reset()
    _, reward1, _, _, info1 = env.step(4)  # buy
    pv_after_buy = info1["portfolio_value"]
    _, reward2, _, _, info2 = env.step(4)  # hold position
    expected_reward = info2["portfolio_value"] - pv_after_buy
    assert abs(reward2 - expected_reward) < 1e-6


def test_episode_terminates():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window(n_intraday=5)
        env = StockTradingEnv("aapl", "2026-02-13")
        env.reset()
        for i in range(4):
            _, _, terminated, _, _ = env.step(2)
            assert not terminated
        _, _, terminated, _, _ = env.step(2)
        assert terminated


def test_transaction_cost_deducted():
    cost_pct = 0.01  # 1% for easy math
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window()
        env = StockTradingEnv("aapl", "2026-02-13", transaction_cost_pct=cost_pct)
        env.reset()
        _, _, _, _, info = env.step(4)  # buy 100%
        # Cost = trade_value * 1% ≈ 100_000 * 1% = 1000
        assert info["portfolio_value"] < 100_000.0
        assert info["cash"] < 0  # cost makes cash negative when 100% in stock


def test_observations_no_nans(env):
    obs, _ = env.reset()
    assert not np.any(np.isnan(obs))
    for _ in range(10):
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not np.any(np.isnan(obs))
        if terminated:
            break


def test_env_intraday_only():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window(n_daily=0)
        env = StockTradingEnv("aapl", "2026-02-13", include_daily=False)
        obs, info = env.reset()
        assert obs.shape == (len(FEATURE_COLS),)
        mock.assert_called_once_with("aapl", "2026-02-13", include_daily=False, daily_cache=None, intraday_cache=None)


def test_no_intraday_data_raises():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window(n_intraday=0)
        with pytest.raises(ValueError, match="No intraday data"):
            StockTradingEnv("aapl", "2026-02-13")


def test_from_arrays():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window()
        norm_data, raw_prices = prepare_env_data("aapl", "2026-02-13")

    env = StockTradingEnv.from_arrays(norm_data, raw_prices)
    obs, info = env.reset()
    assert obs.shape == (len(FEATURE_COLS),)
    assert info["cash"] == 100_000.0

    obs, reward, terminated, truncated, info = env.step(4)
    assert info["shares"] > 0


def test_from_arrays_matches_standard_init():
    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window()
        norm_data, raw_prices = prepare_env_data("aapl", "2026-02-13")
        env_arrays = StockTradingEnv.from_arrays(norm_data, raw_prices)

    with patch("rl_model.env.get_date_window") as mock:
        mock.return_value = make_date_window()
        env_standard = StockTradingEnv("aapl", "2026-02-13")

    obs_a, _ = env_arrays.reset()
    obs_s, _ = env_standard.reset()
    np.testing.assert_array_equal(obs_a, obs_s)

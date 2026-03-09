import time
from queue import Queue
from unittest.mock import patch

import numpy as np
import pytest

from rl_model.pipeline import (
    Actor,
    DataPool,
    DayData,
    EvalResult,
    LoggerActor,
    TrainingPipeline,
    _SENTINEL,
)


# ── Actor base class ────────────────────────────────────────────────


class EchoActor(Actor):
    """Test actor that echoes messages to a queue."""

    def __init__(self, out: Queue, **kwargs):
        super().__init__("echo", **kwargs)
        self.out = out

    def on_receive(self, message):
        self.out.put(message)


def test_actor_start_stop():
    out = Queue()
    actor = EchoActor(out)
    actor.start()
    actor.tell("hello")
    assert out.get(timeout=2) == "hello"
    actor.stop()


def test_actor_processes_multiple_messages():
    out = Queue()
    actor = EchoActor(out)
    actor.start()
    for i in range(10):
        actor.tell(i)
    results = [out.get(timeout=2) for _ in range(10)]
    assert results == list(range(10))
    actor.stop()


def test_actor_handles_exception_without_crashing():
    """Actor should log exceptions but keep processing."""

    class FailOnceActor(Actor):
        def __init__(self, out):
            super().__init__("fail-once")
            self.out = out
            self.call_count = 0

        def on_receive(self, message):
            self.call_count += 1
            if self.call_count == 1:
                raise ValueError("boom")
            self.out.put(message)

    out = Queue()
    actor = FailOnceActor(out)
    actor.start()
    actor.tell("first")   # will raise
    actor.tell("second")  # should still be delivered
    assert out.get(timeout=2) == "second"
    actor.stop()


# ── LoggerActor ──────────────────────────────────────────────────────


def test_logger_actor_collects_results():
    collected = []
    logger = LoggerActor(callback=lambda r: collected.append(r))
    logger.start()

    result = EvalResult("2026-01-01", "2026-01-02", 390, 500.0, 100_500.0, 0.5)
    logger.tell(result)

    # Give the thread a moment to process
    time.sleep(0.05)
    logger.stop()

    assert len(logger.results) == 1
    assert logger.results[0].return_pct == 0.5
    assert len(collected) == 1


# ── DayData / EvalResult ─────────────────────────────────────────────


def test_day_data_slots():
    d = DayData("2026-01-01", np.zeros((5, 3)), np.zeros(5))
    assert d.date == "2026-01-01"
    assert d.error is None


def test_day_data_with_error():
    d = DayData("2026-01-01", None, None, error="connection refused")
    assert d.error == "connection refused"


def test_eval_result_fields():
    r = EvalResult("2026-01-01", "2026-01-02", 390, 1234.56, 101_234.56, 1.23, 45.2)
    assert r.elapsed_s == 45.2


# ── DataPool ─────────────────────────────────────────────────────────


def _fake_prepare(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
    """Deterministic fake data for testing DataPool."""
    n = 10
    norm = np.ones((n, 3), dtype=np.float32) * hash(date) % 100
    prices = np.linspace(100, 110, n, dtype=np.float64)
    return norm, prices


@patch("rl_model.pipeline.IntradayCache")
@patch("rl_model.pipeline.DailyBarCache")
@patch("rl_model.pipeline.prepare_env_data", side_effect=_fake_prepare)
def test_data_pool_delivers_in_order(mock_prep, mock_cache_cls, mock_intraday_cls):
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    pool = DataPool("TEST", dates, n_workers=2, max_buffered=4).start()

    results = list(pool)
    assert len(results) == 4
    assert [r.date for r in results] == dates
    for r in results:
        assert r.normalized_data is not None
        assert r.error is None


@patch("rl_model.pipeline.IntradayCache")
@patch("rl_model.pipeline.DailyBarCache")
@patch("rl_model.pipeline.prepare_env_data", side_effect=_fake_prepare)
def test_data_pool_many_dates(mock_prep, mock_cache_cls, mock_intraday_cls):
    dates = [f"2026-01-{d:02d}" for d in range(1, 21)]
    pool = DataPool("TEST", dates, n_workers=4, max_buffered=4).start()

    results = list(pool)
    assert len(results) == 20
    assert [r.date for r in results] == dates


@patch("rl_model.pipeline.IntradayCache")
@patch("rl_model.pipeline.DailyBarCache")
@patch("rl_model.pipeline.prepare_env_data")
def test_data_pool_handles_errors(mock_prep, mock_cache_cls, mock_intraday_cls):
    def _sometimes_fail(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
        if date == "2026-01-02":
            raise ConnectionError("db down")
        return _fake_prepare(ticker, date)

    mock_prep.side_effect = _sometimes_fail
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    pool = DataPool("TEST", dates, n_workers=2, max_buffered=4).start()

    results = list(pool)
    assert len(results) == 3
    assert results[0].error is None
    assert results[1].error is not None
    assert "db down" in results[1].error
    assert results[2].error is None


@patch("rl_model.pipeline.IntradayCache")
@patch("rl_model.pipeline.DailyBarCache")
@patch("rl_model.pipeline.prepare_env_data", side_effect=_fake_prepare)
def test_data_pool_backpressure(mock_prep, mock_cache_cls, mock_intraday_cls):
    """With max_buffered=2, pool should still work (just slower)."""
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    pool = DataPool("TEST", dates, n_workers=2, max_buffered=2).start()

    results = list(pool)
    assert len(results) == 5


# ── TrainingPipeline ppo_kwargs ──────────────────────────────────────


@patch("rl_model.pipeline.IntradayCache")
@patch("rl_model.pipeline.DailyBarCache")
@patch("rl_model.pipeline.prepare_env_data", side_effect=_fake_prepare)
def test_pipeline_forwards_ppo_kwargs(mock_prep, mock_cache_cls, mock_intraday_cls):
    """TrainingPipeline should pass ppo_kwargs through to the PPO constructor."""
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    ppo_kwargs = {"learning_rate": 1e-4, "ent_coef": 0.05, "gamma": 0.99}

    pipeline = TrainingPipeline(
        ticker="TEST",
        dates=dates,
        timesteps_per_day=128,
        n_envs=2,
        n_steps=64,
        batch_size=64,
        device="cpu",
        ppo_kwargs=ppo_kwargs,
    )

    with patch("rl_model.pipeline.VectorizedStockTradingEnv") as mock_vec_env, \
         patch("stable_baselines3.PPO") as mock_ppo_cls:
        # Set up mock env
        mock_env_instance = mock_vec_env.return_value
        mock_env_instance.observation_space = None
        mock_env_instance.action_space = None

        # Set up mock model
        mock_model = mock_ppo_cls.return_value
        mock_model.num_timesteps = 0
        mock_model.learn.side_effect = lambda **kw: setattr(mock_model, "num_timesteps", 128)
        mock_model.predict.return_value = (0, None)

        # Mock eval env
        with patch("rl_model.pipeline.StockTradingEnv") as mock_eval_env_cls:
            mock_eval_env = mock_eval_env_cls.from_arrays.return_value
            mock_eval_env.initial_cash = 100_000.0
            mock_eval_env.reset.return_value = (np.zeros(23), {})
            mock_eval_env.step.return_value = (
                np.zeros(23), 1.0, True, False,
                {"portfolio_value": 100_100.0},
            )

            pipeline.run()

        # Verify PPO was constructed with our extra kwargs
        mock_ppo_cls.assert_called_once()
        call_kwargs = mock_ppo_cls.call_args
        assert call_kwargs.kwargs["learning_rate"] == 1e-4
        assert call_kwargs.kwargs["ent_coef"] == 0.05
        assert call_kwargs.kwargs["gamma"] == 0.99

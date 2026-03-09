"""Actor-based concurrent training pipeline.

Designed for ARM big.LITTLE (20 cores) + NVIDIA GB10 (unified memory).

Architecture
============

    ┌──────────────────────────────────────────────────┐
    │                  TrainingPipeline                 │
    │          (main thread — GPU access)               │
    └──────┬──────────────────────────────┬────────────┘
           │ pulls DayData               │ pushes EvalResult
    ┌──────▼──────┐               ┌──────▼──────┐
    │  DataPool    │               │   Logger    │
    │  (N threads) │               │  (1 thread) │
    └─────────────┘               └─────────────┘

DataPool: N worker threads fetch DB data in parallel, compute indicators,
          and deliver DayData in chronological order via a bounded queue.
          Efficiency cores (A725) handle I/O; bounded queue = backpressure.

Trainer:  Runs in the calling thread with GPU access.  Builds a vectorized
          numpy env from DayData arrays, trains PPO, evaluates on the next day.
          Unified memory means arrays from DataPool threads need no copy.

Logger:   Receives EvalResult messages and processes them asynchronously
          (CSV writes, progress logging).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Callable

import numpy as np

from rl_model.db import DailyBarCache, IntradayCache
from rl_model.env import StockTradingEnv, prepare_env_data, prepare_all_data
from rl_model.vec_env import FastVectorizedStockTradingEnv

log = logging.getLogger(__name__)

# ── Sentinel ─────────────────────────────────────────────────────────

_SENTINEL = object()


# ── Messages ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class DayData:
    """Pre-computed environment data for one trading day."""
    date: str
    normalized_data: np.ndarray | None
    raw_close_prices: np.ndarray | None
    error: str | None = None


@dataclass(slots=True)
class EvalResult:
    """Evaluation metrics for one walk-forward day."""
    train_date: str
    eval_date: str
    steps: int
    total_reward: float
    final_value: float
    return_pct: float
    elapsed_s: float = 0.0
    train_s: float = 0.0
    eval_s: float = 0.0
    train_timesteps: int = 0


# ── Actor base ───────────────────────────────────────────────────────

class Actor:
    """Lightweight thread-based actor with a bounded message inbox.

    Subclass and override ``on_receive`` to handle messages.
    Call ``tell(msg)`` to send a message (blocks if inbox is full).
    Call ``stop()`` to shut down gracefully.
    """

    def __init__(self, name: str, inbox_size: int = 0):
        self.name = name
        self._inbox: Queue = Queue(maxsize=inbox_size)
        self._thread: Thread | None = None

    def start(self) -> Actor:
        self._thread = Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        return self

    def tell(self, message):
        """Send a message to this actor (blocks if inbox full)."""
        self._inbox.put(message)

    def stop(self, timeout: float = 10.0):
        """Signal shutdown and wait for the actor to finish."""
        self._inbox.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self):
        self.on_start()
        while True:
            msg = self._inbox.get()
            if msg is _SENTINEL:
                break
            try:
                self.on_receive(msg)
            except Exception:
                log.exception("%s: error processing %s", self.name, type(msg).__name__)
        self.on_stop()

    # Override in subclasses ──────────────────────────────────────────

    def on_start(self):
        """Called once before the message loop begins."""

    def on_stop(self):
        """Called once after the message loop ends."""

    def on_receive(self, message):
        """Handle a single message. Must be overridden."""
        raise NotImplementedError


# ── Logger actor ─────────────────────────────────────────────────────

class LoggerActor(Actor):
    """Collects EvalResults and forwards them to a callback asynchronously."""

    def __init__(
        self,
        callback: Callable[[EvalResult], None] | None = None,
        buffer: int = 64,
    ):
        super().__init__("logger", inbox_size=buffer)
        self.callback = callback
        self.results: list[EvalResult] = []

    def on_receive(self, message):
        if isinstance(message, EvalResult):
            self.results.append(message)
            if self.callback:
                self.callback(message)


# ── DataPool ─────────────────────────────────────────────────────────

def _default_data_workers() -> int:
    """Pick a default worker count based on available cores.

    On big.LITTLE (e.g. 20 = 10 perf + 10 eff) we use roughly half the
    cores so efficiency cores handle DB I/O while leaving headroom for
    the training thread and Polars internal parallelism.
    """
    cpus = os.cpu_count() or 4
    return max(2, min(cpus // 3, 8))


class DataPool:
    """Parallel data prefetcher with ordered, bounded delivery.

    Submits all dates to a ThreadPoolExecutor immediately, then delivers
    DayData objects in chronological order through a bounded queue.
    Backpressure is automatic: if the consumer is slow, workers block
    when the queue is full.

    Usage::

        pool = DataPool("AAPL", dates).start()
        for day in pool:
            ...  # DayData in date order
    """

    def __init__(
        self,
        ticker: str,
        dates: list[str],
        n_workers: int | None = None,
        max_buffered: int = 16,
    ):
        self.ticker = ticker
        self.dates = dates
        self.n_workers = n_workers or _default_data_workers()
        self._queue: Queue = Queue(maxsize=max_buffered)
        self._thread: Thread | None = None

    def start(self) -> DataPool:
        self._thread = Thread(target=self._run, name="data-pool", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        log.info(
            "DataPool: prefetching %d dates with %d workers",
            len(self.dates),
            self.n_workers,
        )
        # Bulk-fetch all daily bars once to avoid per-date DB round-trips.
        try:
            cache = DailyBarCache(self.ticker, max(self.dates))
        except Exception:
            log.warning("DailyBarCache failed; falling back to per-date queries", exc_info=True)
            cache = None

        # Bulk-fetch all intraday bars once to avoid 1-query-per-date.
        try:
            intraday_cache = IntradayCache(self.ticker, min(self.dates), max(self.dates))
        except Exception:
            log.warning("IntradayCache failed; falling back to per-date queries", exc_info=True)
            intraday_cache = None

        with ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="db",
        ) as pool:
            futures: list[tuple[str, Future]] = [
                (date, pool.submit(
                    prepare_env_data, self.ticker, date,
                    daily_cache=cache, intraday_cache=intraday_cache,
                ))
                for date in self.dates
            ]
            for date, future in futures:
                try:
                    norm, prices = future.result()
                    self._queue.put(DayData(date, norm, prices))
                except Exception as e:
                    log.error("DataPool: failed %s: %s", date, e)
                    self._queue.put(DayData(date, None, None, error=str(e)))

        self._queue.put(_SENTINEL)

    def __iter__(self):
        """Yield DayData in order. Blocks until each item is ready."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    def join(self, timeout: float = 300.0):
        if self._thread:
            self._thread.join(timeout=timeout)


# ── Training Pipeline ────────────────────────────────────────────────

class TrainingPipeline:
    """Walk-forward training pipeline with actor-based concurrency.

    Designed for the hardware profile:
      - 20 ARM cores (10 perf + 10 eff): DataPool uses efficiency cores
        for I/O-bound prefetching; training env stepping and Polars use
        performance cores.
      - Unified CPU-GPU memory (GB10): numpy arrays produced by DataPool
        threads are directly accessible on GPU — no explicit transfer.
      - Single GPU (48 SMs): training is serial per model.

    All dates are prefetched in parallel at startup (~10 MB for 250 days),
    then training iterates through them sequentially on the GPU.
    """

    def __init__(
        self,
        ticker: str,
        dates: list,
        timesteps_per_day: int = 100_000,
        initial_cash: float = 100_000.0,
        n_envs: int = 256,
        n_data_workers: int | None = None,
        prefetch_depth: int = 16,
        device: str = "cuda",
        net_arch: list[int] | None = None,
        batch_size: int = 1024,
        n_steps: int = 4096,
        ppo_kwargs: dict | None = None,
    ):
        self.ticker = ticker
        self.dates = [
            d.isoformat() if hasattr(d, "isoformat") else d for d in dates
        ]
        self.timesteps_per_day = timesteps_per_day
        self.initial_cash = initial_cash
        self.n_envs = n_envs
        self.n_data_workers = n_data_workers
        self.prefetch_depth = prefetch_depth
        self.device = device
        self.net_arch = net_arch or [512, 512]
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ppo_kwargs = ppo_kwargs or {}

    def run(
        self,
        on_result: Callable[[int, int, EvalResult], None] | None = None,
        model=None,
        on_day_trained: Callable[[object, int, str], None] | None = None,
        start_date: str | None = None,
    ) -> tuple[list[EvalResult], object]:
        """Run the full walk-forward pipeline.

        Args:
            on_result: ``(day_index, total_days, result)`` — called after
                each evaluation completes.
            model: Pre-existing PPO model to continue training.
            on_day_trained: ``(model, day_index, eval_date)`` — called
                after each training+eval cycle for checkpointing.
            start_date: ISO date string to resume from. If provided,
                skips dates before this.

        Returns:
            ``(results, final_model)``
        """
        from sbx import PPO

        if len(self.dates) < 2:
            log.error("Need at least 2 trading dates, got %d", len(self.dates))
            return [], model

        total_days = len(self.dates) - 1

        # Determine start index for resuming
        start_idx = 0
        if start_date:
            try:
                # Find the index of the date strictly AFTER the last trained date
                # Or the index of the start_date itself if it's in our list.
                for idx, d_str in enumerate(self.dates):
                    if d_str >= start_date:
                        start_idx = idx
                        break
                log.info("Pipeline: resuming from index %d (date %s)", start_idx, self.dates[start_idx])
            except Exception as e:
                log.warning("Pipeline: resume failed to find date %s, starting from 0: %s", start_date, e)

        # ── Batch Pre-compute (Polars-vectorized) ───────────────────
        # ...
        log.info("Pipeline: batch-processing indicators for %d dates...", len(self.dates))
        all_data_map = prepare_all_data(self.ticker, self.dates)
        log.info("Pipeline: pre-computation complete.")

        # ── Walk-forward loop (main thread, GPU) ────────────────────
        results: list[EvalResult] = []

        for i in range(start_idx, len(self.dates) - 1):
            train_date_str = self.dates[i]
            eval_date_str = self.dates[i + 1]

            train_arrays = all_data_map.get(train_date_str)
            eval_arrays = all_data_map.get(eval_date_str)

            if train_arrays is None:
                log.warning("Skipping train day %s (load failed)", train_date_str)
                continue
            if eval_arrays is None:
                log.warning("Skipping eval day %s (load failed)", eval_date_str)
                continue

            train_norm, train_prices = train_arrays
            eval_norm, eval_prices = eval_arrays

            day_start = time.monotonic()
            log.info(
                "[%4d/%d] TRAIN  %s → eval %s",
                i + 1, total_days, train_date_str, eval_date_str,
            )

            # Build JAX-accelerated vectorized env — zero CPU overhead for stepping
            vec_env = FastVectorizedStockTradingEnv(
                self.n_envs,
                train_norm,
                train_prices,
                initial_cash=self.initial_cash,
            )

            if model is None:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    device=self.device,
                    verbose=0,
                    policy_kwargs=dict(net_arch=self.net_arch),
                    batch_size=self.batch_size,
                    n_steps=self.n_steps,
                    **self.ppo_kwargs,
                )
            else:
                model.set_env(vec_env)

            ts_before = model.num_timesteps
            train_start = time.monotonic()
            model.learn(
                total_timesteps=self.timesteps_per_day,
                reset_num_timesteps=False,
            )
            train_end = time.monotonic()
            ts_after = model.num_timesteps

            # Evaluate on next day
            eval_env = StockTradingEnv.from_arrays(
                eval_norm,
                eval_prices,
                initial_cash=self.initial_cash,
            )
            eval_start = time.monotonic()
            result = _evaluate(
                model, eval_env, train_date_str, eval_date_str
            )
            eval_end = time.monotonic()
            result.elapsed_s = eval_end - day_start
            result.train_s = train_end - train_start
            result.eval_s = eval_end - eval_start
            result.train_timesteps = ts_after - ts_before
            results.append(result)

            vec_env.close()

            if on_result:
                on_result(i, total_days, result)
            if on_day_trained:
                on_day_trained(model, i, eval_date_str)

        return results, model


# ── Helpers ──────────────────────────────────────────────────────────

def _evaluate(model, env, train_date: str, eval_date: str) -> EvalResult:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break
    return EvalResult(
        train_date=train_date,
        eval_date=eval_date,
        steps=steps,
        total_reward=total_reward,
        final_value=info["portfolio_value"],
        return_pct=(info["portfolio_value"] / env.initial_cash - 1) * 100,
    )

"""Main entry point for the RL Model CLI.

Provides commands for training (single day or walk-forward) and evaluation.
"""

import csv
import glob
import io
import json
import logging
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import optuna
import rich_click as click
import zstandard as zstd
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_model.db import get_next_trading_date, get_trading_dates
from rl_model.env import (
    ACTION_TO_POSITION,
    StockTradingEnv,
    prepare_env_data,
    prepare_env_data_with_timestamps,
)
from rl_model.pipeline import EvalResult, TrainingPipeline
from rl_model.vec_env import FastVectorizedStockTradingEnv

# ── Global CLI Setup ────────────────────────────────────────────────

_global_state = {
    "model": None,
    "ticker": "unknown",
    "last_date": "unknown",
    "interrupted": False
}

def _on_interrupt(signum, frame):
    """Universal interrupt handler to save progress."""
    _global_state["interrupted"] = True
    model = _global_state["model"]
    if model:
        ticker = _global_state["ticker"]
        date = _global_state["last_date"]
        path = f"results/interrupted_{ticker}_{date}"
        model.save(path)
        print(f"\n[INTERRUPT] Model saved to {path}.zip")
    sys.exit(130)

@click.group()
def cli():
    """RL Stock Trading Model CLI."""
    pass

# ── Shared Helpers ──────────────────────────────────────────────────

def setup_logging(level=logging.INFO, name=None):
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, handlers=[console])
    
    # Suppress noisy output
    logging.getLogger("sbx").setLevel(logging.WARNING)
    if name:
        return logging.getLogger(name)
    return root_logger

def load_params(params_file: str | None, default_net_arch: list[int], default_batch_size: int):
    """Load tuned parameters from JSON, returning ppo_kwargs, net_arch, and batch_size."""
    # Finance-specific defaults to override generic SBX defaults
    extra_ppo_kwargs = {
        "ent_coef": 0.01,
        "learning_rate": 0.0001,
    }
    net_arch = default_net_arch
    batch_size = default_batch_size
    
    if params_file:
        with open(params_file) as f:
            tuned = json.load(f)
        ppo_keys = {"learning_rate", "ent_coef", "gamma", "clip_range", "n_epochs", "gae_lambda", "vf_coef"}
        # Update with tuned params
        for k, v in tuned.items():
            if k in ppo_keys:
                extra_ppo_kwargs[k] = v
        if "net_arch" in tuned:
            net_arch = [int(x) for x in str(tuned["net_arch"]).split(",")]
        if "batch_size" in tuned:
            batch_size = int(tuned["batch_size"])
            
    return extra_ppo_kwargs, net_arch, batch_size

class TimeCheckpointCallback(BaseCallback):
    """Saves a model checkpoint every N minutes."""
    def __init__(self, save_path: str, interval_minutes: float, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.interval_seconds = interval_minutes * 60
        self.last_save_time = time.time()
        self.log = logging.getLogger("rl_model")

    def _on_step(self) -> bool:
        if (time.time() - self.last_save_time) >= self.interval_seconds:
            self.model.save(self.save_path)
            self.last_save_time = time.time()
            if self.verbose > 0:
                self.log.info("Time-based checkpoint saved to %s", self.save_path)
        return True

# ── Tuning Helpers ──────────────────────────────────────────────────

def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "n_epochs": trial.suggest_int("n_epochs", 3, 15),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.0),
        "net_arch": trial.suggest_categorical("net_arch", ["64,64", "128,128", "256,256", "512,512"]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096]),
    }

def objective(trial, train_data, train_prices, eval_data, eval_prices,
              timesteps, n_envs, n_steps, initial_cash, device):
    params = suggest_params(trial)
    net_arch = [int(x) for x in params.pop("net_arch").split(",")]
    batch_size = params.pop("batch_size")

    vec_env = FastVectorizedStockTradingEnv(n_envs, train_data, train_prices, initial_cash=initial_cash)
    model = PPO(
        "MlpPolicy", vec_env, device=device, verbose=0,
        policy_kwargs=dict(net_arch=net_arch),
        batch_size=batch_size, n_steps=n_steps, **params
    )
    _global_state["model"] = model
    model.learn(total_timesteps=timesteps)
    vec_env.close()

    eval_env = StockTradingEnv.from_arrays(eval_data, eval_prices, initial_cash=initial_cash)
    obs, info = eval_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated:
            break
    return (info["portfolio_value"] / initial_cash - 1) * 100

# ── Tune Command ────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.argument("date")
@click.option("--n-trials", type=int, default=50, show_default=True)
@click.option("--timesteps", type=int, default=200_000, show_default=True)
@click.option("--n-envs", type=int, default=256, show_default=True)
@click.option("--n-steps", type=int, default=512, show_default=True)
@click.option("--initial-cash", type=float, default=100_000.0, show_default=True)
@click.option("--study-name", type=str, default="ppo-tune", show_default=True)
@click.option("--storage", type=str, default=None)
@click.option("--output-dir", type=str, default="results", show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
def tune(ticker, date, n_trials, timesteps, n_envs, n_steps, initial_cash,
         study_name, storage, output_dir, device):
    """Hyperparameter tuning using Optuna."""
    log = setup_logging()
    signal.signal(signal.SIGINT, _on_interrupt)
    _global_state["ticker"] = ticker.lower()
    _global_state["last_date"] = str(date)
    
    date_obj = datetime.strptime(date, "%Y-%m-%d").date() if isinstance(date, str) else date
    train_date = get_next_trading_date(ticker, date_obj - timedelta(days=1))
    eval_date = get_next_trading_date(ticker, train_date)
    log.info("Tune: Train %s, Eval %s", train_date, eval_date)

    train_data, train_prices = prepare_env_data(ticker, str(train_date))
    eval_data, eval_prices = prepare_env_data(ticker, str(eval_date))

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, train_data, train_prices, eval_data, eval_prices,
                            timesteps, n_envs, n_steps, initial_cash, device),
        n_trials=n_trials,
    )

    best = study.best_trial
    log.info("Best trial #%d: return=%.4f%%", best.number, best.value)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "best_params.json")
    with open(out_path, "w") as f:
        json.dump({"best_return_pct": best.value, **best.params}, f, indent=2)
    log.info("Saved best params to %s", out_path)

# ── Evaluate Command ────────────────────────────────────────────────

ACTION_LABELS = {
    0: "strong_sell",
    1: "sell",
    2: "hold",
    3: "buy",
    4: "strong_buy",
}

TRADE_FIELDS = [
    "timestamp", "ticker", "action", "side", "volume", "price",
    "portfolio_value",
]

def _evaluate_day_logic(model, ticker, date, initial_cash, timestamps, normalized_data,
                        raw_close_prices, writer):
    env = StockTradingEnv.from_arrays(
        normalized_data, raw_close_prices, initial_cash=initial_cash,
    )
    obs, info = env.reset()
    prev_shares = 0.0
    day_trades = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        shares_delta = info["shares"] - prev_shares
        if shares_delta != 0.0:
            step_idx = info["step"] - 1
            ts = timestamps[step_idx] if step_idx < len(timestamps) else timestamps[-1]
            writer.writerow({
                "timestamp": ts,
                "ticker": ticker.upper(),
                "action": ACTION_LABELS[action],
                "side": "BUY" if shares_delta > 0 else "SELL",
                "volume": f"{abs(shares_delta):.4f}",
                "price": f"{info['current_price']:.4f}",
                "portfolio_value": f"{info['portfolio_value']:.2f}",
            })
            day_trades += 1

        prev_shares = info["shares"]
        if terminated:
            break

    return_pct = (info["portfolio_value"] / initial_cash - 1) * 100
    return return_pct, info["portfolio_value"], day_trades

@cli.command()
@click.argument("ticker")
@click.argument("start_date")
@click.argument("end_date")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True),
              help="Path to trained model .zip file.")
@click.option("--initial-cash", type=float, default=100_000.0, show_default=True)
@click.option("--output", type=str, default=None,
              help="Output path for trades CSV.")
def evaluate(ticker, start_date, end_date, model_path, initial_cash, output):
    """Evaluate a trained model across a date range."""
    log = setup_logging(name="evaluate")
    
    m_path = model_path[:-4] if model_path.endswith(".zip") else model_path
    log.info("Loading model from %s", model_path)
    ppo_model = PPO.load(m_path, device="cuda")

    dates = get_trading_dates(ticker, start_date, end_date)
    if not dates:
        log.error("No trading dates found.")
        return

    os.makedirs("results", exist_ok=True)
    if output is None:
        output = f"results/trades_{ticker.lower()}_{start_date}_{end_date}.csv.zst"

    cctx = zstd.ZstdCompressor()
    with open(output, "wb") as raw_file:
        with cctx.stream_writer(raw_file) as compressor:
            with io.TextIOWrapper(compressor, encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=TRADE_FIELDS)
                writer.writeheader()

                returns = []
                total_trades = 0
                for i, date in enumerate(dates):
                    date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
                    try:
                        norm, prices, ts = prepare_env_data_with_timestamps(ticker, date_str)
                    except ValueError as e:
                        log.warning("Skipping %s: %s", date_str, e)
                        continue

                    ret_pct, final_val, d_trades = _evaluate_day_logic(
                        ppo_model, ticker, date_str, initial_cash, ts, norm, prices, writer
                    )
                    returns.append(ret_pct)
                    total_trades += d_trades
                    log.info("[%d/%d] %s ret=%+.4f%%", i+1, len(dates), date_str, ret_pct)

    log.info("Evaluation complete. Trades saved to %s", output)

# ── Train Day Command ───────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.argument("date")
@click.option("--timesteps", type=int, default=1_000_000, show_default=True)
@click.option("--initial-cash", type=float, default=100_000.0, show_default=True)
@click.option("--output", default="ppo_stock_trading", show_default=True)
@click.option("--n-envs", type=int, default=256, show_default=True)
@click.option("--net-arch", type=str, default="512,512", show_default=True)
@click.option("--batch-size", type=int, default=1024, show_default=True)
@click.option("--n-steps", type=int, default=512, show_default=True)
@click.option("--save-interval", type=float, default=10.0, show_default=True,
              help="Checkpoint interval in minutes.")
@click.option("--from-scratch", is_flag=True, default=False)
@click.option("--params-file", type=click.Path(exists=True), default=None)
def train_day(ticker, date, timesteps, initial_cash, output, n_envs, net_arch,
              batch_size, n_steps, save_interval, from_scratch, params_file):
    """Train a PPO agent on a single day of intraday data."""
    setup_logging()
    signal.signal(signal.SIGINT, _on_interrupt)
    _global_state["ticker"] = ticker.lower()
    _global_state["last_date"] = str(date)

    default_net_arch = [int(x) for x in net_arch.split(",")]
    
    extra_ppo_kwargs, parsed_net_arch, batch_size = load_params(
        params_file, default_net_arch, batch_size
    )
    if params_file:
        print(f"Loaded tuned params from {params_file}: {extra_ppo_kwargs}")

    norm, prices = prepare_env_data(ticker, date)
    vec_env = FastVectorizedStockTradingEnv(n_envs, norm, prices, initial_cash=initial_cash)

    ppo_kwargs = dict(
        device="cuda",
        verbose=1,
        policy_kwargs=dict(net_arch=parsed_net_arch),
        batch_size=batch_size,
        n_steps=n_steps,
        **extra_ppo_kwargs,
    )

    model_path = f"{output}.zip"
    if not from_scratch and os.path.exists(model_path):
        model = PPO.load(output, env=vec_env, **ppo_kwargs)
    else:
        model = PPO("MlpPolicy", vec_env, **ppo_kwargs)
    
    _global_state["model"] = model
    
    # Enable periodic checkpointing
    ckpt_path = f"{output}_checkpoint"
    callback = TimeCheckpointCallback(ckpt_path, save_interval, verbose=1)
    
    print(f"Training started (saving checkpoints to {ckpt_path}.zip every {save_interval} mins)")
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(output)
    vec_env.close()

# ── Train Walk Command ──────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.argument("start_date")
@click.argument("end_date")
@click.option("--timesteps", type=int, default=100_000, show_default=True)
@click.option("--initial-cash", type=float, default=100_000.0, show_default=True)
@click.option("--save-interval", type=float, default=10.0, show_default=True,
              help="Checkpoint interval in minutes.")
@click.option("--n-envs", type=int, default=256, show_default=True)
@click.option("--net-arch", type=str, default="512,512", show_default=True)
@click.option("--batch-size", type=int, default=1024, show_default=True)
@click.option("--n-steps", type=int, default=512, show_default=True)
@click.option("--model", "model_path", type=click.Path(exists=True), default=None)
@click.option("--resume", is_flag=True, default=False,
              help="Auto-resume from the latest checkpoint or provided model.")
@click.option("--from-scratch", is_flag=True, default=False)
@click.option("--params-file", type=click.Path(exists=True), default=None)
def train_walk(ticker, start_date, end_date, timesteps, initial_cash,
               save_interval, n_envs, net_arch, batch_size,
               n_steps, model_path, resume, from_scratch, params_file):
    """Walk-forward PPO training across a date range."""
    log = setup_logging(name="walk")
    signal.signal(signal.SIGINT, _on_interrupt)
    _global_state["ticker"] = ticker.lower()
    _global_state["last_date"] = str(start_date)
    
    last_save_time = time.time()

    dates = get_trading_dates(ticker, start_date, end_date)
    if len(dates) < 2:
        return

    default_net_arch = [int(x) for x in net_arch.split(",")]
    ppo_kwargs, parsed_net_arch, batch_size = load_params(
        params_file, default_net_arch, batch_size
    )
    if params_file:
        log.info("Loaded tuned params from %s: %s", params_file, ppo_kwargs)

    def on_result(i, total, result):
        log.info("[%d/%d] %s ret=%+.4f%%", i+1, total, result.eval_date, result.return_pct)

    def on_day_trained(model, i, eval_date):
        nonlocal last_save_time
        _global_state["model"] = model
        _global_state["last_date"] = eval_date
        
        elapsed_mins = (time.time() - last_save_time) / 60
        if save_interval and elapsed_mins >= save_interval:
            path = f"results/checkpoint_{ticker.lower()}_{eval_date}.zip"
            model.save(path)
            log.info("CKPT   saved %s (%.1f mins elapsed)", path, elapsed_mins)
            last_save_time = time.time()

    # ── Resume logic ────────────────────────────────────────────────

    existing_model = None
    resume_point = None
    if not from_scratch and (resume or model_path):
        target_path = model_path
        if not target_path:
            # Auto-find latest checkpoint or final model for this ticker
            candidates = (glob.glob(f"results/checkpoint_{ticker.lower()}_*.zip") +
                         glob.glob(f"results/model_{ticker.lower()}_*.zip"))
            if candidates:
                target_path = max(candidates, key=os.path.getmtime)
                log.info("RESUME auto-discovered latest: %s", target_path)

        if target_path and os.path.exists(target_path):
            log.info("RESUME loading from %s", target_path)
            base_path = target_path[:-4] if target_path.endswith(".zip") else target_path
            existing_model = PPO.load(base_path, device="cuda")
            
            # Extract date for start point
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(target_path))
            if date_match:
                resume_point = date_match.group(1)
                log.info("RESUME starting from date: %s", resume_point)

    # ── Run pipeline ─────────────────────────────────────────────────

    pipeline = TrainingPipeline(
        ticker=ticker, dates=dates, timesteps_per_day=timesteps,
        initial_cash=initial_cash, n_envs=n_envs, net_arch=parsed_net_arch,
        batch_size=batch_size, n_steps=n_steps, ppo_kwargs=ppo_kwargs,
    )

    results, model = pipeline.run(
        on_result=on_result, 
        on_day_trained=on_day_trained, 
        model=existing_model,
        start_date=resume_point
    )
    
    # Save final model with dates in filename
    final_path = f"results/model_{ticker.lower()}_{dates[0]}_{dates[-1]}"
    model.save(final_path)
    log.info("DONE   final model saved to %s.zip", final_path)

if __name__ == "__main__":
    cli()

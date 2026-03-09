from datetime import datetime

import gymnasium
import numpy as np
import polars as pl

from rl_model.db import get_date_window
from rl_model.indicators import add_all_indicators, normalize

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "vwap",
    "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26", "ema_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "atr_14",
    "obv",
    "stoch_k", "stoch_d",
]

ACTION_TO_POSITION = {
    0: 0.00,  # strong sell
    1: 0.25,  # sell
    2: 0.50,  # hold
    3: 0.75,  # buy
    4: 1.00,  # strong buy
}

_ACTION_TO_POSITION = np.array([0.00, 0.25, 0.50, 0.75, 1.00])


def _load_env_arrays(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
    """Shared loader returning (normalized_data, raw_close_prices, timestamps)."""
    raw_df = get_date_window(
        ticker, date, include_daily=include_daily,
        daily_cache=daily_cache, intraday_cache=intraday_cache,
    )
    indicator_df = add_all_indicators(raw_df)

    target_date = datetime.strptime(date, "%Y-%m-%d").date() if isinstance(date, str) else date
    intraday_mask = raw_df["timestamp"].dt.date() == target_date

    raw_intraday = indicator_df.filter(intraday_mask)
    norm_intraday = normalize(raw_intraday)

    if len(norm_intraday) == 0:
        raise ValueError(f"No intraday data for {ticker} on {date}")

    normalized_data = np.nan_to_num(
        norm_intraday.select(FEATURE_COLS).to_numpy(), nan=0.0
    ).astype(np.float32)
    raw_close_prices = raw_intraday["close"].to_numpy().astype(np.float64)
    timestamps = raw_intraday["timestamp"].to_list()

    return normalized_data, raw_close_prices, timestamps


def prepare_env_data(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
    """Load and process data for a trading env.

    Returns (normalized_data, raw_close_prices) numpy arrays.
    Can be called in a background thread to overlap I/O with training.
    """
    normalized_data, raw_close_prices, _ = _load_env_arrays(
        ticker, date, include_daily=include_daily,
        daily_cache=daily_cache, intraday_cache=intraday_cache,
    )
    return normalized_data, raw_close_prices


def prepare_env_data_with_timestamps(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
    """Like prepare_env_data but also returns intraday timestamps.

    Returns (normalized_data, raw_close_prices, timestamps).
    """
    return _load_env_arrays(
        ticker, date, include_daily=include_daily,
        daily_cache=daily_cache, intraday_cache=intraday_cache,
    )


def prepare_all_data(ticker, dates, include_daily=True):
    """Batch load and process data for multiple dates efficiently.
    
    Instead of per-day Polars operations, this performs bulk indicators
    and global normalization (or per-window normalization) which is
    faster on high-core-count ARM systems.
    """
    from rl_model.db import DailyBarCache, IntradayCache
    
    # Bulk fetch everything
    daily_cache = DailyBarCache(ticker, max(dates))
    intraday_cache = IntradayCache(ticker, min(dates), max(dates))
    
    raw_df = get_date_window(
        ticker, dates[-1], include_daily=include_daily,
        daily_cache=daily_cache, intraday_cache=intraday_cache,
    )
    # add_all_indicators is vectorized across the entire dataframe
    indicator_df = add_all_indicators(raw_df)
    
    results = {}
    for date in dates:
        target_date = datetime.strptime(date, "%Y-%m-%d").date() if isinstance(date, str) else date
        intraday_mask = raw_df["timestamp"].dt.date() == target_date
        raw_intraday = indicator_df.filter(intraday_mask)
        
        # We still normalize per-day to match existing behavior, but 
        # indicator calculation was done in one big pass.
        norm_intraday = normalize(raw_intraday)
        
        if len(norm_intraday) > 0:
            norm_arr = np.nan_to_num(
                norm_intraday.select(FEATURE_COLS).to_numpy(), nan=0.0
            ).astype(np.float32)
            prices_arr = raw_intraday["close"].to_numpy().astype(np.float32)
            results[date] = (norm_arr, prices_arr)
            
    return results


class StockTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ticker: str,
        date: str,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        render_mode: str | None = None,
        include_daily: bool = True,
    ):
        super().__init__()
        self.ticker = ticker
        self.date = date
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.render_mode = render_mode

        normalized_data, raw_close_prices = prepare_env_data(
            ticker, date, include_daily
        )
        self._init_from_arrays(normalized_data, raw_close_prices)

    @classmethod
    def from_arrays(
        cls,
        normalized_data: np.ndarray,
        raw_close_prices: np.ndarray,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        render_mode: str | None = None,
    ):
        """Create env from pre-computed numpy arrays. Skips DB and indicator work."""
        env = cls.__new__(cls)
        gymnasium.Env.__init__(env)
        env.initial_cash = initial_cash
        env.transaction_cost_pct = transaction_cost_pct
        env.render_mode = render_mode
        env._init_from_arrays(normalized_data, raw_close_prices)
        return env

    def _init_from_arrays(self, normalized_data, raw_close_prices):
        self._num_steps = len(normalized_data)
        self._normalized_data = normalized_data
        self._raw_close_prices = raw_close_prices

        self.action_space = gymnasium.spaces.Discrete(5)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(normalized_data.shape[1],),
            dtype=np.float32,
        )

        self._current_step = 0
        self._cash = self.initial_cash
        self._shares = 0.0
        self._portfolio_value = self.initial_cash

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._cash = self.initial_cash
        self._shares = 0.0
        self._portfolio_value = self.initial_cash
        return self._get_obs(), self._get_info()

    def step(self, action):
        current_price = self._raw_close_prices[self._current_step]
        current_portfolio_value = self._cash + self._shares * current_price

        # Rebalance to target allocation
        target_pct = _ACTION_TO_POSITION[action]
        target_stock_value = target_pct * current_portfolio_value
        target_shares = target_stock_value / current_price
        shares_delta = target_shares - self._shares

        # Transaction cost
        trade_value = abs(shares_delta) * current_price
        cost = trade_value * self.transaction_cost_pct

        self._shares = target_shares
        self._cash = current_portfolio_value - target_stock_value - cost

        # Advance
        self._current_step += 1
        terminated = self._current_step >= self._num_steps

        if terminated:
            new_price = current_price
        else:
            new_price = self._raw_close_prices[self._current_step]

        new_portfolio_value = self._cash + self._shares * new_price
        reward = float(new_portfolio_value - self._portfolio_value)
        self._portfolio_value = new_portfolio_value

        obs = self._normalized_data[-1] if terminated else self._get_obs()
        return obs, reward, terminated, False, self._get_info()

    def _get_obs(self):
        return self._normalized_data[self._current_step]

    def _get_info(self):
        step = min(self._current_step, self._num_steps - 1)
        return {
            "step": self._current_step,
            "cash": self._cash,
            "shares": self._shares,
            "portfolio_value": self._portfolio_value,
            "current_price": float(self._raw_close_prices[step]),
        }

import polars as pl

from rl_model.db import get_date_window


def add_sma(df: pl.DataFrame, periods: list[int] = [10, 20, 50]) -> pl.DataFrame:
    return df.with_columns(
        pl.col("close").rolling_mean(p).alias(f"sma_{p}") for p in periods
    )


def add_ema(df: pl.DataFrame, periods: list[int] = [12, 26, 50]) -> pl.DataFrame:
    return df.with_columns(
        pl.col("close").ewm_mean(span=p, ignore_nulls=True).alias(f"ema_{p}")
        for p in periods
    )


def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(period)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(period)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return df.with_columns(rsi.alias(f"rsi_{period}"))


def add_macd(
    df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pl.DataFrame:
    ema_fast = pl.col("close").ewm_mean(span=fast, ignore_nulls=True)
    ema_slow = pl.col("close").ewm_mean(span=slow, ignore_nulls=True)
    df = df.with_columns((ema_fast - ema_slow).alias("macd"))
    df = df.with_columns(
        pl.col("macd").ewm_mean(span=signal, ignore_nulls=True).alias("macd_signal")
    )
    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    )
    return df


def add_bollinger_bands(
    df: pl.DataFrame, period: int = 20, std: float = 2.0
) -> pl.DataFrame:
    mid = pl.col("close").rolling_mean(period)
    rolling_std = pl.col("close").rolling_std(period)
    return df.with_columns(
        mid.alias("bb_mid"),
        (mid + std * rolling_std).alias("bb_upper"),
        (mid - std * rolling_std).alias("bb_lower"),
    )


def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(tr.rolling_mean(period).alias(f"atr_{period}"))


def add_obv(df: pl.DataFrame) -> pl.DataFrame:
    sign = (
        pl.when(pl.col("close") > pl.col("close").shift(1))
        .then(1)
        .when(pl.col("close") < pl.col("close").shift(1))
        .then(-1)
        .otherwise(0)
    )
    return df.with_columns((sign * pl.col("volume")).cum_sum().alias("obv"))


def add_stochastic(
    df: pl.DataFrame, k_period: int = 14, d_period: int = 3
) -> pl.DataFrame:
    lowest = pl.col("low").rolling_min(k_period)
    highest = pl.col("high").rolling_max(k_period)
    stoch_k = 100 * (pl.col("close") - lowest) / (highest - lowest)
    df = df.with_columns(stoch_k.alias("stoch_k"))
    df = df.with_columns(
        pl.col("stoch_k").rolling_mean(d_period).alias("stoch_d")
    )
    return df


def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    prev_close = pl.col("close").shift(1)
    delta = pl.col("close").diff()

    # Phase 1: all indicators that depend only on original columns (14 cols)
    df = df.with_columns(
        # SMA
        *[pl.col("close").rolling_mean(p).alias(f"sma_{p}") for p in (10, 20, 50)],
        # EMA
        *[pl.col("close").ewm_mean(span=p, ignore_nulls=True).alias(f"ema_{p}")
          for p in (12, 26, 50)],
        # RSI
        (100 - (100 / (1 + delta.clip(lower_bound=0).rolling_mean(14)
                        / (-delta.clip(upper_bound=0)).rolling_mean(14)))).alias("rsi_14"),
        # Bollinger Bands
        pl.col("close").rolling_mean(20).alias("bb_mid"),
        (pl.col("close").rolling_mean(20) + 2.0 * pl.col("close").rolling_std(20)).alias("bb_upper"),
        (pl.col("close").rolling_mean(20) - 2.0 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        # ATR
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ).rolling_mean(14).alias("atr_14"),
        # OBV
        (pl.when(pl.col("close") > prev_close).then(1)
         .when(pl.col("close") < prev_close).then(-1)
         .otherwise(0) * pl.col("volume")).cum_sum().alias("obv"),
        # MACD line
        (pl.col("close").ewm_mean(span=12, ignore_nulls=True)
         - pl.col("close").ewm_mean(span=26, ignore_nulls=True)).alias("macd"),
        # Stochastic K
        (100 * (pl.col("close") - pl.col("low").rolling_min(14))
         / (pl.col("high").rolling_max(14) - pl.col("low").rolling_min(14))).alias("stoch_k"),
    )

    # Phase 2: indicators that depend on phase 1 outputs
    df = df.with_columns(
        pl.col("macd").ewm_mean(span=9, ignore_nulls=True).alias("macd_signal"),
        pl.col("stoch_k").rolling_mean(3).alias("stoch_d"),
    )

    # Phase 3: depends on phase 2
    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
    )

    return df


SKIP_NORMALIZE = {"id", "ticker", "date", "datetime", "timestamp"}


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype.is_numeric() and c not in SKIP_NORMALIZE
    ]
    return df.with_columns(
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
        for c in numeric_cols
    )


def get_indicators_at(
    ticker: str, date: str, normalized: bool = True
) -> pl.DataFrame:
    df = get_date_window(ticker, date)
    df = add_all_indicators(df)
    if normalized:
        df = normalize(df)
    return df

from unittest.mock import patch

import polars as pl

from rl_model.indicators import (
    add_all_indicators,
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_obv,
    add_rsi,
    add_sma,
    add_stochastic,
    get_indicators_at,
    normalize,
)


def make_ohlcv(n: int = 50) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    import random

    random.seed(42)
    close = 100.0
    rows = []
    for i in range(n):
        change = random.uniform(-2, 2)
        o = close + random.uniform(-1, 1)
        c = close + change
        h = max(o, c) + random.uniform(0, 1)
        l = min(o, c) - random.uniform(0, 1)
        v = random.randint(500_000, 3_000_000)
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        close = c
    return pl.DataFrame(rows)


def test_add_sma():
    df = add_sma(make_ohlcv())
    assert "sma_10" in df.columns
    assert "sma_20" in df.columns
    assert "sma_50" in df.columns
    assert df["sma_10"][9] is not None


def test_add_sma_custom_periods():
    df = add_sma(make_ohlcv(), periods=[5, 15])
    assert "sma_5" in df.columns
    assert "sma_15" in df.columns
    assert "sma_10" not in df.columns


def test_add_ema():
    df = add_ema(make_ohlcv())
    assert "ema_12" in df.columns
    assert "ema_26" in df.columns
    assert "ema_50" in df.columns
    assert df["ema_12"][0] is not None


def test_add_rsi():
    df = add_rsi(make_ohlcv())
    assert "rsi_14" in df.columns
    non_null = df.filter(pl.col("rsi_14").is_not_null())
    assert non_null["rsi_14"].min() >= 0
    assert non_null["rsi_14"].max() <= 100


def test_add_rsi_custom_period():
    df = add_rsi(make_ohlcv(), period=7)
    assert "rsi_7" in df.columns


def test_add_macd():
    df = add_macd(make_ohlcv())
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_add_bollinger_bands():
    df = add_bollinger_bands(make_ohlcv())
    assert "bb_upper" in df.columns
    assert "bb_mid" in df.columns
    assert "bb_lower" in df.columns
    non_null = df.filter(pl.col("bb_mid").is_not_null())
    assert (non_null["bb_upper"] >= non_null["bb_mid"]).all()
    assert (non_null["bb_lower"] <= non_null["bb_mid"]).all()


def test_add_atr():
    df = add_atr(make_ohlcv())
    assert "atr_14" in df.columns
    non_null = df.filter(pl.col("atr_14").is_not_null())
    assert (non_null["atr_14"] >= 0).all()


def test_add_obv():
    df = add_obv(make_ohlcv())
    assert "obv" in df.columns


def test_add_stochastic():
    df = add_stochastic(make_ohlcv())
    assert "stoch_k" in df.columns
    assert "stoch_d" in df.columns
    non_null = df.filter(pl.col("stoch_k").is_not_null())
    assert non_null["stoch_k"].min() >= 0
    assert non_null["stoch_k"].max() <= 100


def test_add_all_indicators():
    df = add_all_indicators(make_ohlcv())
    expected = [
        "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "ema_50",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14",
        "obv",
        "stoch_k", "stoch_d",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


INDICATOR_COLS = [
    "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26", "ema_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "atr_14",
    "obv",
    "stoch_k", "stoch_d",
]


def test_normalize_zero_mean_unit_std():
    df = normalize(add_all_indicators(make_ohlcv()))
    for c in df.columns:
        if not df[c].dtype.is_numeric():
            continue
        col = df[c].drop_nulls()
        if len(col) == 0 or col.std() == 0:
            continue
        assert abs(col.mean()) < 1e-6, f"{c} mean not ~0: {col.mean()}"
        assert abs(col.std() - 1.0) < 1e-6, f"{c} std not ~1: {col.std()}"


def test_normalize_skips_non_numeric():
    df = make_ohlcv().with_columns(pl.lit("AAPL").alias("ticker"))
    df = normalize(df)
    assert df["ticker"][0] == "AAPL"


@patch("rl_model.indicators.get_date_window")
def test_get_indicators_at_normalized(mock_window):
    mock_window.return_value = make_ohlcv(90)

    result = get_indicators_at("AAPL", "2026-02-13")

    assert isinstance(result, pl.DataFrame)
    for col in ["close", "sma_10", "ema_12"]:
        vals = result[col].drop_nulls()
        assert abs(vals.mean()) < 1e-6, f"{col} not normalized"


@patch("rl_model.indicators.get_date_window")
def test_get_indicators_at_unnormalized(mock_window):
    mock_window.return_value = make_ohlcv(90)

    result = get_indicators_at("AAPL", "2026-02-13", normalized=False)

    assert result["close"].mean() > 1.0


@patch("rl_model.indicators.get_date_window")
def test_get_indicators_at(mock_window):
    mock_window.return_value = make_ohlcv(90)

    result = get_indicators_at("AAPL", "2026-02-13")

    mock_window.assert_called_once_with("AAPL", "2026-02-13")
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 90
    for col in INDICATOR_COLS:
        assert col in result.columns, f"Missing column: {col}"

from datetime import date, datetime
from unittest.mock import MagicMock, call, patch

import pytest

import polars as pl

import rl_model.db as db


@patch("rl_model.db._get_pool")
def test_get_connection_uses_pool(mock_get_pool):
    mock_pool = MagicMock()
    mock_get_pool.return_value = mock_pool
    db.get_connection()
    mock_pool.connection.assert_called_once()


@patch("rl_model.db._query")
@patch("rl_model.db.get_connection")
def test_get_last_n_days_default_n(mock_get_conn, mock_query):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_query.return_value = pl.DataFrame({"date": ["2026-02-26"], "close": [150.0]})

    df = db.get_last_n_days("AAPL", "2026-02-26")

    mock_query.assert_called_once()
    call_args = mock_query.call_args
    assert "aapl_daily" in call_args[0][0]
    assert call_args[1]["params"] == ("2026-02-26", 90)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert df["close"][0] == 150.0
    mock_conn.close.assert_called_once()


@patch("rl_model.db._query")
@patch("rl_model.db.get_connection")
def test_get_last_n_days_custom_n(mock_get_conn, mock_query):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_query.return_value = pl.DataFrame()

    df = db.get_last_n_days("TSLA", "2026-01-15", n=60)

    call_args = mock_query.call_args
    assert "tsla_daily" in call_args[0][0]
    assert call_args[1]["params"] == ("2026-01-15", 60)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0


@patch("rl_model.db._query")
@patch("rl_model.db.get_connection")
def test_get_last_n_days_returns_multiple_rows(mock_get_conn, mock_query):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_query.return_value = pl.DataFrame({
        "date": ["2026-02-26", "2026-02-25", "2026-02-24"],
        "close": [150.0, 148.5, 149.0],
    })

    df = db.get_last_n_days("GOOG", "2026-02-26", n=3)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert df["date"][0] == "2026-02-26"


@patch("rl_model.db._query")
@patch("rl_model.db.get_connection")
def test_get_intraday_data(mock_get_conn, mock_query):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_query.return_value = pl.DataFrame({
        "timestamp": ["2026-02-13 09:30:00", "2026-02-13 09:31:00"],
        "close": [260.0, 260.5],
    })

    df = db.get_intraday_data("AAPL", "2026-02-13")

    mock_query.assert_called_once()
    call_args = mock_query.call_args
    assert "aapl_historical" in call_args[0][0]
    assert call_args[1]["params"] == ("2026-02-13", "2026-02-14")
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    mock_conn.close.assert_called_once()


@patch("rl_model.db.get_intraday_data")
@patch("rl_model.db.get_last_n_days")
def test_get_date_window_without_daily(mock_last_n, mock_intraday):
    from datetime import datetime

    mock_intraday.return_value = pl.DataFrame({
        "timestamp": [datetime(2026, 2, 13, 9, 30), datetime(2026, 2, 13, 9, 31)],
        "open": [260.0, 260.5], "high": [261.0, 261.5],
        "low": [259.5, 260.0], "close": [260.5, 261.0],
        "volume": [50000, 60000], "vwap": [260.2, 260.7],
    })

    result = db.get_date_window("AAPL", "2026-02-13", include_daily=False)

    mock_last_n.assert_not_called()
    mock_intraday.assert_called_once_with("AAPL", "2026-02-13")
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


@patch("rl_model.db.get_intraday_data")
@patch("rl_model.db.get_last_n_days")
def test_get_date_window(mock_last_n, mock_intraday):
    from datetime import date, datetime

    mock_last_n.return_value = pl.DataFrame({
        "date": [date(2026, 2, 11), date(2026, 2, 12)],
        "open": [258.0, 259.0], "high": [260.0, 261.0],
        "low": [257.0, 258.0], "close": [259.0, 260.0],
        "volume": [1000000, 1100000], "vwap": [258.5, 259.5],
    })
    mock_intraday.return_value = pl.DataFrame({
        "timestamp": [datetime(2026, 2, 13, 9, 30), datetime(2026, 2, 13, 9, 31)],
        "open": [260.0, 260.5], "high": [261.0, 261.5],
        "low": [259.5, 260.0], "close": [260.5, 261.0],
        "volume": [50000, 60000], "vwap": [260.2, 260.7],
    })

    result = db.get_date_window("AAPL", "2026-02-13")

    mock_last_n.assert_called_once_with("AAPL", "2026-02-12")
    mock_intraday.assert_called_once_with("AAPL", "2026-02-13")
    assert isinstance(result, pl.DataFrame)
    assert "timestamp" in result.columns
    assert len(result) == 4
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


@patch("rl_model.db.get_connection")
def test_get_trading_dates(mock_get_conn):
    from datetime import date

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        (date(2026, 2, 10),),
        (date(2026, 2, 11),),
        (date(2026, 2, 12),),
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_get_conn.return_value.__enter__ = lambda s: mock_conn
    mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = db.get_trading_dates("AAPL", "2026-02-10", "2026-02-12")

    mock_cur.execute.assert_called_once()
    sql_arg = mock_cur.execute.call_args[0][0]
    assert "aapl_historical" in sql_arg
    assert result == [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)]


# ── DailyBarCache ────────────────────────────────────────────────────


def _make_mock_connection(rows):
    """Create a mock connection factory that returns *rows* via _query."""
    mock_conn = MagicMock()

    def conn_fn():
        return mock_conn

    return conn_fn, mock_conn, rows


def test_daily_bar_cache_single_query():
    """DailyBarCache makes exactly one _query call on construction."""
    rows = pl.DataFrame({
        "date": [date(2026, 1, d) for d in range(1, 11)],
        "open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10,
        "close": [100.5] * 10, "volume": [1000] * 10, "vwap": [100.2] * 10,
    })
    conn_fn, mock_conn, _ = _make_mock_connection(rows)

    with patch("rl_model.db._query", return_value=rows) as mock_query:
        cache = db.DailyBarCache("AAPL", date(2026, 1, 15), connection_fn=conn_fn)

        assert mock_query.call_count == 1
        sql_arg = mock_query.call_args[0][0]
        assert "aapl_daily" in sql_arg
        assert "ORDER BY date ASC" in sql_arg
    mock_conn.close.assert_called_once()


def test_daily_bar_cache_get_last_n():
    """get_last_n returns correct slices for different dates."""
    rows = pl.DataFrame({
        "date": [date(2026, 1, d) for d in range(1, 31)],
        "open": [float(d) for d in range(1, 31)],
        "high": [float(d) for d in range(1, 31)],
        "low": [float(d) for d in range(1, 31)],
        "close": [float(d) for d in range(1, 31)],
        "volume": [1000] * 30,
        "vwap": [float(d) for d in range(1, 31)],
    })
    conn_fn, mock_conn, _ = _make_mock_connection(rows)

    with patch("rl_model.db._query", return_value=rows):
        cache = db.DailyBarCache("AAPL", date(2026, 1, 30), connection_fn=conn_fn)

    # Request last 5 bars up to Jan 10
    result = cache.get_last_n(date(2026, 1, 10), n=5)
    assert len(result) == 5
    assert result["date"].to_list() == [date(2026, 1, d) for d in range(6, 11)]

    # Request last 3 bars up to Jan 3 (only 3 available)
    result = cache.get_last_n(date(2026, 1, 3), n=5)
    assert len(result) == 3

    # String date input
    result = cache.get_last_n("2026-01-15", n=2)
    assert len(result) == 2
    assert result["date"].to_list() == [date(2026, 1, 14), date(2026, 1, 15)]


@patch("rl_model.db.get_intraday_data")
def test_get_date_window_with_cache(mock_intraday):
    """get_date_window uses the cache instead of querying when provided."""
    rows = pl.DataFrame({
        "date": [date(2026, 2, d) for d in range(1, 13)],
        "open": [100.0] * 12, "high": [101.0] * 12, "low": [99.0] * 12,
        "close": [100.5] * 12, "volume": [1000] * 12, "vwap": [100.2] * 12,
    })
    conn_fn, mock_conn, _ = _make_mock_connection(rows)

    with patch("rl_model.db._query", return_value=rows):
        cache = db.DailyBarCache("AAPL", date(2026, 2, 12), connection_fn=conn_fn)

    mock_intraday.return_value = pl.DataFrame({
        "timestamp": [datetime(2026, 2, 13, 9, 30)],
        "open": [260.0], "high": [261.0], "low": [259.5],
        "close": [260.5], "volume": [50000], "vwap": [260.2],
    })

    with patch("rl_model.db.get_last_n_days") as mock_last_n:
        result = db.get_date_window("AAPL", "2026-02-13", daily_cache=cache)
        mock_last_n.assert_not_called()

    assert isinstance(result, pl.DataFrame)
    assert "timestamp" in result.columns
    # 12 daily rows + 1 intraday row
    assert len(result) == 13


@patch("rl_model.db.get_intraday_data")
@patch("rl_model.db.get_last_n_days")
def test_get_date_window_without_cache_still_works(mock_last_n, mock_intraday):
    """Backward compatibility: get_date_window without cache queries DB directly."""
    mock_last_n.return_value = pl.DataFrame({
        "date": [date(2026, 2, 12)],
        "open": [259.0], "high": [261.0], "low": [258.0],
        "close": [260.0], "volume": [1100000], "vwap": [259.5],
    })
    mock_intraday.return_value = pl.DataFrame({
        "timestamp": [datetime(2026, 2, 13, 9, 30)],
        "open": [260.0], "high": [261.0], "low": [259.5],
        "close": [260.5], "volume": [50000], "vwap": [260.2],
    })

    result = db.get_date_window("AAPL", "2026-02-13")

    mock_last_n.assert_called_once()
    assert len(result) == 2


# ── get_next_trading_date ────────────────────────────────────────────


@patch("rl_model.db.get_connection")
def test_get_next_trading_date(mock_get_conn):
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (date(2026, 2, 14),)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_get_conn.return_value.__enter__ = lambda s: mock_conn
    mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = db.get_next_trading_date("AAPL", "2026-02-13")

    mock_cur.execute.assert_called_once()
    sql_arg = mock_cur.execute.call_args[0][0]
    assert "aapl_historical" in sql_arg
    assert result == date(2026, 2, 14)


@patch("rl_model.db.get_connection")
def test_get_next_trading_date_no_data(mock_get_conn):
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_get_conn.return_value.__enter__ = lambda s: mock_conn
    mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(ValueError, match="No trading date found"):
        db.get_next_trading_date("AAPL", "2026-02-13")

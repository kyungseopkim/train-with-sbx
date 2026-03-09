import logging
import os
from datetime import datetime, timedelta

import polars as pl
import pymysql
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()


def _query(sql, conn, params=None):
    """Execute a parameterized query and return a Polars DataFrame."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return pl.DataFrame()
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if not rows:
            return pl.DataFrame(schema=cols)
        return pl.DataFrame(rows, schema=cols, orient="row")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

_pool: PooledDB | None = None


def _get_pool() -> PooledDB:
    global _pool
    if _pool is None:
        _pool = PooledDB(
            pymysql,
            maxconnections=10,
            mincached=0,
            blocking=True,
            ping=2,  # Check connection status before use
            failures=(pymysql.err.OperationalError, pymysql.err.InternalError),
            **DB_CONFIG
        )
    return _pool


def get_connection():
    return _get_pool().connection()


def get_last_n_days(ticker, date, n=90):
    table = f"{ticker.lower()}_daily"
    sql = f"SELECT * FROM `{table}` WHERE date <= %s ORDER BY date DESC LIMIT %s"
    conn = get_connection()
    try:
        return _query(sql, conn, params=(date, n))
    finally:
        conn.close()


class DailyBarCache:
    """Bulk-loaded daily bars for a single ticker.

    Fetches all daily bars up to `max_date` in one query.
    Thread-safe after construction (read-only).
    """

    def __init__(self, ticker: str, max_date, connection_fn=None):
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d").date()
        table = f"{ticker.lower()}_daily"
        sql = f"SELECT * FROM `{table}` WHERE date <= %s ORDER BY date ASC"
        conn_fn = connection_fn or get_connection
        conn = conn_fn()
        try:
            self._df = _query(sql, conn, params=(max_date.isoformat(),))
        finally:
            conn.close()
        log.info(
            "DailyBarCache: loaded %d rows for %s up to %s",
            len(self._df), ticker, max_date,
        )

    def get_last_n(self, date, n=90) -> pl.DataFrame:
        """Return the last *n* daily bars on or before *date*."""
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        filtered = self._df.filter(pl.col("date") <= date)
        return filtered.tail(n)


class IntradayCache:
    """Bulk-loaded intraday bars for a single ticker.

    Fetches all intraday bars between `min_date` and `max_date` in one query.
    Thread-safe after construction (read-only).
    """

    def __init__(self, ticker: str, min_date, max_date, connection_fn=None):
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
        if isinstance(max_date, str):
            max_date = datetime.strptime(max_date, "%Y-%m-%d").date()
        table = f"{ticker.lower()}_historical"
        next_day = (max_date + timedelta(days=1)).isoformat()
        sql = (
            f"SELECT * FROM `{table}` "
            f"WHERE timestamp >= %s AND timestamp < %s ORDER BY timestamp"
        )
        conn_fn = connection_fn or get_connection
        conn = conn_fn()
        try:
            self._df = _query(sql, conn, params=(min_date.isoformat(), next_day))
        finally:
            conn.close()
        # Pre-compute date column for fast filtering
        if len(self._df) > 0:
            self._df = self._df.with_columns(
                pl.col("timestamp").dt.date().alias("_date")
            )
        log.info(
            "IntradayCache: loaded %d rows for %s (%s to %s)",
            len(self._df), ticker, min_date, max_date,
        )

    def get_date(self, date) -> pl.DataFrame:
        """Return all intraday bars for a single date."""
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if len(self._df) == 0:
            return self._df
        return self._df.filter(pl.col("_date") == date).drop("_date")


def get_intraday_data(ticker, date):
    table = f"{ticker.lower()}_historical"
    sql = f"SELECT * FROM `{table}` WHERE timestamp >= %s AND timestamp < %s ORDER BY timestamp"
    if isinstance(date, str):
        d = datetime.strptime(date, "%Y-%m-%d").date()
    else:
        d = date
    next_day = (d + timedelta(days=1)).isoformat()
    conn = get_connection()
    try:
        return _query(sql, conn, params=(d.isoformat(), next_day))
    finally:
        conn.close()


def get_trading_dates(ticker, start_date, end_date):
    table = f"{ticker.lower()}_historical"
    sql = f"SELECT DISTINCT DATE(timestamp) AS date FROM `{table}` WHERE DATE(timestamp) BETWEEN %s AND %s ORDER BY date"
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (start_date.isoformat(), end_date.isoformat()))
            rows = cur.fetchall()
            return [row[0] for row in rows]


def get_next_trading_date(ticker, date):
    """Return the next distinct trading date after *date* for *ticker*.

    Raises ValueError if no subsequent date exists.
    """
    table = f"{ticker.lower()}_historical"
    sql = (
        f"SELECT DISTINCT DATE(timestamp) AS date FROM `{table}` "
        f"WHERE DATE(timestamp) > %s ORDER BY date LIMIT 1"
    )
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (date.isoformat(),))
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"No trading date found after {date} for {ticker}"
                )
            return row[0]


SHARED_COLS = ["open", "high", "low", "close", "volume", "vwap"]


def get_date_window(ticker, date, include_daily=True, daily_cache=None, intraday_cache=None):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()

    if intraday_cache is not None:
        intraday = intraday_cache.get_date(date)
    else:
        intraday = get_intraday_data(ticker, date.isoformat())
    intraday_empty = intraday.is_empty() or "timestamp" not in intraday.columns

    if not intraday_empty:
        intraday_unified = intraday.select(
            pl.col("timestamp"),
            *[pl.col(c) for c in SHARED_COLS],
        )

    if not include_daily:
        if intraday_empty:
            return pl.DataFrame(schema={"timestamp": pl.Datetime, **{c: pl.Float64 for c in SHARED_COLS}})
        return intraday_unified.sort("timestamp")

    prev_date = date - timedelta(days=1)
    if daily_cache is not None:
        daily = daily_cache.get_last_n(prev_date)
    else:
        daily = get_last_n_days(ticker, prev_date.isoformat())
    daily_unified = daily.select(
        pl.col("date").cast(pl.Datetime).alias("timestamp"),
        *[pl.col(c) for c in SHARED_COLS],
    )
    if intraday_empty:
        return daily_unified.sort("timestamp")
    return pl.concat([daily_unified, intraday_unified]).sort("timestamp")

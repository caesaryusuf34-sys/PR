from __future__ import annotations
import pandas as pd

def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s")

def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def iso(x) -> str:
    return ts(x).strftime("%Y-%m-%dT%H:%M:%SZ")


def ns(x):
    """UTC nanoseconds (int64) for a tz-aware Timestamp or Series - safe for numpy comparisons."""
    if isinstance(x, pd.Series):
        return pd.to_datetime(x, utc=True).dt.as_unit("ns").values.view("int64")
    return ts(x).as_unit("ns").value

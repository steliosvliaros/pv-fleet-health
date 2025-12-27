import pandas as pd
import numpy as np

def safe_quantile(x: pd.Series, q: float) -> float:
    x2 = x.dropna()
    if len(x2) == 0:
        return np.nan
    return float(x2.quantile(q))

def mad(x: pd.Series) -> float:
    x2 = x.dropna().astype(float)
    if len(x2) == 0:
        return np.nan
    med = float(np.median(x2))
    return float(np.median(np.abs(x2 - med)))

def ensure_tz_aware(ts: pd.Series, tz: str) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    return ts.dt.tz_convert(tz)

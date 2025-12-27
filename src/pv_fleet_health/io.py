import os
from typing import Optional, Tuple
import pandas as pd
from .config import Config
from .utils import ensure_tz_aware

def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def parse_timestamp(df: pd.DataFrame, col: str, tz: str, fmt: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if fmt:
        ts = pd.to_datetime(out[col], format=fmt, errors="coerce")
    else:
        ts = pd.to_datetime(out[col], errors="coerce")
    out[col] = ensure_tz_aware(ts, tz)
    return out

def load_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    scada = load_table(cfg.scada_path)
    scada = parse_timestamp(scada, cfg.timestamp_col, cfg.default_timezone, cfg.timestamp_format)

    events = load_table(cfg.events_path)
    # parse event time cols if present
    for c in ["Start Date", "End Date"]:
        if c in events.columns:
            events[c] = ensure_tz_aware(events[c], cfg.default_timezone)

    meta = load_table(cfg.metadata_path) if cfg.metadata_path and os.path.exists(cfg.metadata_path) else None
    return scada, events, meta

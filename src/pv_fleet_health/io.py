import os
from pathlib import Path

import pandas as pd

from .config import Config
from .utils import ensure_tz_aware


def load_table(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    path_str = str(path).lower()
    if path_str.endswith(".parquet"):
        return pd.read_parquet(str(path))
    # Try to detect delimiter for CSV files
    if path_str.endswith(".csv"):
        # Read first line to detect delimiter
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","
        return pd.read_csv(str(path), sep=delimiter)
    return pd.read_csv(str(path))


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def parse_timestamp(df: pd.DataFrame, col: str, tz: str, fmt: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if fmt:
        ts = pd.to_datetime(out[col], format=fmt, errors="coerce")
    else:
        ts = pd.to_datetime(out[col], errors="coerce")
    out[col] = ensure_tz_aware(ts, tz)
    return out


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    scada = load_table(cfg.scada_path)
    scada = parse_timestamp(scada, cfg.timestamp_col, cfg.default_timezone, cfg.timestamp_format)

    events = load_table(cfg.events_path)
    # parse event time cols if present
    for c in ["Start Date", "End Date"]:
        if c in events.columns:
            events[c] = ensure_tz_aware(events[c], cfg.default_timezone)

    meta = (
        load_table(cfg.metadata_path)
        if cfg.metadata_path and os.path.exists(cfg.metadata_path)
        else None
    )
    return scada, events, meta

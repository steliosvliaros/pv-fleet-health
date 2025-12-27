from typing import Dict
import pandas as pd
import numpy as np
from .config import Config

def compute_time_index_audit(scada_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plant, g in scada_long.groupby("plant_name"):
        ts = g["ts"].dropna().sort_values()
        if ts.empty:
            rows.append({"plant_name": plant, "min_ts": None, "max_ts": None, "dup_frac": None, "median_dt_min": None, "n": 0})
            continue

        dup_frac = float(ts.duplicated().mean())
        dt = ts.drop_duplicates().diff().dropna()
        dt_min = dt.dt.total_seconds() / 60.0
        median_dt = float(np.nanmedian(dt_min)) if len(dt_min) else np.nan

        rows.append({
            "plant_name": plant,
            "min_ts": ts.min(),
            "max_ts": ts.max(),
            "dup_frac": dup_frac,
            "median_dt_min": median_dt,
            "n": int(len(ts)),
        })
    return pd.DataFrame(rows)

def resample_signals(scada_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Resample per plant/component/signal to cfg.standard_freq.
    Policy:
      - counters: last
      - interval energy: sum
      - others: mean
    """
    df = scada_long.dropna(subset=["ts"]).copy()
    df = df.sort_values("ts")

    def signal_type(sig: str) -> str:
        if str(sig).startswith("energy_kwh_counter"):
            return "counter"
        if sig == "energy_kwh_interval":
            return "energy_interval"
        return "instant_or_avg"

    df["signal_type"] = df["canonical_signal"].map(signal_type)

    key_cols = ["plant_name", "component_type", "component_id", "canonical_signal", "unit", "signal_type"]

    out = []
    for keys, g in df.groupby(key_cols, sort=False):
        g2 = g.set_index("ts")[["value"]].sort_index()
        stype = keys[-1]
        if stype == "counter":
            r = g2.resample(cfg.standard_freq).last()
        elif stype == "energy_interval":
            r = g2.resample(cfg.standard_freq).sum(min_count=1)
        else:
            r = g2.resample(cfg.standard_freq).mean()

        r = r.reset_index().rename(columns={"value": "value_rs"})
        rec = dict(zip(key_cols, keys))
        for k, v in rec.items():
            r[k] = v
        out.append(r)

    rs = pd.concat(out, ignore_index=True)
    return rs

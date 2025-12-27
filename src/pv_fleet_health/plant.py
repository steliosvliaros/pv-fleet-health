from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from .config import Config

def missing_data_policy_apply(s: pd.Series, cfg: Config, canonical_signal: str) -> pd.Series:
    s2 = s.copy()
    if canonical_signal in cfg.allow_interp_signals:
        # convert max gap minutes -> max number of points at standard freq
        max_gap_points = int(pd.Timedelta(minutes=cfg.max_interp_gap_minutes) / pd.Timedelta(cfg.standard_freq))
        s2 = s2.interpolate(limit=max_gap_points, limit_direction="both")
    if canonical_signal in cfg.allow_ffill_signals:
        s2 = s2.ffill()
    return s2

def counter_to_interval(counter: pd.Series, cfg: Config) -> pd.Series:
    c = counter.sort_index()
    d = c.diff()
    d[d < cfg.counter_reset_negative_kwh_threshold] = np.nan
    return d

def choose_power_level(scada_rs: pd.DataFrame, cfg: Config, plant: str, poa: Optional[pd.Series]) -> Tuple[Optional[str], str]:
    sub = scada_rs[(scada_rs["plant_name"] == plant) & (scada_rs["canonical_signal"] == "ac_power_kw")]
    if sub.empty:
        return None, "No ac_power_kw"

    def completeness(level: str) -> float:
        g = sub[sub["component_type"] == level]
        if g.empty:
            return -1.0
        s = g.groupby("ts")["value_rs"].sum(min_count=1)
        if poa is not None:
            s = s.reindex(poa.index)
            daylight = poa >= cfg.daylight_poa_threshold_wm2
            return float(s.loc[daylight].notna().mean()) if daylight.any() else float(s.notna().mean())
        return float(s.notna().mean())

    c_array = completeness("array")
    c_group = completeness("array_group")

    if c_array < 0 and c_group < 0:
        return None, "No array or array_group power"
    if c_array >= c_group:
        return "array", f"array completeness {c_array:.3f} >= group {c_group:.3f}"
    return "array_group", f"group completeness {c_group:.3f} > array {c_array:.3f}"

def build_plant_series(
    scada_rs: pd.DataFrame,
    cfg: Config,
    plant: str,
    poa: Optional[pd.Series],
    tmod: Optional[pd.Series],
    aggregation_level: Optional[str],
) -> pd.DataFrame:
    idx = None
    for s in [poa, tmod]:
        if s is not None:
            idx = s.index if idx is None else idx.union(s.index)

    # power series
    p = None
    if aggregation_level is not None:
        psub = scada_rs[
            (scada_rs["plant_name"] == plant) &
            (scada_rs["canonical_signal"] == "ac_power_kw") &
            (scada_rs["component_type"] == aggregation_level)
        ]
        if not psub.empty:
            p = psub.groupby("ts")["value_rs"].sum(min_count=1).sort_index()
            idx = p.index if idx is None else idx.union(p.index)

    # energy interval preferred
    e_int = None
    if aggregation_level is not None:
        esub = scada_rs[
            (scada_rs["plant_name"] == plant) &
            (scada_rs["canonical_signal"] == "energy_kwh_interval") &
            (scada_rs["component_type"] == aggregation_level)
        ]
        if not esub.empty:
            e_int = esub.groupby("ts")["value_rs"].sum(min_count=1).sort_index()
            idx = e_int.index if idx is None else idx.union(e_int.index)

    # counters fallback
    if e_int is None and aggregation_level is not None:
        csub = scada_rs[
            (scada_rs["plant_name"] == plant) &
            (scada_rs["canonical_signal"].str.startswith("energy_kwh_counter")) &
            (scada_rs["component_type"] == aggregation_level)
        ]
        if not csub.empty:
            csum = csub.groupby("ts")["value_rs"].sum(min_count=1).sort_index()
            e_int = counter_to_interval(csum, cfg)
            idx = e_int.index if idx is None else idx.union(e_int.index)

    if idx is None:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(idx).sort_values().unique()
    df = pd.DataFrame(index=idx)
    df["poa_wm2"] = poa.reindex(idx) if poa is not None else np.nan
    df["tmod_c"] = tmod.reindex(idx) if tmod is not None else np.nan
    df["p_ac_kw"] = p.reindex(idx) if p is not None else np.nan
    df["e_kwh"] = e_int.reindex(idx) if e_int is not None else np.nan

    # apply missing policy to met only
    df["poa_wm2"] = missing_data_policy_apply(df["poa_wm2"], cfg, "poa_irradiance_wm2")
    df["tmod_c"] = missing_data_policy_apply(df["tmod_c"], cfg, "tmod_c")

    # plausibility clip (leave NaN)
    df.loc[(df["poa_wm2"] < cfg.min_valid_poa_wm2) | (df["poa_wm2"] > cfg.max_valid_poa_wm2), "poa_wm2"] = np.nan
    df.loc[(df["tmod_c"] < cfg.min_valid_tmod_c) | (df["tmod_c"] > cfg.max_valid_tmod_c), "tmod_c"] = np.nan

    return df

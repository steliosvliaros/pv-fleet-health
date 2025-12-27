from typing import Optional, Dict
import numpy as np
import pandas as pd
from .config import Config

def get_dc_kwp(metadata: Optional[pd.DataFrame], plant: str) -> Optional[float]:
    if metadata is None or "plant_name" not in metadata.columns:
        return None
    m = metadata[metadata["plant_name"] == plant]
    if len(m) != 1:
        return None
    for col in ["DC_kWp", "dc_kwp", "dc_kWp", "dc_kw"]:
        if col in m.columns and pd.notna(m.iloc[0][col]):
            return float(m.iloc[0][col])
    return None

def compute_kpis(df_labeled: pd.DataFrame, cfg: Config, dc_kwp: Optional[float]) -> Dict[str, pd.DataFrame]:
    df = df_labeled.copy()
    dt_h = pd.Timedelta(cfg.standard_freq).total_seconds() / 3600.0

    intrinsic = ~df["event_label"].isin(["curtailment", "grid_outage"])
    daylight_kpi = df["poa_wm2"] >= cfg.poa_for_kpi_min_wm2

    # derived energy if not available
    if "e_kwh" not in df.columns or df["e_kwh"].isna().all():
        df["e_kwh"] = df["p_ac_kw"] * dt_h

    df["e_kwh_intrinsic"] = df["e_kwh"].where(intrinsic, np.nan)

    daily = pd.DataFrame(index=df.resample("D").sum(numeric_only=True).index)
    daily["energy_kwh"] = df["e_kwh"].resample("D").sum(min_count=1)
    daily["energy_kwh_intrinsic"] = df["e_kwh_intrinsic"].resample("D").sum(min_count=1)

    if dc_kwp and dc_kwp > 0:
        daily["specific_yield_kwh_per_kwp"] = daily["energy_kwh_intrinsic"] / dc_kwp

    # simple performance index: (P/POA)/median(P/POA) on intrinsic daylight points
    mask = intrinsic & daylight_kpi & df["p_ac_kw"].notna() & df["poa_wm2"].notna()
    df["perf_index"] = np.nan
    if mask.sum() > 100:
        raw = (df.loc[mask, "p_ac_kw"] / df.loc[mask, "poa_wm2"]).replace([np.inf, -np.inf], np.nan)
        scale = np.nanmedian(raw)
        if np.isfinite(scale) and scale > 0:
            df.loc[mask, "perf_index"] = (df.loc[mask, "p_ac_kw"] / df.loc[mask, "poa_wm2"]) / scale

    daily["perf_index_median"] = df["perf_index"].resample("D").median()

    monthly = daily.resample("MS").agg({
        "energy_kwh": "sum",
        "energy_kwh_intrinsic": "sum",
        "perf_index_median": "median",
        "specific_yield_kwh_per_kwp": "mean" if "specific_yield_kwh_per_kwp" in daily.columns else "sum",
    })

    return {"ts": df, "daily": daily, "monthly": monthly}

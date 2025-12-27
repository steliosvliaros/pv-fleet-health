from typing import Dict, List
import numpy as np
import pandas as pd
from .config import Config

def stuck_sensor_fraction(series: pd.Series, window: int = 12, tol: float = 1e-6) -> float:
    s = series.dropna()
    if len(s) < window * 2:
        return np.nan
    roll_std = s.rolling(window).std()
    return float((roll_std <= tol).mean())

def dq_score_plant(scada_rs: pd.DataFrame, cfg: Config, plant: str) -> Dict:
    sub = scada_rs[scada_rs["plant_name"] == plant].copy()

    poa = sub[sub["canonical_signal"] == "poa_irradiance_wm2"].groupby("ts")["value_rs"].median()
    tmod = sub[sub["canonical_signal"] == "tmod_c"].groupby("ts")["value_rs"].median()
    pwr = sub[sub["canonical_signal"] == "ac_power_kw"].groupby("ts")["value_rs"].sum(min_count=1)
    pf = sub[sub["canonical_signal"] == "pf"].groupby("ts")["value_rs"].median()

    base = pd.DataFrame({"poa": poa, "tmod": tmod, "p_kw": pwr, "pf": pf})

    daylight = base["poa"] >= cfg.daylight_poa_threshold_wm2
    if daylight.sum() == 0:
        daylight = base["poa"].notna()

    comp = {
        "poa_missing_frac_day": float(base.loc[daylight, "poa"].isna().mean()),
        "tmod_missing_frac_day": float(base.loc[daylight, "tmod"].isna().mean()),
        "p_missing_frac_day": float(base.loc[daylight, "p_kw"].isna().mean()),
    }
    plaus = {
        "poa_oob_frac": float(((base["poa"] < cfg.min_valid_poa_wm2) | (base["poa"] > cfg.max_valid_poa_wm2)).mean()),
        "tmod_oob_frac": float(((base["tmod"] < cfg.min_valid_tmod_c) | (base["tmod"] > cfg.max_valid_tmod_c)).mean()),
        "pf_oob_frac": float((base["pf"].abs() > cfg.max_pf_abs).mean()) if base["pf"].notna().any() else np.nan,
    }
    stuck = {
        "poa_stuck_frac": stuck_sensor_fraction(base["poa"]),
        "tmod_stuck_frac": stuck_sensor_fraction(base["tmod"]),
    }

    # counter resets (plant-level heuristic)
    counters = sub[sub["canonical_signal"].str.startswith("energy_kwh_counter")]
    reset_frac = np.nan
    if not counters.empty:
        csum = counters.groupby("ts")["value_rs"].sum(min_count=1).dropna().sort_index()
        d = csum.diff()
        reset_frac = float((d < cfg.counter_reset_negative_kwh_threshold).mean())

    def frac_to_score(x: float) -> float:
        if pd.isna(x):
            return 0.7
        return float(max(0.0, 1.0 - min(1.0, x * 5)))

    score_components = {
        "completeness_score": float(np.mean([frac_to_score(comp["poa_missing_frac_day"]),
                                             frac_to_score(comp["tmod_missing_frac_day"]),
                                             frac_to_score(comp["p_missing_frac_day"])])),
        "plausibility_score": float(np.mean([frac_to_score(plaus["poa_oob_frac"]),
                                             frac_to_score(plaus["tmod_oob_frac"]),
                                             frac_to_score(plaus["pf_oob_frac"])])),
        "stuck_score": float(np.mean([frac_to_score(stuck["poa_stuck_frac"]),
                                      frac_to_score(stuck["tmod_stuck_frac"])])),
        "counter_score": frac_to_score(reset_frac),
    }
    dq_score = float(np.mean(list(score_components.values())))
    confidence = "High" if dq_score >= 0.85 else ("Medium" if dq_score >= 0.70 else "Low")

    return {
        "plant_name": plant,
        **comp, **plaus, **stuck,
        "counter_reset_frac": reset_frac,
        **score_components,
        "dq_score": dq_score,
        "monitoring_confidence": confidence,
    }

def dq_report_fleet(scada_rs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    plants = sorted(scada_rs["plant_name"].dropna().unique().tolist())
    return pd.DataFrame([dq_score_plant(scada_rs, cfg, p) for p in plants])

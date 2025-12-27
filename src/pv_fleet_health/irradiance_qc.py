from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from .config import Config
from .dq import stuck_sensor_fraction

try:
    import pvlib
    PVLIB_AVAILABLE = True
except Exception:
    PVLIB_AVAILABLE = False

@dataclass
class IrradianceSensorScore:
    plant_name: str
    component_type: str
    component_id: str
    completeness_day: float
    stuck_frac: float
    clearsky_violation_frac: float
    score: float

def _get_lat_lon(meta_row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    lat = None
    lon = None
    for k in ["lat", "latitude"]:
        if k in meta_row.index and pd.notna(meta_row[k]):
            lat = float(meta_row[k])
    for k in ["lon", "longitude", "lng"]:
        if k in meta_row.index and pd.notna(meta_row[k]):
            lon = float(meta_row[k])
    return lat, lon

def clearsky_envelope_violations(poa: pd.Series, cfg: Config, meta_row: Optional[pd.Series] = None) -> pd.Series:
    s = poa.dropna()
    if s.empty:
        return pd.Series(dtype=bool)

    if PVLIB_AVAILABLE and meta_row is not None:
        lat, lon = _get_lat_lon(meta_row)
        if lat is not None and lon is not None:
            loc = pvlib.location.Location(lat, lon, tz=poa.index.tz)
            cs = loc.get_clearsky(s.index)
            envelope = cs["ghi"] * 1.2  # envelope factor
            viol = s > envelope
            return viol.reindex(poa.index).fillna(False)

    # empirical fallback: daily high-quantile envelope
    df = s.to_frame("poa")
    df["date"] = df.index.date
    env = df.groupby("date")["poa"].quantile(cfg.clearsky_qc_quantile)
    df = df.join(env.rename("env"), on="date")
    viol = df["poa"] > (df["env"] * 1.15)
    return viol.reindex(poa.index).fillna(False)

def select_best_irradiance_sensor(scada_rs: pd.DataFrame, cfg: Config, plant: str, metadata: Optional[pd.DataFrame]) -> Tuple[Optional[Dict], pd.DataFrame]:
    sub = scada_rs[(scada_rs["plant_name"] == plant) & (scada_rs["canonical_signal"] == "poa_irradiance_wm2")].copy()
    if sub.empty:
        return None, pd.DataFrame()

    meta_row = None
    if metadata is not None and "plant_name" in metadata.columns:
        mm = metadata[metadata["plant_name"] == plant]
        if len(mm) == 1:
            meta_row = mm.iloc[0]

    scores = []
    for (ctype, cid), g in sub.groupby(["component_type", "component_id"]):
        s = g.groupby("ts")["value_rs"].median().sort_index()
        daylight = s >= cfg.daylight_poa_threshold_wm2
        completeness = float(s.loc[daylight].notna().mean()) if daylight.any() else float(s.notna().mean())
        stuck = stuck_sensor_fraction(s)
        viol = clearsky_envelope_violations(s, cfg, meta_row=meta_row)
        viol_frac = float(viol.mean()) if len(viol) else np.nan

        # heuristic score
        def good_frac(x):
            if pd.isna(x):
                return 0.7
            return max(0.0, 1.0 - min(1.0, x))

        score = 0.45 * completeness + 0.25 * good_frac(stuck) + 0.30 * good_frac(viol_frac)
        scores.append(IrradianceSensorScore(plant, str(ctype), str(cid), completeness, stuck, viol_frac, float(score)))

    tbl = pd.DataFrame([s.__dict__ for s in scores]).sort_values("score", ascending=False)
    best = tbl.iloc[0].to_dict() if len(tbl) else None
    return best, tbl

def get_best_poa_series(scada_rs: pd.DataFrame, cfg: Config, plant: str, best_sensor: Optional[Dict]) -> Optional[pd.Series]:
    sub = scada_rs[(scada_rs["plant_name"] == plant) & (scada_rs["canonical_signal"] == "poa_irradiance_wm2")]
    if sub.empty:
        return None

    if not best_sensor:
        return sub.groupby("ts")["value_rs"].median().sort_index()

    ctype = best_sensor["component_type"]
    cid = str(best_sensor["component_id"])
    ssub = sub[(sub["component_type"] == ctype) & (sub["component_id"].astype(str) == cid)]
    if ssub.empty:
        return sub.groupby("ts")["value_rs"].median().sort_index()
    return ssub.groupby("ts")["value_rs"].median().sort_index()

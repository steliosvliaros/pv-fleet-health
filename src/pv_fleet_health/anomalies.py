import numpy as np
import pandas as pd
from .config import Config
from .utils import mad

def detect_anomalies(df: pd.DataFrame, cfg: Config, p_expected: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["p_expected_kw"] = p_expected
    out["residual_kw"] = out["p_ac_kw"] - out["p_expected_kw"]

    intrinsic = ~out["event_label"].isin(["curtailment", "grid_outage"])
    daylight = out["poa_wm2"] >= cfg.poa_for_kpi_min_wm2
    mask = intrinsic & daylight & out["residual_kw"].notna()

    r = out.loc[mask, "residual_kw"]
    med = float(np.nanmedian(r)) if len(r) else np.nan
    m = mad(r) if len(r) else np.nan
    denom = (1.4826 * m + 1e-9) if np.isfinite(m) else np.nan

    out["resid_z"] = (out["residual_kw"] - med) / denom if np.isfinite(denom) else np.nan

    window = int(pd.Timedelta(days=cfg.rolling_window_days) / pd.Timedelta(cfg.standard_freq))
    out["resid_roll_med"] = out["residual_kw"].rolling(window, min_periods=max(10, window // 5)).median()

    out["anomaly_point"] = False
    out.loc[mask & (out["resid_z"].abs() >= cfg.residual_z_threshold), "anomaly_point"] = True
    return out

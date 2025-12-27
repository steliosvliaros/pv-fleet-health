import pandas as pd
from .config import Config

BUCKETS = ["curtailment", "grid_outage", "planned_maintenance", "inverter_fault", "comms_data", "other_unknown"]

def compute_losses(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    dt_h = pd.Timedelta(cfg.standard_freq).total_seconds() / 3600.0

    out["e_exp_kwh"] = out["p_expected_kw"] * dt_h
    out["e_act_kwh"] = out["p_ac_kw"] * dt_h

    daylight = out["poa_wm2"] >= cfg.poa_for_kpi_min_wm2
    out.loc[~daylight, ["e_exp_kwh", "e_act_kwh"]] = pd.NA

    out["loss_kwh"] = (out["e_exp_kwh"] - out["e_act_kwh"])

    for b in BUCKETS:
        out[f"loss_{b}"] = out["loss_kwh"].where(out["event_label"] == b, 0.0)

    intrinsic = ~out["event_label"].isin(["curtailment", "grid_outage"])
    out["loss_unexplained"] = out["loss_kwh"].where(intrinsic & (out["loss_kwh"] > 0), 0.0)

    daily = out.resample("D").sum(numeric_only=True)
    keep = ["loss_kwh", "loss_unexplained"] + [f"loss_{b}" for b in BUCKETS]
    return daily[keep].copy()

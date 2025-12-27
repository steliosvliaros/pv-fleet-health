from typing import Dict, Optional
import numpy as np
import pandas as pd
from .kpi import get_dc_kwp

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def build_fleet_scorecard(health_cards: Dict[str, Dict], dq_report: pd.DataFrame, plr_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plant, card in health_cards.items():
        if not card.get("ok"):
            continue

        monthly = card["monthly"]
        last12 = monthly.tail(12)
        perf12 = float(last12["perf_index_median"].median()) if "perf_index_median" in last12.columns else np.nan

        losses_daily = card["losses_daily"]
        last90 = losses_daily.tail(90)
        total_loss90 = float(last90["loss_kwh"].sum()) if "loss_kwh" in last90.columns else np.nan
        unexp90 = float(last90["loss_unexplained"].sum()) if "loss_unexplained" in last90.columns else np.nan
        unexp_frac90 = (unexp90 / total_loss90) if (total_loss90 and total_loss90 > 0) else np.nan

        dq = dq_report[dq_report["plant_name"] == plant]
        dq_score = float(dq["dq_score"].iloc[0]) if len(dq) else np.nan
        conf = dq["monitoring_confidence"].iloc[0] if len(dq) else None

        plr = plr_table[plr_table["plant_name"] == plant]
        plr_val = float(plr["plr_pct_per_year"].iloc[0]) if len(plr) and plr["ok"].iloc[0] else np.nan

        rows.append({
            "plant_name": plant,
            "perf_index_12m_median": perf12,
            "dq_score": dq_score,
            "monitoring_confidence": conf,
            "unexplained_loss_frac_90d": unexp_frac90,
            "plr_pct_per_year": plr_val,
        })

    score = pd.DataFrame(rows)
    score["perf_rank_pct"] = score["perf_index_12m_median"].rank(pct=True)
    score["dq_rank_pct"] = score["dq_score"].rank(pct=True)
    score["unexplained_loss_rank_pct"] = (1.0 - score["unexplained_loss_frac_90d"].rank(pct=True))
    score["plr_rank_pct"] = score["plr_pct_per_year"].rank(pct=True, ascending=False)
    return score

def fleet_clustering(scorecard: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = scorecard.copy()
    feats = ["perf_index_12m_median", "dq_score", "unexplained_loss_frac_90d", "plr_pct_per_year"]
    X = df[feats].copy().fillna(df[feats].median(numeric_only=True))
    if not SKLEARN_AVAILABLE or len(df) < k:
        df["cluster"] = 0
        return df
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X.values)
    return df

def build_action_plan(scorecard: pd.DataFrame) -> pd.DataFrame:
    df = scorecard.copy()
    df["priority_score"] = (
        (1.0 - df["perf_rank_pct"].fillna(0.5)) * 0.5 +
        (1.0 - df["unexplained_loss_rank_pct"].fillna(0.5)) * 0.3 +
        (df["dq_rank_pct"].fillna(0.5)) * 0.2
    )

    def rec(r):
        actions = []
        if r.get("dq_score", 1.0) < 0.7:
            actions.append("Fix monitoring/data quality first (POA/Tmod/power completeness & QC).")
        if r.get("perf_index_12m_median", 1.0) < np.nanmedian(df["perf_index_12m_median"]):
            actions.append("Investigate intrinsic underperformance (soiling, derates, equipment health).")
        if pd.notna(r.get("plr_pct_per_year")) and r["plr_pct_per_year"] < -1.0:
            actions.append("Degradation deep dive (compare inverters/arrays, targeted inspections).")
        if pd.notna(r.get("unexplained_loss_frac_90d")) and r["unexplained_loss_frac_90d"] > 0.5:
            actions.append("Improve event tagging & diagnose hidden losses (intermittent faults/controls).")
        return " ".join(actions) if actions else "Monitor."

    df["recommended_actions"] = df.apply(rec, axis=1)
    return df.sort_values("priority_score", ascending=False)[[
        "plant_name", "priority_score", "monitoring_confidence",
        "perf_index_12m_median", "unexplained_loss_frac_90d", "plr_pct_per_year", "recommended_actions"
    ]]

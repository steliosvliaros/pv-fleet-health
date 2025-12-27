from typing import Optional, List
import re
import pandas as pd
from .config import Config

EVENT_CATEGORY_RULES = [
    (r"(?i)curtail|setpoint|limit|active power limit", "curtailment"),
    (r"(?i)grid|utility|voltage ride|frequency ride|disconnection", "grid_outage"),
    (r"(?i)maint|service|inspection|planned", "planned_maintenance"),
    (r"(?i)inverter|fault|trip|error|alarm", "inverter_fault"),
    (r"(?i)comms|communication|no data|offline|scada", "comms_data"),
]

CATEGORY_PRIORITY = ["curtailment", "grid_outage", "planned_maintenance", "inverter_fault", "comms_data", "other_unknown"]

def categorize_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def cat(desc: str) -> str:
        s = "" if pd.isna(desc) else str(desc)
        for pat, c in EVENT_CATEGORY_RULES:
            if re.search(pat, s):
                return c
        return "other_unknown"
    if "Description" in out.columns:
        out["category"] = out["Description"].apply(cat)
    else:
        out["category"] = "other_unknown"
    return out

def infer_plant_from_source(source: str, plants: List[str]) -> Optional[str]:
    if pd.isna(source):
        return None
    s = str(source)
    m = re.search(r"\[(.+?)\]", s)
    if m:
        return m.group(1).strip()
    for p in plants:
        if p and p in s:
            return p
    return None

def normalize_events(events_raw: pd.DataFrame, cfg: Config, plants: List[str]) -> pd.DataFrame:
    df = events_raw.copy()

    df = df.rename(columns={"Start Date": "start_ts", "End Date": "end_ts"})
    # duration -> end
    if "Duration" in df.columns:
        def parse_duration(x):
            if pd.isna(x):
                return pd.NaT
            if isinstance(x, (int, float)):
                return pd.to_timedelta(float(x), unit="s")
            try:
                return pd.to_timedelta(str(x))
            except Exception:
                return pd.NaT
        dur = df["Duration"].apply(parse_duration)
        missing_end = df["end_ts"].isna() & df["start_ts"].notna() & dur.notna()
        df.loc[missing_end, "end_ts"] = df.loc[missing_end, "start_ts"] + dur[missing_end]

    df = categorize_events(df)

    if "Source" in df.columns:
        df["plant_name"] = df["Source"].apply(lambda x: infer_plant_from_source(x, plants))
    else:
        df["plant_name"] = None

    df = df[df["plant_name"].notna()].copy()
    df = df.dropna(subset=["start_ts", "end_ts"])
    df = df[df["end_ts"] >= df["start_ts"]].copy()

    keep = ["plant_name", "start_ts", "end_ts", "Severity", "category", "Description", "State", "Ack", "Source"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

def merge_overlapping_events(events_norm: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (plant, cat), g in events_norm.groupby(["plant_name", "category"]):
        g = g.sort_values("start_ts")
        cur_s, cur_e = None, None
        for _, r in g.iterrows():
            s, e = r["start_ts"], r["end_ts"]
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                out_rows.append({"plant_name": plant, "category": cat, "start_ts": cur_s, "end_ts": cur_e})
                cur_s, cur_e = s, e
        if cur_s is not None:
            out_rows.append({"plant_name": plant, "category": cat, "start_ts": cur_s, "end_ts": cur_e})
    return pd.DataFrame(out_rows)

def join_events_to_timeseries(plant_df: pd.DataFrame, events_norm: pd.DataFrame, plant: str) -> pd.DataFrame:
    df = plant_df.copy()
    df["event_label"] = "none"
    ev = events_norm[events_norm["plant_name"] == plant].copy()
    if ev.empty:
        return df

    for cat in CATEGORY_PRIORITY:
        df[f"ev_{cat}"] = False

    for _, r in ev.iterrows():
        cat = r["category"]
        col = f"ev_{cat}"
        if col not in df.columns:
            continue
        mask = (df.index >= r["start_ts"]) & (df.index <= r["end_ts"])
        df.loc[mask, col] = True

    def primary(row):
        for cat in CATEGORY_PRIORITY:
            if row.get(f"ev_{cat}", False):
                return cat
        return "none"

    df["event_label"] = df.apply(primary, axis=1)
    return df

import re
from typing import List, Tuple
import pandas as pd
from .mapping import map_raw_signal_to_canonical

HEADER_RE = re.compile(r"^\[(?P<plant>.+?)\]\s+(?P<rest>.+?)$")
UNIT_RE = re.compile(r"\((?P<unit>[^)]+)\)\s*$")

def parse_component(rest: str) -> Tuple[str, str, str]:
    s = rest.strip()
    um = UNIT_RE.search(s)
    s_nounit = s[: um.start()].strip() if um else s

    m = re.match(r"^(Array Group)\s+(?P<gid>.+?)\s+(?P<sig>.+)$", s_nounit, flags=re.IGNORECASE)
    if m:
        return "array_group", m.group("gid").strip(), m.group("sig").strip()

    m = re.match(r"^(Array)\s+(?P<n>\d+)\s+(?P<sig>.+)$", s_nounit, flags=re.IGNORECASE)
    if m:
        return "array", m.group("n").strip(), m.group("sig").strip()

    m = re.match(r"^(Inverter)\s+(?P<n>\d+)\s+(?P<sig>.+)$", s_nounit, flags=re.IGNORECASE)
    if m:
        return "inverter", m.group("n").strip(), m.group("sig").strip()

    return "unknown", "unknown", s_nounit.strip()

def build_signal_catalog(columns: List[str], timestamp_col: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col == timestamp_col:
            continue

        raw_col = str(col)
        hm = HEADER_RE.match(raw_col)
        plant = hm.group("plant").strip() if hm else None
        rest = hm.group("rest").strip() if hm else raw_col

        um = UNIT_RE.search(rest)
        unit_raw = um.group("unit").strip() if um else None

        ctype, cid, raw_signal = parse_component(rest)
        canonical, unit_norm, meta = map_raw_signal_to_canonical(raw_signal, unit_raw)

        rows.append(
            {
                "raw_column_name": raw_col,
                "plant_name": plant,
                "component_type": ctype,
                "component_id": cid,
                "raw_signal_name": raw_signal,
                "unit_raw": unit_raw,
                "unit": unit_norm,
                "canonical_signal": canonical,
                **meta,
            }
        )
    return pd.DataFrame(rows)

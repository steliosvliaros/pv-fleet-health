import re
from typing import Optional, Dict, Tuple, List

UNIT_NORMALIZATION = {
    "W*m^-2": "W/m^2",
    "W*m-2": "W/m^2",
    "W/m2": "W/m^2",
    "Vac": "V",
    "Vdc": "V",
    "kVAr": "kvar",
    "kVA": "kva",
    "C": "°C",
    "Ohm": "ohm",
    "%": "%",
    "Hz": "Hz",
    "A": "A",
    "kW": "kW",
    "kWh": "kWh",
    "m^2": "m^2",
    "-": "-",
}

def normalize_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    u2 = str(u).strip()
    return UNIT_NORMALIZATION.get(u2, u2)

CANONICAL_SIGNAL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)\bCumulative\b.*\benergy\b.*\bmeasured\b", "energy_kwh_counter_measured"),
    (r"(?i)\bCumulative\b.*\benergy\b", "energy_kwh_counter"),
    (r"(?i)\bArray output energy\b", "energy_kwh_interval"),
    (r"(?i)\bPanel group output energy\b", "energy_kwh_interval"),
    (r"(?i)\bArray active output power\b", "ac_power_kw"),
    (r"(?i)\bArray output power\b", "ac_power_kw"),
    (r"(?i)\bPanel group output power\b", "ac_power_kw"),
    (r"(?i)\bArray apparent output power\b", "ac_power_kva"),
    (r"(?i)\bArray reactive output power\b", "q_kvar"),
    (r"(?i)\bTotal power factor\b", "pf"),
    (r"(?i)\bTotal Irradiance\b", "poa_irradiance_wm2"),
    (r"(?i)\bModule Temperature\b", "tmod_c"),
    (r"(?i)\bAmbient Temperature\b", "tamb_c"),
    (r"(?i)\bInternal temperature\b", "inv_internal_temp_c"),
    (r"(?i)\bInverter insulation resistance\b", "insulation_resistance_ohm"),
    (r"(?i)\bInverter leakage current\b", "leakage_current_a"),
    (r"(?i)\bMaintenance performance ratio\b", "vendor_pr_pct"),
    (r"(?i)\bAC voltage unbalance\b", "ac_voltage_unbalance_pct"),
    (r"(?i)\bAC output frequency error\b", "ac_frequency_error_hz"),
    (r"(?i)\bArray output frequency\b", "ac_frequency_hz"),
    (r"(?i)\bArray output current of phase (?P<phase>L[123])\b", "ac_current_a_{phase}"),
    (r"(?i)\bArray output current\b", "ac_current_a"),
    (r"(?i)\bN phase AC output current\b", "ac_current_a_N"),
    (r"(?i)\bAverage strings current\b", "avg_string_current_a"),
    (r"(?i)\bArray output voltage of phase (?P<phase>L[123])\b", "ac_voltage_v_{phase}"),
    (r"(?i)\bArray output voltage\b", "ac_voltage_v"),
    (r"(?i)\bPanel group output voltage\b", "dc_voltage_v"),
    (r"(?i)\bDC Link Voltage\b", "dc_link_voltage_v"),
    (r"(?i)\bDC Link Current\b", "dc_link_current_a"),
    (r"(?i)\bNominal output power\b", "nameplate_kwp"),
    (r"(?i)\bPanel area\b", "panel_area_m2"),
]

CANONICAL_EXPECTED_UNITS: Dict[str, str] = {
    "energy_kwh_counter_measured": "kWh",
    "energy_kwh_counter": "kWh",
    "energy_kwh_interval": "kWh",
    "ac_power_kw": "kW",
    "ac_power_kva": "kva",
    "q_kvar": "kvar",
    "pf": "-",
    "poa_irradiance_wm2": "W/m^2",
    "tmod_c": "°C",
    "tamb_c": "°C",
    "inv_internal_temp_c": "°C",
    "insulation_resistance_ohm": "ohm",
    "leakage_current_a": "A",
    "vendor_pr_pct": "%",
    "ac_voltage_unbalance_pct": "%",
    "ac_frequency_error_hz": "Hz",
    "ac_frequency_hz": "Hz",
    "ac_current_a": "A",
    "ac_current_a_L1": "A",
    "ac_current_a_L2": "A",
    "ac_current_a_L3": "A",
    "ac_current_a_N": "A",
    "avg_string_current_a": "A",
    "ac_voltage_v": "V",
    "ac_voltage_v_L1": "V",
    "ac_voltage_v_L2": "V",
    "ac_voltage_v_L3": "V",
    "dc_voltage_v": "V",
    "dc_link_voltage_v": "V",
    "dc_link_current_a": "A",
    "nameplate_kwp": "kWp",
    "panel_area_m2": "m^2",
}

def map_raw_signal_to_canonical(raw_signal: str, unit: Optional[str]) -> Tuple[str, Optional[str], Dict]:
    u_norm = normalize_unit(unit)
    rs = (raw_signal or "").strip()

    for pattern, template in CANONICAL_SIGNAL_PATTERNS:
        m = re.search(pattern, rs)
        if m:
            canonical = template
            if "{phase}" in template:
                canonical = template.format(phase=m.groupdict().get("phase"))
            expected = CANONICAL_EXPECTED_UNITS.get(canonical)
            meta = {
                "mapped": True,
                "pattern": pattern,
                "expected_unit": expected,
                "unit_ok": (expected is None or u_norm is None or u_norm == expected),
            }
            return canonical, u_norm, meta

    return f"unmapped::{rs}", u_norm, {"mapped": False, "pattern": None, "expected_unit": None, "unit_ok": None}

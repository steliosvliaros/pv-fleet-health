from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    # I/O
    scada_path: str
    events_path: str
    metadata_path: str | None = None
    timestamp_col: str = "Timestamp"
    timestamp_format: str | None = None

    # Time
    default_timezone: str = "Europe/Athens"
    standard_freq: str = "5min"

    # Daylight & plausibility
    daylight_poa_threshold_wm2: float = 50.0
    poa_for_kpi_min_wm2: float = 200.0
    min_valid_poa_wm2: float = 0.0
    max_valid_poa_wm2: float = 1400.0
    min_valid_tmod_c: float = -20.0
    max_valid_tmod_c: float = 90.0
    max_pf_abs: float = 1.2

    # Missing data policy
    max_interp_gap_minutes: int = 15
    drop_if_missing_key_fraction: float = 0.10
    allow_interp_signals: set[str] = frozenset({"poa_irradiance_wm2", "tmod_c", "tamb_c"})
    allow_ffill_signals: set[str] = frozenset()

    # Counters
    counter_reset_negative_kwh_threshold: float = -0.01

    # Irradiance QC
    clearsky_qc_quantile: float = 0.98
    clearsky_qc_min_points_per_day: int = 50

    # Modeling
    model_min_points: int = 2000
    walkforward_train_days: int = 60
    walkforward_test_days: int = 14

    # Anomaly thresholds
    residual_z_threshold: float = 4.0
    rolling_window_days: int = 7

    # Prototype toggles
    selected_plant: str | None = None
    random_seed: int = 42

    def seed(self) -> None:
        np.random.seed(self.random_seed)


def load_config_yaml(path: str) -> Config:
    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as f:
        d: dict[str, Any] = yaml.safe_load(f)

    # Resolve relative paths in the data fields
    base_dir = config_path.parent
    for key in ["scada_path", "events_path", "metadata_path"]:
        if key in d and d[key]:
            p = Path(d[key])
            if not p.is_absolute():
                d[key] = str(base_dir / p)

    # Fill defaults for absent keys by constructing with **d
    cfg = Config(**d)
    cfg.seed()
    return cfg

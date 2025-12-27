from typing import Dict
import numpy as np
import pandas as pd
from .config import Config

try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["poa"] = df["poa_wm2"]
    X["poa2"] = df["poa_wm2"] ** 2
    X["tmod"] = df["tmod_c"]
    X["poa_tmod"] = df["poa_wm2"] * df["tmod_c"]
    return X

def fit_expected_power_model(df: pd.DataFrame, cfg: Config) -> Dict:
    intrinsic = ~df["event_label"].isin(["curtailment", "grid_outage"])
    daylight = df["poa_wm2"] >= cfg.poa_for_kpi_min_wm2
    mask = intrinsic & daylight & df["p_ac_kw"].notna() & df["poa_wm2"].notna() & df["tmod_c"].notna()
    if mask.sum() < cfg.model_min_points:
        return {"ok": False, "reason": f"Not enough points: {mask.sum()}"}

    X = build_features(df.loc[mask])
    y = df.loc[mask, "p_ac_kw"].astype(float)

    if SKLEARN_AVAILABLE:
        model = Pipeline([("scaler", StandardScaler()), ("huber", HuberRegressor())])
        model.fit(X, y)
        return {"ok": True, "model": model, "mask": mask, "features": list(X.columns)}
    if STATSMODELS_AVAILABLE:
        Xc = sm.add_constant(X)
        model = sm.RLM(y, Xc).fit()
        return {"ok": True, "model": model, "mask": mask, "features": list(X.columns), "sm": True}
    return {"ok": False, "reason": "No sklearn or statsmodels installed"}

def predict_expected(model_obj: Dict, df: pd.DataFrame) -> pd.Series:
    if not model_obj.get("ok"):
        return pd.Series(index=df.index, dtype=float)
    X = build_features(df)
    if model_obj.get("sm"):
        import statsmodels.api as sm
        Xc = sm.add_constant(X, has_constant="add")
        yhat = model_obj["model"].predict(Xc)
    else:
        yhat = model_obj["model"].predict(X)
    return pd.Series(yhat, index=df.index, name="p_expected_kw")

def validate_walkforward(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    intrinsic = ~df["event_label"].isin(["curtailment", "grid_outage"])
    daylight = df["poa_wm2"] >= cfg.poa_for_kpi_min_wm2
    usable = intrinsic & daylight & df["p_ac_kw"].notna() & df["poa_wm2"].notna() & df["tmod_c"].notna()

    if usable.sum() < cfg.model_min_points:
        return pd.DataFrame([{"ok": False, "reason": "insufficient usable points"}])

    start = df.index.min().normalize()
    end = df.index.max().normalize()

    rows = []
    cur = start
    while cur + pd.Timedelta(days=cfg.walkforward_train_days + cfg.walkforward_test_days) <= end:
        train_end = cur + pd.Timedelta(days=cfg.walkforward_train_days)
        test_end = train_end + pd.Timedelta(days=cfg.walkforward_test_days)

        train = df[(df.index >= cur) & (df.index < train_end)]
        test = df[(df.index >= train_end) & (df.index < test_end)]

        mobj = fit_expected_power_model(train, cfg)
        if not mobj.get("ok"):
            rows.append({"train_start": cur, "train_end": train_end, "test_end": test_end, "ok": False, "reason": mobj.get("reason")})
            cur = train_end
            continue

        yhat = predict_expected(mobj, test)
        m = usable.reindex(test.index).fillna(False)
        err = (test.loc[m, "p_ac_kw"] - yhat.loc[m]).dropna()
        if len(err) == 0:
            rows.append({"train_start": cur, "train_end": train_end, "test_end": test_end, "ok": False, "reason": "no usable test points"})
        else:
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            rows.append({"train_start": cur, "train_end": train_end, "test_end": test_end, "ok": True, "mae_kw": mae, "rmse_kw": rmse, "n": int(len(err))})
        cur = train_end

    return pd.DataFrame(rows)

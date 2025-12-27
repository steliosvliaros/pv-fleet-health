import pandas as pd


def wide_to_long(
    scada_wide: pd.DataFrame, signal_catalog: pd.DataFrame, timestamp_col: str
) -> pd.DataFrame:
    cols = [timestamp_col] + signal_catalog["raw_column_name"].tolist()
    df = scada_wide.loc[:, cols].copy()

    long = df.melt(id_vars=[timestamp_col], var_name="raw_column_name", value_name="value")

    # Convert value column to numeric, handling European decimal separators
    long["value"] = pd.to_numeric(long["value"].astype(str).str.replace(",", "."), errors="coerce")

    long = long.merge(
        signal_catalog[
            [
                "raw_column_name",
                "plant_name",
                "component_type",
                "component_id",
                "canonical_signal",
                "unit",
                "mapped",
                "unit_ok",
                "expected_unit",
            ]
        ],
        on="raw_column_name",
        how="left",
    )
    long = long.rename(columns={timestamp_col: "ts"})
    return long

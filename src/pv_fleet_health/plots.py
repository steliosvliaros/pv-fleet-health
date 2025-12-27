from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

def plot_missingness_bars(miss_df: pd.DataFrame, title: str = "Missingness by plant & signal"):
    pivot = miss_df.pivot_table(index="plant_name", columns="signal", values="missing_frac")
    ax = pivot.plot(kind="bar", figsize=(14, 5))
    ax.set_ylabel("Missing fraction")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def quicklook_timeseries(df: pd.DataFrame, plant: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    if "p_ac_kw" in df.columns:
        df["p_ac_kw"].plot(ax=ax)
    ax.set_title(f"AC Power (kW) – {plant}")
    ax.set_ylabel("kW")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 4))
    if "poa_wm2" in df.columns:
        df["poa_wm2"].plot(ax=ax)
    ax.set_title(f"POA Irradiance (W/m^2) – {plant}")
    ax.set_ylabel("W/m^2")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 4))
    if "tmod_c" in df.columns:
        df["tmod_c"].plot(ax=ax)
    ax.set_title(f"Module Temp (°C) – {plant}")
    ax.set_ylabel("°C")
    plt.tight_layout()
    plt.show()

def scatter_power_vs_irradiance(df: pd.DataFrame, plant: str, nmax: int = 20000):
    sub = df[["poa_wm2", "p_ac_kw"]].dropna()
    if len(sub) > nmax:
        sub = sub.sample(nmax, random_state=42)
    plt.figure(figsize=(6, 5))
    plt.scatter(sub["poa_wm2"], sub["p_ac_kw"], s=2)
    plt.xlabel("POA (W/m^2)")
    plt.ylabel("AC Power (kW)")
    plt.title(f"Power vs Irradiance – {plant}")
    plt.tight_layout()
    plt.show()

def hist_basic(series: pd.Series, title: str, bins: int = 60):
    s = series.dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.show()

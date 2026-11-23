import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from build_dataset import on_grid, read_folder

TZ = "Europe/Budapest"
YEARS = [2023, 2024, 2025, 2026]
SEASONS = {3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn"}


def prepare(data, raw):
    df = pd.read_parquet(data)
    local = df.index.tz_convert(TZ)
    keep = (local.hour >= 9) & (local.hour < 17) & local.year.isin(YEARS)
    df = df[keep].copy()
    local = local[keep]
    df["year"], df["date"] = local.year, local.date

    gen = read_folder(raw / "generation_HU")
    cols = [c for c in gen.columns if "solar" in c.lower()]
    col = ([c for c in cols if "aggregated" in c.lower()] or cols)[0]
    df["solar_actual"] = on_grid(gen[[col]], df.index).iloc[:, 0]

    day_max = df.groupby("date")["solar_fc"].transform("max")
    df = df[(df["solar_fc"] > 0.3 * day_max) & (df["solar_fc"] > 0) & df["solar_actual"].notna()].copy()
    df["ratio"] = df["solar_actual"] / df["solar_fc"]
    df["fc_bin"] = df.groupby(["year", "month"])["solar_fc"].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    if "wx_shortwave_radiation" in df:
        df["wx_bin"] = pd.cut(df["wx_shortwave_radiation"], [0, 300, 450, 600, 750, 2000], labels=False)
    return df


def matched_gap(df, cell, treat, control_mask=None, min_n=5):
    rows = []
    for year, g in df.groupby("year"):
        if control_mask is not None:
            g = g[control_mask.loc[g.index]]
        stats = g.groupby(cell + [treat])["ratio"].agg(["median", "size"]).unstack(treat)
        if ("size", True) not in stats or ("size", False) not in stats:
            rows.append({"year": year, "gap": np.nan, "cells": 0, "treated_periods": 0})
            continue
        stats = stats.dropna()
        stats = stats[(stats[("size", True)] >= min_n) & (stats[("size", False)] >= min_n)]
        if stats.empty:
            rows.append({"year": year, "gap": np.nan, "cells": 0, "treated_periods": 0})
            continue
        gap = stats[("median", True)] - stats[("median", False)]
        w = stats[("size", True)]
        rows.append({"year": year, "gap": float((gap * w).sum() / w.sum()), "cells": len(stats), "treated_periods": int(w.sum())})
    return pd.DataFrame(rows).set_index("year")


def curtailed_energy(df):
    df = df.copy()
    normal = df[df["price"] > 20]
    typical = normal.groupby(["year", "month", "hour"])["ratio"].median()
    fallback = normal.groupby("year")["ratio"].median()
    key = pd.MultiIndex.from_arrays([df["year"], df["month"], df["hour"]])
    df["typical"] = typical.reindex(key).to_numpy()
    df["typical"] = df["typical"].fillna(df["year"].map(fallback))
    neg = df[df["price"] <= 0]
    expected = neg["solar_fc"] * neg["typical"]
    out = pd.DataFrame({
        "periods_price_le_0": neg.groupby("year").size(),
        "expected_GWh": (expected * 0.25).groupby(neg["year"]).sum() / 1000,
        "actual_GWh": (neg["solar_actual"] * 0.25).groupby(neg["year"]).sum() / 1000,
    })
    out["curtailed_GWh"] = out["expected_GWh"] - out["actual_GWh"]
    out["curtailed_share"] = out["curtailed_GWh"] / out["expected_GWh"]
    return out


def tipping_by_season(data, min_pos=30):
    df = pd.read_parquet(data)
    local = df.index.tz_convert(TZ)
    keep = (local.hour >= 9) & (local.hour < 17) & local.year.isin(YEARS) & local.month.isin(list(SEASONS))
    df = df[keep].dropna(subset=["residual_fc"]).copy()
    df["year"] = local[keep].year
    df["season"] = df["month"].map(SEASONS)
    rows = []
    for (year, season), g in df.groupby(["year", "season"]):
        pos = int(g["y"].sum())
        threshold = np.nan
        if pos >= min_pos and pos < len(g):
            fit = LogisticRegression().fit(g[["residual_fc"]] / 1000, g["y"])
            threshold = -fit.intercept_[0] / fit.coef_[0][0] * 1000
        rows.append({"year": year, "season": season, "periods_price_le_0": pos, "residual_load_at_50pct_MW": threshold})
    return pd.DataFrame(rows).pivot(index="year", columns="season", values="residual_load_at_50pct_MW")[["spring", "summer", "autumn"]], pd.DataFrame(rows)


def plot_season(table, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for season in table.columns:
        ax.plot(table.index, table[season], marker="o", label=season)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(table.index)
    ax.set_ylabel("residual load at 50% chance of price ≤ 0, MW")
    ax.set_title("Tipping point by season, 9:00–17:00")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()

    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    df = prepare(a.data, Path(a.raw))
    df["neg"] = df["price"] <= 0
    df["weekend"] = df["offday"] == 1

    checks = {
        "price<=0 vs >0, same month/hour/forecast level": matched_gap(df, ["month", "hour", "fc_bin"], "neg"),
        "placebo: weekend vs weekday, price > 20 only": matched_gap(df, ["month", "hour", "fc_bin"], "weekend", control_mask=df["price"] > 20),
    }
    if "wx_bin" in df:
        checks["price<=0 vs >0, same month/hour/irradiance"] = matched_gap(df.dropna(subset=["wx_bin"]), ["month", "hour", "wx_bin"], "neg")
    matched = pd.concat(checks, axis=1)
    matched.to_csv(out / "curtailment_matched.csv")
    print("1. difference in actual/forecast solar between the two groups, matched cells (negative = less output)")
    for name, t in checks.items():
        print(f"\n   {name}")
        print(t.round(3).to_string())

    energy = curtailed_energy(df)
    energy.to_csv(out / "curtailed_energy.csv")
    print("\n2. solar energy missing at price <= 0, against the ratio seen at prices above 20 EUR/MWh (9:00-17:00, sunny periods):")
    print(energy.round({"expected_GWh": 1, "actual_GWh": 1, "curtailed_GWh": 1, "curtailed_share": 3}).to_string())

    season, long = tipping_by_season(a.data)
    long.to_csv(out / "tipping_point_by_season.csv", index=False)
    plot_season(season, figs / "tipping_point_by_season.png")
    print("\n3. residual load (MW) at which the chance of price <= 0 reaches 50%, by season (blank = under 30 such periods):")
    print(season.round(0).to_string())
    print(f"\nsaved the tables in {out}/ and {figs}/tipping_point_by_season.png")


if __name__ == "__main__":
    main()
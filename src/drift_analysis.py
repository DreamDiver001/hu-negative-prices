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
PRICE_BINS = [-np.inf, -50, 0, 20, 50, np.inf]
PRICE_LABELS = ["< -50", "-50 to 0", "0 to 20", "20 to 50", "> 50"]


def daylight(df):
    local = df.index.tz_convert(TZ)
    keep = (local.hour >= 9) & (local.hour < 17) & local.year.isin(YEARS)
    out = df[keep].copy()
    out["year"] = local[keep].year
    return out


def tipping_point(df, step=250):
    rows, curves = [], {}
    for year, g in df.groupby("year"):
        g = g.dropna(subset=["residual_fc"])
        if g["y"].nunique() < 2:
            continue
        x = g[["residual_fc"]].to_numpy() / 1000
        fit = LogisticRegression().fit(x, g["y"])
        threshold = -fit.intercept_[0] / fit.coef_[0][0] * 1000
        rows.append({
            "year": year,
            "periods": len(g),
            "share_nonpos": g["y"].mean(),
            "residual_load_at_50pct_MW": threshold,
            "median_residual_load_when_nonpos_MW": g.loc[g["y"] == 1, "residual_fc"].median(),
        })
        bins = (g["residual_fc"] // step) * step
        c = g.groupby(bins)["y"].agg(["mean", "size"])
        curves[year] = c[c["size"] >= 40]["mean"]
    return pd.DataFrame(rows), curves


def solar_ratio(df, raw):
    gen = read_folder(raw / "generation_HU")
    solar_cols = [c for c in gen.columns if "solar" in c.lower()]
    pick = [c for c in solar_cols if "aggregated" in c.lower()] or solar_cols
    if not pick:
        raise SystemExit("no solar column in generation_HU")
    actual = on_grid(gen[[pick[0]]], df.index).iloc[:, 0]
    df = df.assign(solar_actual=actual)
    sunny = df["solar_fc"] > 0.3 * df.groupby([df["year"], df.index.tz_convert(TZ).date])["solar_fc"].transform("max")
    d = df[sunny & df["solar_actual"].notna() & (df["solar_fc"] > 0)].copy()
    d["ratio"] = d["solar_actual"] / d["solar_fc"]
    d["price_bin"] = pd.cut(d["price"], PRICE_BINS, labels=PRICE_LABELS)
    table = d.pivot_table(index="year", columns="price_bin", values="ratio", aggfunc="median", observed=False)
    counts = d.pivot_table(index="year", columns="price_bin", values="ratio", aggfunc="size", observed=False)
    return table, counts


def plot_tipping(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for year, c in curves.items():
        ax.plot(c.index / 1000, c.values, marker=".", label=str(year))
    ax.set_xlabel("residual load forecast, GW (load minus solar minus wind)")
    ax.set_ylabel("share of periods with price ≤ 0")
    ax.set_title("Price ≤ 0 against residual load, 9:00–17:00")
    ax.legend(title="year")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_ratio(table, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for year, row in table.iterrows():
        ax.plot(range(len(row)), row.values, marker="o", label=str(year))
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_xlabel("day-ahead price, €/MWh")
    ax.set_ylabel("actual solar / day-ahead solar forecast (median)")
    ax.set_title("Solar output against its forecast, by price level")
    ax.legend(title="year")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()

    df = daylight(pd.read_parquet(a.data))
    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    tip, curves = tipping_point(df)
    tip.to_csv(out / "tipping_point.csv", index=False)
    plot_tipping(curves, figs / "tipping_point.png")
    print("residual load at which the chance of price <= 0 reaches 50% (9:00-17:00):")
    print(tip.round({"share_nonpos": 3, "residual_load_at_50pct_MW": 0, "median_residual_load_when_nonpos_MW": 0}).to_string(index=False))

    ratio, counts = solar_ratio(df, Path(a.raw))
    ratio.to_csv(out / "solar_ratio_by_price.csv")
    plot_ratio(ratio, figs / "solar_ratio_by_price.png")
    print("\nmedian actual solar / forecast solar, sunny periods, by price (€/MWh):")
    print(ratio.round(3).to_string())
    print("\nnumber of periods behind each cell:")
    print(counts.to_string())
    print(f"\nsaved {figs}/tipping_point.png, {figs}/solar_ratio_by_price.png and the tables in {out}/")


if __name__ == "__main__":
    main()
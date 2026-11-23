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
    df["neg"] = df["price"] <= 0
    df["band_low"] = df["price"] <= 50
    return df


def gap(g, cell, treat, min_n=5):
    stats = g.groupby(cell + [treat])["ratio"].agg(["median", "size"]).unstack(treat)
    if ("size", True) not in stats or ("size", False) not in stats:
        return np.nan, 0, 0
    stats = stats.dropna()
    stats = stats[(stats[("size", True)] >= min_n) & (stats[("size", False)] >= min_n)]
    if stats.empty:
        return np.nan, 0, 0
    w = stats[("size", True)]
    d = stats[("median", True)] - stats[("median", False)]
    return float((d * w).sum() / w.sum()), len(stats), int(w.sum())


def curtailed(g):
    normal = g[g["price"] > 20]
    neg = g[g["neg"]]
    if normal.empty or neg.empty:
        return np.nan, np.nan
    typical = normal.groupby(["month", "hour"])["ratio"].median()
    key = pd.MultiIndex.from_arrays([neg["month"], neg["hour"]])
    t = pd.Series(typical.reindex(key).to_numpy(), index=neg.index).fillna(normal["ratio"].median())
    expected = (neg["solar_fc"] * t).sum() * 0.25 / 1000
    actual = neg["solar_actual"].sum() * 0.25 / 1000
    return expected - actual, (expected - actual) / expected


def with_ci(df, stat, reps, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in df.groupby("year"):
        point = stat(g)
        days = g["date"].to_numpy()
        unique = pd.unique(days)
        pos = {d: np.flatnonzero(days == d) for d in unique}
        draws = []
        for _ in range(reps):
            pick = rng.choice(len(unique), size=len(unique), replace=True)
            draws.append(stat(g.iloc[np.concatenate([pos[unique[i]] for i in pick])])[0])
        draws = np.array(draws, dtype=float)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5]) if np.isfinite(draws).any() else (np.nan, np.nan)
        rows.append({"year": year, "estimate": point[0], "ci_low": lo, "ci_high": hi, "detail": point[1:]})
    return pd.DataFrame(rows).set_index("year")


def tipping_by_season(data, min_pos=100):
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
    long = pd.DataFrame(rows)
    wide = long.pivot(index="year", columns="season", values="residual_load_at_50pct_MW")
    return wide[[s for s in ["spring", "summer", "autumn"] if s in wide]], long


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


def tidy(t, cols):
    t = t.copy()
    for i, c in enumerate(cols):
        t[c] = t["detail"].map(lambda d: d[i])
    return t.drop(columns="detail")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    p.add_argument("--reps", type=int, default=200)
    a = p.parse_args()

    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    df = prepare(a.data, Path(a.raw))
    normal = df[df["price"] > 20]
    checks = {
        "price<=0 vs >0, same month/hour/forecast level": (df, ["month", "hour", "fc_bin"], "neg"),
        "placebo: price 20-50 vs >50, same month/hour/forecast level": (normal, ["month", "hour", "fc_bin"], "band_low"),
    }
    if "wx_bin" in df:
        checks["price<=0 vs >0, same month/hour/irradiance"] = (df.dropna(subset=["wx_bin"]), ["month", "hour", "wx_bin"], "neg")

    print(f"1. gap in actual/forecast solar between the two groups (negative = less output), 95% CI from {a.reps} day resamples")
    tables = []
    for name, (d, cell, treat) in checks.items():
        t = tidy(with_ci(d, lambda g, c=cell, tr=treat: gap(g, c, tr), a.reps), ["cells", "periods"])
        tables.append(t.assign(check=name))
        print(f"\n   {name}")
        print(t.round(3).to_string())
    pd.concat(tables).to_csv(out / "curtailment_matched.csv")

    energy = tidy(with_ci(df, curtailed, a.reps), ["share"])
    energy = energy.rename(columns={"estimate": "curtailed_GWh", "ci_low": "GWh_ci_low", "ci_high": "GWh_ci_high", "share": "share_of_expected"})
    energy.to_csv(out / "curtailed_energy.csv")
    print("\n2. solar energy missing at price <= 0 against the usual ratio at prices above 20 EUR/MWh (9:00-17:00, sunny periods)")
    print(energy.round(3).to_string())

    season, long = tipping_by_season(a.data)
    long.to_csv(out / "tipping_point_by_season.csv", index=False)
    plot_season(season, figs / "tipping_point_by_season.png")
    print("\n3. residual load (MW) at which the chance of price <= 0 reaches 50%, by season (blank = under 100 such periods)")
    print(season.round(0).to_string())
    print(f"\nsaved the tables in {out}/ and {figs}/tipping_point_by_season.png")


if __name__ == "__main__":
    main()

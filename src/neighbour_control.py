import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from build_dataset import on_grid, read_folder
from drift_analysis import TZ, daylight

ZONES = ["AT", "SK", "RO", "SI", "HR"]
BORDERS = ["AT", "SK", "RO", "HR", "SI", "RS", "UA"]


def have(folder):
    return folder.exists() and any(folder.glob("*.parquet"))


def neighbour_prices(raw, grid):
    cols = {z: on_grid(read_folder(raw / f"prices_{z}").iloc[:, :1], grid).iloc[:, 0]
            for z in ZONES if have(raw / f"prices_{z}")}
    if not cols:
        raise SystemExit("no neighbour prices, run: python src/collect_entsoe.py --only " + " ".join(f"prices_{z}" for z in ZONES))
    return pd.DataFrame(cols)


def net_export(raw, grid):
    parts = []
    for n in BORDERS:
        out, inn = raw / f"flow_HU_{n}", raw / f"flow_{n}_HU"
        if have(out) and have(inn):
            parts.append(on_grid(read_folder(out).iloc[:, :1], grid).iloc[:, 0]
                         - on_grid(read_folder(inn).iloc[:, :1], grid).iloc[:, 0])
    if not parts:
        return None, 0
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts)), len(parts)


def prepare(df, raw):
    nb = neighbour_prices(raw, df.index)
    df = df.assign(
        nb_share=(nb <= 0).sum(axis=1) / nb.notna().sum(axis=1),
        nb_all=(nb <= 0).all(axis=1),
        nb_none=~(nb <= 0).any(axis=1),
        same_as_at=(df["price"] - nb["AT"]).abs() < 0.01 if "AT" in nb else np.nan,
    )
    ex, borders = net_export(raw, df.index)
    if ex is not None:
        df["net_export"] = ex
    return df, list(nb.columns), borders


def coupling(df):
    rows = []
    for year, g in df.groupby("year"):
        neg = g[g["y"] == 1]
        row = {
            "year": year,
            "hu_nonpos_periods": len(neg),
            "all_neighbours_nonpos_too": neg["nb_all"].mean(),
            "no_neighbour_nonpos": neg["nb_none"].mean(),
            "hu_price_equals_at_daylight": g["same_as_at"].mean(),
        }
        if "net_export" in g:
            row["median_net_export_MW_when_nonpos"] = neg["net_export"].median()
            row["median_net_export_MW_otherwise"] = g.loc[g["y"] == 0, "net_export"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def design(g, controls):
    cols = [g["residual_fc"] / 1000] + [g[c] / (1000 if c == "net_export" else 1) for c in controls]
    return np.column_stack(cols)


def threshold(fit, ref):
    b = fit.coef_[0]
    return -(fit.intercept_[0] + sum(b[i + 1] * v for i, v in enumerate(ref))) / b[0] * 1000


def scaled(ref, controls):
    return [v / 1000 if c == "net_export" else v for c, v in zip(controls, ref)]


def tipping(df, specs, reps, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, controls, ref in specs:
        need = ["residual_fc"] + controls
        for year, g in df.dropna(subset=need).groupby("year"):
            if g["y"].sum() < 100:
                continue
            x, y = design(g, controls), g["y"].to_numpy()
            point = threshold(LogisticRegression().fit(x, y), scaled(ref, controls))
            days = pd.factorize(g.index.tz_convert(TZ).date)[0]
            n_days = days.max() + 1
            boot = []
            for _ in range(reps):
                w = np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)[days]
                if y[w > 0].min() == y[w > 0].max():
                    continue
                boot.append(threshold(LogisticRegression().fit(x, y, sample_weight=w), scaled(ref, controls)))
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            rows.append({"model": name, "year": year, "nonpos": int(y.sum()), "tipping_MW": point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--out", default="results")
    p.add_argument("--reps", type=int, default=200)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    df, zones, borders = prepare(daylight(pd.read_parquet(a.data)), Path(a.raw))
    print(f"neighbour prices: {', '.join(zones)}; borders with flows in both directions: {borders}")

    c = coupling(df)
    c.to_csv(out / "neighbour_coupling.csv", index=False)
    print("\nhow often Hungary goes to zero or below on its own (9:00-17:00)")
    print(c.round(3).to_string(index=False))

    pooled_share = df["nb_share"].mean()
    specs = [
        ("residual load only", [], []),
        ("+ neighbours, at the 2023-26 average share", ["nb_share"], [pooled_share]),
        ("+ neighbours, none at or below zero", ["nb_share"], [0.0]),
    ]
    if "net_export" in df:
        covered = df["net_export"].notna().mean()
        if covered < 0.8:
            print(f"\nnet exports cover only {covered:.0%} of periods, read those rows with care")
        pooled_ex = df["net_export"].mean()
        specs += [
            ("+ net exports, at the 2023-26 average", ["net_export"], [pooled_ex]),
            ("+ neighbours and net exports, both at average", ["nb_share", "net_export"], [pooled_share, pooled_ex]),
        ]
        print(f"\naverage share of neighbours at or below zero: {pooled_share:.3f}; average net export: {pooled_ex:.0f} MW")

    t = tipping(df, specs, a.reps)
    t.to_csv(out / "neighbour_tipping.csv", index=False)
    t["cell"] = t.apply(lambda r: f"{r.tipping_MW:,.0f} [{r.ci_low:,.0f}, {r.ci_high:,.0f}]", axis=1)
    print(f"\nresidual load (MW) at which the chance of price <= 0 reaches 50%, 95% interval from {a.reps} resamples of days")
    print(t.pivot(index="model", columns="year", values="cell").reindex([s[0] for s in specs]).to_string())
    shift = t.pivot(index="model", columns="year", values="tipping_MW")
    if {2025, 2026} <= set(shift.columns):
        print("\nchange 2025 -> 2026 (MW):")
        print((shift[2026] - shift[2025]).reindex([s[0] for s in specs]).round(0).to_string())
    print(f"\nsaved neighbour_coupling.csv and neighbour_tipping.csv in {out}/")


if __name__ == "__main__":
    main()
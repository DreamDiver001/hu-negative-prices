import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import CYCLIC, FEATURES, TZ, add_cyclic
from experiments import DIRECTION, regularised
from make_figures import INK, MUTED, YEAR_COLORS, save, style

GROUPS = {
    "ENTSO-E forecasts": ["load_fc", "solar_fc", "wind_fc", "residual_fc", "solar_share_fc", "solar_fc_daymax", "residual_fc_daymin"],
    "calendar": ["qh", "hour", "weekday", "month", "doy", "holiday", "offday"],
    "price history": ["price_lag1d", "price_lag7d", "price_min_prevday", "price_mean_prevday", "nonpos_share_prevday"],
}
GROUP_COLORS = {"ENTSO-E forecasts": "#2a78d6", "calendar": "#eb6834", "price history": "#1baf7a", "weather": "#eda100"}
NAMES = {
    "residual_fc": "residual load forecast", "residual_fc_daymin": "day's lowest residual load",
    "solar_share_fc": "solar share of load", "solar_fc": "solar forecast", "solar_fc_daymax": "day's peak solar",
    "load_fc": "load forecast", "wind_fc": "wind forecast", "qh": "quarter-hour of day", "hour": "hour",
    "weekday": "weekday", "month": "month", "doy": "day of year", "holiday": "public holiday",
    "offday": "weekend or holiday", "price_lag1d": "price a day earlier", "price_lag7d": "price a week earlier",
    "price_min_prevday": "yesterday's lowest price", "price_mean_prevday": "yesterday's mean price",
    "nonpos_share_prevday": "yesterday's share ≤ 0", "wx_spread_radiation": "radiation spread across sites",
    "wx_temperature_2m": "temperature", "wx_shortwave_radiation": "solar radiation",
}


def name(col):
    return NAMES.get(col, col.removeprefix("wx_").replace("_", " "))
PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "matplotlib", "entsoe-py", "holidays"]


def group_of(col):
    for g, cols in GROUPS.items():
        if col in cols:
            return g
    return "weather" if col.startswith("wx_") else "other"


def shap_values(model, x):
    contrib = model.predict(x, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], index=x.index, columns=x.columns)


def importance(shap, mask=None):
    s = shap if mask is None else shap[mask]
    imp = s.abs().mean().sort_values(ascending=False)
    out = pd.DataFrame({"mean_abs_shap": imp, "share": imp / imp.sum()})
    out["group"] = [group_of(c) for c in out.index]
    return out


def figure(imp, shap, x, out, top=12):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.3})
    for ax in (left, right):
        style(ax)
    t = imp.head(top).iloc[::-1]
    left.barh([name(c) for c in t.index], t["share"] * 100,
              color=[GROUP_COLORS.get(g, MUTED) for g in t["group"]], height=0.65)
    left.set_xlabel("share of mean |SHAP|, %", color=INK, fontsize=9.5)
    left.tick_params(axis="y", labelsize=8.5)
    left.grid(axis="y", visible=False)
    left.set_title("What the regularised model relies on", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in GROUP_COLORS.items() if g in set(imp["group"])]
    left.legend(handles, [g for g in GROUP_COLORS if g in set(imp["group"])], frameon=False, fontsize=8.5,
                labelcolor=INK, loc="lower right")

    sample = x.sample(min(len(x), 20000), random_state=0)
    years = sample.index.tz_convert(TZ).year
    for year in sorted(set(years)):
        m = years == year
        right.scatter(sample.loc[m, "residual_fc"] / 1000, shap.loc[sample.index[m], "residual_fc"],
                      s=4, alpha=0.35, color=YEAR_COLORS.get(year, MUTED), label=str(year), linewidths=0)
    right.axhline(0, color=MUTED, lw=0.8, ls="--")
    right.set_xlabel("residual load forecast, GW", color=INK, fontsize=9.5)
    right.set_ylabel("SHAP value, log-odds of price ≤ 0", color=INK, fontsize=9.5)
    right.set_title("Effect of residual load", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    leg = right.legend(frameon=False, fontsize=8.5, labelcolor=INK, title="year", title_fontsize=8.5,
                       markerscale=3, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.13)
    return save(fig, out)


def versions():
    rows = [("python", platform.python_version())]
    for p in PACKAGES:
        try:
            rows.append((p, version(p)))
        except PackageNotFoundError:
            rows.append((p, "not installed"))
    return pd.DataFrame(rows, columns=["package", "version"])


def settings(tree):
    def params(m):
        keep = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
                "subsample_freq", "colsample_bytree", "reg_lambda", "random_state"]
        p = m.get_params()
        return ", ".join(f"{k}={p[k]}" for k in keep if p.get(k) not in (None, 0.0) or k == "random_state")
    from lightgbm import LGBMClassifier
    default = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0)
    mono = ", ".join(f"{c} {'+' if d > 0 else '-'}" for c, d in DIRECTION.items())
    return pd.DataFrame([
        ("logistic regression", "median imputation, standardised, L2 with C=1, max_iter=2000", "features + sin/cos of time of day and year + weather"),
        ("LightGBM, default", params(default), "features + weather"),
        ("LightGBM, regularised", params(regularised(tree)) + f", monotone: {mono}", "features + weather"),
        ("LightGBM, calibrated", "regularised + isotonic regression fitted on the last 6 months of training", "features + weather"),
        ("price quantiles", "LightGBM quantile loss, n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, reg_lambda=5", "features + weather"),
        ("backtest", "train from 2023-01, refit monthly, test 2024-01 to 2026-11", ""),
    ], columns=["model", "settings", "inputs"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--first-test", default="2024-01")
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    df = add_cyclic(pd.read_parquet(a.data))
    df = df[df.index >= pd.Timestamp(a.train_start, tz=TZ)].dropna(subset=["y"])
    tree = FEATURES + [c for c in df.columns if c.startswith("wx_")]

    print(f"fitting the regularised model on {len(df):,} periods up to {df.index.max().tz_convert(TZ):%Y-%m-%d}")
    model = regularised(tree).fit(df[tree], df["y"])
    x = df.loc[df.index >= pd.Timestamp(a.first_test + "-01", tz=TZ), tree]
    shap = shap_values(model, x)

    overall = importance(shap)
    nonpos = importance(shap, df.loc[x.index, "y"].to_numpy() == 1)
    table = overall.join(nonpos[["share"]].rename(columns={"share": "share_when_nonpos"}))
    table.to_csv(out / "shap_importance.csv")
    groups = table.groupby("group")[["share", "share_when_nonpos"]].sum().sort_values("share", ascending=False)
    groups.to_csv(out / "shap_groups.csv")

    print("\nshare of mean |SHAP| by feature (all test periods, and only those with price <= 0)")
    print(table.head(15).assign(share=lambda t: (t["share"] * 100).round(1),
                                share_when_nonpos=lambda t: (t["share_when_nonpos"] * 100).round(1))
          [["group", "share", "share_when_nonpos"]].to_string())
    print("\nby group, %")
    print((groups * 100).round(1).to_string())

    old = regularised(tree).fit(*[d[d.index < pd.Timestamp("2025-10-01", tz=TZ)] for d in (df[tree], df["y"])])
    after = x[x.index >= pd.Timestamp("2025-10-01", tz=TZ)]
    if len(after):
        compare = pd.DataFrame({
            "trained to Sep 2025": importance(shap_values(old, after))["share"],
            "trained on all": importance(shap_values(model, after))["share"],
        }).sort_values("trained on all", ascending=False).head(10) * 100
        compare.round(1).to_csv(out / "shap_before_after_switch.csv")
        print("\nsame periods since Oct 2025, model trained before the switch vs on everything, %")
        print(compare.round(1).to_string())

    print("\nsaved", figure(overall, shap, x, Path(a.figures) / "paper_shap"))

    s, v = settings(tree), versions()
    s.to_csv(out / "model_settings.csv", index=False)
    v.to_csv(out / "versions.csv", index=False)
    print("\nmodel settings")
    print(s.to_string(index=False))
    print("\nversions")
    print(v.to_string(index=False))
    print(f"\nsaved shap_importance.csv, shap_groups.csv, shap_before_after_switch.csv, model_settings.csv, versions.csv in {out}/")


if __name__ == "__main__":
    main()
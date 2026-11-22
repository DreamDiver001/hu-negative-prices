import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "Europe/Budapest"
SWITCH = pd.Timestamp("2025-10-01", tz=TZ)

FEATURES = [
    "qh", "hour", "weekday", "month", "doy", "holiday", "offday",
    "load_fc", "solar_fc", "wind_fc", "residual_fc", "solar_share_fc",
    "solar_fc_daymax", "residual_fc_daymin",
    "price_lag1d", "price_lag7d", "price_min_prevday", "price_mean_prevday", "nonpos_share_prevday",
]
CYCLIC = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def add_cyclic(df):
    df = df.copy()
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["qh"] / 96), np.cos(2 * np.pi * df["qh"] / 96)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * df["doy"] / 366), np.cos(2 * np.pi * df["doy"] / 366)
    return df


def climatology(train, test):
    rate = train.groupby(["month", "qh"])["y"].mean()
    key = pd.MultiIndex.from_arrays([test["month"], test["qh"]])
    return rate.reindex(key).fillna(train["y"].mean()).to_numpy()


def persistence(train, test):
    return (test["price_lag1d"] <= 0).astype(float).to_numpy()


def logistic(cols):
    def fit_predict(train, test):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(train[cols], train["y"])
        return model.predict_proba(test[cols])[:, 1]
    return fit_predict


def lightgbm(cols):
    def fit_predict(train, test):
        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
        )
        model.fit(train[cols], train["y"])
        return model.predict_proba(test[cols])[:, 1]
    return fit_predict


def period_label(idx):
    local = idx.tz_convert(TZ)
    labels = np.where(local.year >= 2025, "Jan-Sep 2025", local.year.astype(str))
    return np.where(local >= SWITCH, "Oct 2025 on (15-min)", labels)


def backtest(df, models, first_test):
    ym = df.index.tz_convert(TZ).strftime("%Y-%m")
    out = []
    for m in sorted(x for x in set(ym) if x >= first_test):
        train, test = df[ym < m], df[ym == m]
        t0 = time.time()
        preds = test[["y", "price"]].copy()
        for name, fit_predict in models.items():
            if train["y"].nunique() < 2 and name not in ("climatology", "persistence"):
                preds[name] = climatology(train, test)
            else:
                preds[name] = fit_predict(train, test)
        out.append(preds)
        print(f"{m}: {len(test):,} test rows, {int(test['y'].sum()):,} with price <= 0 ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(out)
    preds["period"] = period_label(preds.index)
    return preds


def score(preds, models):
    rows = []
    groups = [("all", preds)] + list(preds.groupby("period", sort=False))
    for period, g in groups:
        ref = brier_score_loss(g["y"], g["climatology"])
        for name in models:
            brier = brier_score_loss(g["y"], g[name])
            rows.append({
                "period": period,
                "model": name,
                "base_rate": g["y"].mean(),
                "brier": brier,
                "brier_skill": 1 - brier / ref if ref > 0 else np.nan,
                "pr_auc": average_precision_score(g["y"], g[name]) if g["y"].any() else np.nan,
            })
    return pd.DataFrame(rows)


def reliability(preds, names, path):
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    for name in names:
        b = pd.cut(preds[name], bins, include_lowest=True)
        g = preds.groupby(b, observed=True).agg(p=(name, "mean"), obs=("y", "mean"), n=("y", "size"))
        ax.plot(g["p"], g["obs"], marker="o", label=name)
    ax.set_xlabel("predicted probability of price ≤ 0")
    ax.set_ylabel("observed share")
    ax.set_title("Reliability, all test months")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--first-test", default="2024-01")
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()

    df = add_cyclic(pd.read_parquet(a.data))
    df = df[df.index >= pd.Timestamp(a.train_start, tz=TZ)]
    wx = [c for c in df.columns if c.startswith("wx_")]

    models = {
        "climatology": climatology,
        "persistence": persistence,
        "residual_load_logit": logistic(["residual_fc"]),
        "logistic": logistic(FEATURES + CYCLIC + wx),
        "lightgbm": lightgbm(FEATURES + wx),
    }

    preds = backtest(df, models, a.first_test)
    metrics = score(preds, models)

    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    preds.to_parquet(out / "baseline_predictions.parquet")
    metrics.to_csv(out / "baseline_metrics.csv", index=False)
    reliability(preds, ["logistic", "lightgbm"], figs / "reliability.png")

    final = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
        importance_type="gain",
    ).fit(df[FEATURES + wx], df["y"])
    importance = pd.Series(final.feature_importances_, index=FEATURES + wx).sort_values(ascending=False)
    (importance / importance.sum()).round(4).to_csv(out / "feature_importance.csv", header=["share_of_gain"])

    pd.set_option("display.width", 140)
    order = list(dict.fromkeys(metrics["period"]))
    for col in ["brier_skill", "pr_auc"]:
        table = metrics.pivot(index="model", columns="period", values=col)[order].loc[list(models)]
        print(f"\n{col} (higher is better)")
        print(table.round(3).to_string())
    print("\nbase rate by period:")
    print(metrics.groupby("period", sort=False)["base_rate"].first().round(4).to_string())
    print("\ntop features (LightGBM, share of gain):")
    print((importance / importance.sum()).head(10).round(3).to_string())
    print(f"\nsaved {out}/baseline_metrics.csv, {out}/baseline_predictions.parquet, {figs}/reliability.png")


if __name__ == "__main__":
    main()

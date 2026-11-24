import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from baselines import CYCLIC, FEATURES, TZ, add_cyclic, backtest, climatology, logistic, score
from experiments import lgbm_reg

ENTSOE_FORECASTS = ["solar_fc", "wind_fc", "residual_fc", "solar_share_fc", "solar_fc_daymax", "residual_fc_daymin"]


def daily_loss_gap(preds, a, b):
    loss_a = (preds[a] - preds["y"]) ** 2
    loss_b = (preds[b] - preds["y"]) ** 2
    day = preds.index.tz_convert(TZ).tz_localize(None).normalize()
    return (loss_a - loss_b).groupby(day).mean()


def diebold_mariano(d, lag=7):
    d = d.dropna().to_numpy()
    n = len(d)
    u = d - d.mean()
    var = (u @ u) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return d.mean(), stat, 2 * norm.sf(abs(stat)), n


def dm_table(preds, pairs, source):
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        mean, stat, p, n = diebold_mariano(daily_loss_gap(preds, a, b))
        rows.append({
            "source": source, "model_a": a, "model_b": b, "days": n,
            "mean_brier_gap_a_minus_b": mean, "dm_stat": stat, "p_value": p,
            "better": a if mean < 0 else b,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--train-start", default="2024-01-01")
    p.add_argument("--first-test", default="2025-01")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    df = add_cyclic(pd.read_parquet(a.data))
    df = df[df.index >= pd.Timestamp(a.train_start, tz=TZ)]
    wx = [c for c in df.columns if c.startswith("wx_")]
    if not wx:
        raise SystemExit("no weather columns: run collect_weather.py --mode dayahead and build_dataset.py first")

    full = FEATURES + wx
    no_fc = [c for c in FEATURES if c not in ENTSOE_FORECASTS] + wx
    models = {
        "climatology": climatology,
        "logistic, all features": logistic(full + CYCLIC),
        "logistic, no ENTSO-E wind/solar forecast": logistic(no_fc + CYCLIC),
        "lgbm_reg, all features": lgbm_reg(full),
        "lgbm_reg, no ENTSO-E wind/solar forecast": lgbm_reg(no_fc),
    }
    print(f"timing check: trained from {a.train_start}, tested from {a.first_test}, weather in every training row")
    preds = backtest(df, models, a.first_test)
    metrics = score(preds, models)
    metrics.to_csv(out / "timing_check_metrics.csv", index=False)

    order = list(dict.fromkeys(metrics["period"]))
    for col in ["brier_skill", "pr_auc"]:
        t = metrics.pivot(index="model", columns="period", values=col)[order].loc[list(models)]
        print(f"\n{col}")
        print(t.round(3).to_string())

    tests = [dm_table(preds, [
        ("logistic, no ENTSO-E wind/solar forecast", "logistic, all features"),
        ("lgbm_reg, no ENTSO-E wind/solar forecast", "lgbm_reg, all features"),
        ("logistic, no ENTSO-E wind/solar forecast", "climatology"),
    ], "timing check")]

    exp = out / "experiment_predictions.parquet"
    if exp.exists():
        tests.append(dm_table(pd.read_parquet(exp), [
            ("logistic", "climatology"),
            ("lgbm_reg", "climatology"),
            ("lgbm_reg", "lightgbm"),
            ("lgbm_reg", "logistic"),
            ("lgbm_reg_cal", "lgbm_reg"),
        ], "experiments (2024 on)"))
    else:
        print(f"\n{exp} not found, run experiments.py to test the RQ1 model comparisons too")

    dm = pd.concat(tests, ignore_index=True)
    dm.to_csv(out / "dm_tests.csv", index=False)
    print("\nDiebold-Mariano tests on daily mean Brier loss (negative gap = model_a better; HAC variance, 7-day lag)")
    print(dm.drop(columns="source").assign(
        mean_brier_gap_a_minus_b=dm["mean_brier_gap_a_minus_b"].map("{:.5f}".format),
        dm_stat=dm["dm_stat"].round(2),
        p_value=dm["p_value"].map("{:.3g}".format),
    ).to_string(index=False))
    print(f"\nsaved {out}/timing_check_metrics.csv and {out}/dm_tests.csv")


if __name__ == "__main__":
    main()
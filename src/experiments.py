import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from baselines import CYCLIC, FEATURES, TZ, add_cyclic, backtest, climatology, lightgbm, logistic, reliability, score

SWITCH = pd.Timestamp("2025-10-01", tz=TZ)
DIRECTION = {"residual_fc": -1, "residual_fc_daymin": -1, "solar_share_fc": 1}


def regularised(cols):
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        monotone_constraints=[DIRECTION.get(c, 0) for c in cols],
        verbose=-1, random_state=0,
    )


def lgbm_reg(cols):
    def fit_predict(train, test):
        return regularised(cols).fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]
    return fit_predict


def lgbm_reg_cal(cols, months=6, min_positives=20):
    def fit_predict(train, test):
        cut = train.index.max() - pd.DateOffset(months=months)
        early, recent = train[train.index <= cut], train[train.index > cut]
        full = regularised(cols).fit(train[cols], train["y"])
        raw = full.predict_proba(test[cols])[:, 1]
        if early["y"].nunique() < 2 or recent["y"].sum() < min_positives:
            return raw
        held_out = regularised(cols).fit(early[cols], early["y"]).predict_proba(recent[cols])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(held_out, recent["y"])
        return iso.predict(raw)
    return fit_predict


def quarter_disagreement(df):
    after = df[df.index >= SWITCH]
    hour = after.index.floor("h")
    per_hour = after.groupby(hour)["y"].agg(["sum", "size"])
    per_hour = per_hour[per_hour["size"] == 4]
    any_neg = per_hour[per_hour["sum"] > 0]
    return len(per_hour), len(any_neg), (any_neg["sum"] < 4).mean() if len(any_neg) else np.nan


def season(idx):
    local = idx.tz_convert(TZ)
    q = (local.month - 1) // 3 + 1
    return pd.Index(local.year.astype(str) + "-Q" + q.astype(str))


def transfer(df, retrained, frozen_models):
    before, after = df[df.index < SWITCH], df[df.index >= SWITCH]
    rows = []
    for name, fit_predict in frozen_models.items():
        frozen = pd.Series(fit_predict(before, after), index=after.index)
        both = retrained.loc[after.index].assign(frozen=frozen)
        groups = [("all after switch", both)] + list(both.groupby(season(both.index)))
        for label, g in groups:
            ref = brier_score_loss(g["y"], g["climatology"])
            for kind, col in [("frozen", "frozen"), ("retrained monthly", name)]:
                b = brier_score_loss(g["y"], g[col])
                rows.append({
                    "model": name, "training": kind, "period": label,
                    "base_rate": g["y"].mean(),
                    "brier_skill": 1 - b / ref if ref > 0 else np.nan,
                    "pr_auc": average_precision_score(g["y"], g[col]) if g["y"].any() else np.nan,
                })
    return pd.DataFrame(rows)


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
    lin, tree = FEATURES + CYCLIC + wx, FEATURES + wx

    models = {
        "climatology": climatology,
        "logistic": logistic(lin),
        "lightgbm": lightgbm(tree),
        "lgbm_reg": lgbm_reg(tree),
        "lgbm_reg_cal": lgbm_reg_cal(tree),
    }
    print("monthly backtest")
    preds = backtest(df, models, a.first_test)
    metrics = score(preds, models)

    print("\nfrozen at the switch (RQ2)")
    rq2 = transfer(df, preds, {"logistic": logistic(lin), "lgbm_reg": lgbm_reg(tree)})
    hours, neg_hours, split = quarter_disagreement(df)

    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    preds.to_parquet(out / "experiment_predictions.parquet")
    metrics.to_csv(out / "experiment_metrics.csv", index=False)
    rq2.to_csv(out / "transfer_metrics.csv", index=False)
    reliability(preds, ["logistic", "lightgbm", "lgbm_reg", "lgbm_reg_cal"], figs / "reliability_experiments.png")

    pd.set_option("display.width", 160)
    order = list(dict.fromkeys(metrics["period"]))
    for col in ["brier_skill", "pr_auc"]:
        t = metrics.pivot(index="model", columns="period", values=col)[order].loc[list(models)]
        print(f"\n{col}, monthly retraining (higher is better)")
        print(t.round(3).to_string())

    print("\nRQ2: trained once before 1 Oct 2025 vs retrained every month")
    t = rq2.pivot_table(index=["model", "training"], columns="period", values=["brier_skill", "pr_auc"], sort=False, dropna=False)
    print(t.round(3).to_string())

    print(f"\nafter the switch: {hours:,} full hours, {neg_hours:,} with at least one quarter <= 0;")
    print(f"in {split:.1%} of those, the four quarters did not all agree")
    print(f"\nsaved {out}/experiment_metrics.csv, {out}/transfer_metrics.csv, {figs}/reliability_experiments.png")


if __name__ == "__main__":
    main()
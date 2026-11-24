import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from baselines import CYCLIC, FEATURES, TZ, add_cyclic, backtest, climatology, logistic, period_label
from battery import backtest_prices, quantile_model, schedule
from curtailment_checks import gap, prepare
from experiments import lgbm_reg, season
from robustness import daily_loss_gap, diebold_mariano


def skill(y, p, ref):
    r = brier_score_loss(y, ref)
    return 1 - brier_score_loss(y, p) / r if r > 0 else np.nan


# 1. RQ2 control year

def control_year(df, preds, models):
    runs = [
        ("freeze 1 Oct 2024, test 2025", "2024-10-01", "2025-01-01", "2025-10-01"),
        ("freeze 1 Oct 2025, test 2026", "2025-10-01", "2026-01-01", "2026-10-01"),
    ]
    rows = []
    for label, freeze, start, end in runs:
        freeze, start, end = (pd.Timestamp(x, tz=TZ) for x in (freeze, start, end))
        train, test = df[df.index < freeze], df[(df.index >= start) & (df.index < end)]
        ret = preds.loc[preds.index.intersection(test.index)]
        test = test.loc[ret.index]
        for name, fit_predict in models.items():
            fit = fit_predict if train["y"].nunique() == 2 else climatology
            frozen = pd.Series(fit(train, test), index=test.index)
            for q, idx in test.groupby(season(test.index)).groups.items():
                g = ret.loc[idx]
                rows.append({
                    "run": label, "model": name, "quarter": q[-2:], "base_rate": g["y"].mean(),
                    "frozen": skill(g["y"], frozen.loc[idx], g["climatology"]),
                    "retrained": skill(g["y"], g[name], g["climatology"]),
                })
    out = pd.DataFrame(rows)
    out["gap"] = out["retrained"] - out["frozen"]
    return out


# 3. stronger baselines

def climatology_12m(train, test):
    recent = train[train.index > train.index.max() - pd.DateOffset(months=12)]
    return climatology(recent if recent["y"].any() else train, test)


def persistence_7d(train, test):
    return test["nonpos_7d"].to_numpy()


def add_nonpos_7d(df):
    p = df["price"]
    hits = sum((p.reindex(df.index - pd.Timedelta(days=k)).to_numpy() <= 0).astype(float) for k in range(1, 8))
    return df.assign(nonpos_7d=hits / 7)


def baselines(df, exp, first_test):
    calendar = CYCLIC + ["offday", "holiday", "weekday", "month"]
    models = {
        "climatology_12m": climatology_12m,
        "persistence_7d": persistence_7d,
        "calendar_logit": logistic(calendar),
    }
    new = backtest(df, models, first_test)
    preds = new.join(exp[["climatology", "logistic", "lgbm_reg"]], how="inner")
    names = ["climatology", "climatology_12m", "persistence_7d", "calendar_logit", "logistic", "lgbm_reg"]
    rows = []
    for period, g in [("all", preds)] + list(preds.groupby("period", sort=False)):
        for n in names:
            rows.append({
                "period": period, "model": n,
                "skill_vs_climatology": skill(g["y"], g[n], g["climatology"]),
                "skill_vs_climatology_12m": skill(g["y"], g[n], g["climatology_12m"]),
                "pr_auc": average_precision_score(g["y"], g[n]) if g["y"].any() else np.nan,
            })
    tests = []
    for a, b in [("logistic", "climatology_12m"), ("lgbm_reg", "climatology_12m"),
                 ("logistic", "calendar_logit"), ("logistic", "persistence_7d")]:
        mean, stat, p, n = diebold_mariano(daily_loss_gap(preds, a, b))
        tests.append({"model_a": a, "model_b": b, "days": n, "mean_brier_gap": mean, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows), pd.DataFrame(tests)


# 2. curtailment with weekends matched and week blocks

def with_ci(d, stat, reps, block, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in d.groupby("year"):
        keys = g[block].to_numpy()
        unique = pd.unique(keys)
        pos = {k: np.flatnonzero(keys == k) for k in unique}
        draws = []
        for _ in range(reps):
            pick = rng.choice(len(unique), size=len(unique), replace=True)
            draws.append(stat(g.iloc[np.concatenate([pos[unique[i]] for i in pick])])[0])
        draws = np.array(draws, dtype=float)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5]) if np.isfinite(draws).any() else (np.nan, np.nan)
        rows.append({"year": year, "gap": stat(g)[0], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("year")


def curtailment(data, raw, reps):
    df = prepare(data, raw)
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").astype(str)
    normal = df[df["price"] > 20]
    checks = {
        "price<=0 vs >0, cells + weekend/holiday": (df, ["month", "hour", "fc_bin", "offday"], "neg"),
        "placebo 20-50 vs >50, cells + weekend/holiday": (normal, ["month", "hour", "fc_bin", "offday"], "band_low"),
        "weekend vs weekday at price >20 (should be ~0)": (normal.assign(is_off=normal["offday"] == 1), ["month", "hour", "fc_bin"], "is_off"),
    }
    if "wx_bin" in df:
        checks["price<=0 vs >0, irradiance cells + weekend/holiday"] = (df.dropna(subset=["wx_bin"]), ["month", "hour", "wx_bin", "offday"], "neg")
    tables = []
    for name, (d, cell, treat) in checks.items():
        for block in ["date", "week"]:
            t = with_ci(d, lambda g, c=cell, tr=treat: gap(g, c, tr), reps, block)
            tables.append(t.assign(check=name, resampled=block))
    return pd.concat(tables)


# 4. battery: q30/q70 and within-day error correlation

def error_structure(preds):
    local = preds.index.tz_convert(TZ)
    day = local.tz_localize(None).normalize()
    err = preds["price"] - preds["q50"]
    rows = []
    for period, idx in [("all", preds.index)] + list(preds.groupby("period", sort=False).groups.items()):
        e, d = err.loc[idx], day[preds.index.get_indexer(idx)]
        daily = e.groupby(d).transform("mean")
        price_spread = preds.loc[idx, "price"].groupby(d).agg(lambda s: s.max() - s.min())
        fc_spread = preds.loc[idx, "q50"].groupby(d).agg(lambda s: s.max() - s.min())
        rows.append({
            "period": period,
            "share_of_error_variance_shared_by_the_day": float(daily.var() / e.var()),
            "mae_price_level": float(e.abs().mean()),
            "mae_daily_spread": float((price_spread - fc_spread).abs().mean()),
            "corr_forecast_vs_real_spread": float(np.corrcoef(price_spread, fc_spread)[0, 1]),
        })
    return pd.DataFrame(rows)


def battery_q30_q70(preds, power=1.0, energy=2.0, eff=0.88, deg=10.0, dt=0.25):
    local = preds.index.tz_convert(TZ)
    preds = preds.assign(day=local.tz_localize(None).normalize())
    rows = []
    for day, g in preds.groupby("day"):
        price = g["price"].to_numpy()
        plans = {
            "perfect foresight": (price, price),
            "median forecast": (g["q50"].to_numpy(), g["q50"].to_numpy()),
            "cautious q70 buy, q30 sell": (g["q70"].to_numpy(), g["q30"].to_numpy()),
            "cautious q90 buy, q10 sell": (g["q90"].to_numpy(), g["q10"].to_numpy()),
        }
        for name, (buy, sell) in plans.items():
            c, d = schedule(buy, sell, power, energy, eff, deg)
            rows.append({"day": day, "strategy": name, "profit": float(((d - c) * price).sum() * dt - deg * d.sum() * dt),
                         "idle": d.sum() * dt < 1e-6})
    daily = pd.DataFrame(rows)
    daily["period"] = period_label(pd.DatetimeIndex(daily["day"]).tz_localize(TZ))
    out = []
    for period, g in [("all", daily)] + list(daily.groupby("period", sort=False)):
        best = g.loc[g["strategy"] == "perfect foresight", "profit"].sum()
        for name, s in g.groupby("strategy", sort=False):
            out.append({"period": period, "strategy": name, "eur_per_day": s["profit"].mean(),
                        "share_of_perfect": s["profit"].sum() / best, "loss_days": int((s["profit"] < 0).sum()),
                        "idle_days": int(s["idle"].sum())})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--out", default="results")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--first-test", default="2024-01")
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--skip", nargs="*", default=[], choices=["control", "curtailment", "baselines", "battery"])
    a = p.parse_args()
    out = Path(a.out)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)

    df = add_cyclic(pd.read_parquet(a.data))
    df = add_nonpos_7d(df[df.index >= pd.Timestamp(a.train_start, tz=TZ) - pd.Timedelta(days=8)])
    df = df[df.index >= pd.Timestamp(a.train_start, tz=TZ)]
    wx = [c for c in df.columns if c.startswith("wx_")]
    exp_file = out / "experiment_predictions.parquet"
    if not exp_file.exists() and not {"control", "baselines"} <= set(a.skip):
        raise SystemExit("run experiments.py first, this reuses its monthly predictions")

    if "control" not in a.skip:
        print("\n=== 1. RQ2 control year: is losing skill by Q3 normal, or special to 2026? ===")
        exp = pd.read_parquet(exp_file)
        t = control_year(df, exp, {"logistic": logistic(FEATURES + CYCLIC + wx), "lgbm_reg": lgbm_reg(FEATURES + wx)})
        t.to_csv(out / "revision_control_year.csv", index=False)
        print("Brier skill vs climatology; gap = retrained minus frozen (bigger = frozen model fell further behind)")
        print(t.pivot_table(index=["run", "model"], columns="quarter", values=["frozen", "retrained", "gap"], sort=False).round(3).to_string())

    if "curtailment" not in a.skip:
        print("\n=== 2. curtailment with weekends/holidays in the matching, day and week resampling ===")
        t = curtailment(a.data, Path(a.raw), a.reps)
        t.to_csv(out / "revision_curtailment.csv")
        for (name, block), g in t.groupby(["check", "resampled"], sort=False):
            print(f"\n  {name}  [resampled by {block}]")
            print(g[["gap", "ci_low", "ci_high"]].round(3).to_string())

    if "baselines" not in a.skip:
        print("\n=== 3. stronger baselines ===")
        exp = pd.read_parquet(exp_file)
        m, dm = baselines(df, exp, a.first_test)
        m.to_csv(out / "revision_baselines.csv", index=False)
        dm.to_csv(out / "revision_baselines_dm.csv", index=False)
        order = list(dict.fromkeys(m["period"]))
        for col in ["skill_vs_climatology", "skill_vs_climatology_12m", "pr_auc"]:
            print(f"\n{col}")
            print(m.pivot(index="model", columns="period", values=col)[order].round(3).to_string())
        print("\nDiebold-Mariano (negative gap = model_a better)")
        print(dm.assign(mean_brier_gap=dm["mean_brier_gap"].map("{:.5f}".format), p_value=dm["p_value"].map("{:.3g}".format)).round(2).to_string(index=False))

    if "battery" not in a.skip:
        print("\n=== 4. battery: milder cautious strategy and within-day error correlation ===")
        saved = out / "price_forecasts.parquet"
        extra = out / "price_forecasts_q30_q70.parquet"
        if not saved.exists():
            raise SystemExit("run battery.py first, it saves the price forecasts")
        preds = pd.read_parquet(saved)
        if extra.exists():
            q = pd.read_parquet(extra)
        else:
            cols = FEATURES + wx
            print("training q30 and q70 price models (slow, saved for next time)")
            q = backtest_prices(df, {"q30": quantile_model(cols, 0.3), "q70": quantile_model(cols, 0.7)}, a.first_test)[["q30", "q70"]]
            q.to_parquet(extra)
        preds = preds.join(q, how="inner")
        qs = np.sort(preds[["q10", "q30", "q50", "q70", "q90"]].to_numpy(), axis=1)
        preds[["q10", "q30", "q50", "q70", "q90"]] = qs
        es = error_structure(preds)
        es.to_csv(out / "revision_error_structure.csv", index=False)
        print("how much of the forecast error is shared by the whole day (1 = all hours off by the same amount)")
        print(es.round(3).to_string(index=False))
        b = battery_q30_q70(preds)
        b.to_csv(out / "revision_battery_quantiles.csv", index=False)
        print("\nbattery, 1 MW / 2 MWh")
        print(b.round(3).to_string(index=False))

    print(f"\nsaved the revision_*.csv tables in {out}/")


if __name__ == "__main__":
    main()
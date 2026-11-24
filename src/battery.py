import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import linprog

import time

from baselines import FEATURES, TZ, period_label

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def quantile_model(cols, alpha):
    def fit_predict(train, test):
        model = LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            verbose=-1, random_state=0,
        )
        return model.fit(train[cols], train["price"]).predict(test[cols])
    return fit_predict


def naive(lag):
    def fit_predict(train, test):
        return test[lag].fillna(train["price"].median()).to_numpy()
    return fit_predict


def backtest_prices(df, models, first_test):
    ym = df.index.tz_convert(TZ).strftime("%Y-%m")
    out = []
    for m in sorted(x for x in set(ym) if x >= first_test):
        train, test = df[ym < m], df[ym == m]
        t0 = time.time()
        preds = test[["price"]].copy()
        for name, fit_predict in models.items():
            preds[name] = fit_predict(train, test)
        out.append(preds)
        print(f"{m}: {len(test):,} test rows ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(out)
    preds["period"] = period_label(preds.index)
    return preds


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def conformal(preds, days=90, level=0.8, min_days=30):
    # widen q10/q90 by the error spread of the previous `days` days of out-of-sample forecasts
    local = preds.index.tz_convert(TZ)
    day = pd.Series(local.tz_localize(None).normalize(), index=preds.index)
    month = local.strftime("%Y-%m")
    score = np.maximum(preds["q10"] - preds["price"], preds["price"] - preds["q90"])
    shift = pd.Series(0.0, index=preds.index)
    for m in sorted(set(month)):
        start = pd.Timestamp(f"{m}-01")
        past = score[(day >= start - pd.Timedelta(days=days)) & (day < start)]
        if day[(day >= start - pd.Timedelta(days=days)) & (day < start)].nunique() >= min_days:
            n = len(past)
            shift[month == m] = np.quantile(past, min(1.0, np.ceil((n + 1) * level) / n))
    return preds.assign(q10c=preds["q10"] - shift, q90c=preds["q90"] + shift, conformal_shift=shift)


def forecast_metrics(preds):
    rows = []
    for period, g in [("all", preds)] + list(preds.groupby("period", sort=False)):
        rows.append({
            "period": period,
            "mae_q50": float((g["price"] - g["q50"]).abs().mean()),
            "mae_naive_1d": float((g["price"] - g["naive_1d"]).abs().mean()),
            "mae_naive_7d": float((g["price"] - g["naive_7d"]).abs().mean()),
            "pinball_q10": pinball(g["price"], g["q10"], 0.1),
            "pinball_q90": pinball(g["price"], g["q90"], 0.9),
            "coverage_q10_q90": float(((g["price"] >= g["q10"]) & (g["price"] <= g["q90"])).mean()),
            "coverage_calibrated": float(((g["price"] >= g["q10c"]) & (g["price"] <= g["q90c"])).mean()),
            "mean_width_calibrated": float((g["q90c"] - g["q10c"]).mean()),
        })
    return pd.DataFrame(rows)


def schedule(buy, sell, power, energy, eff, deg, dt=0.25):
    n = len(buy)
    ec = ed = np.sqrt(eff)
    # variables: charge c (n), discharge d (n), state of charge s (n)
    cost = np.concatenate([buy * dt, -sell * dt + deg * dt, np.zeros(n)])
    a_eq = np.zeros((n, 3 * n))
    for t in range(n):
        a_eq[t, t] = -ec * dt
        a_eq[t, n + t] = dt / ed
        a_eq[t, 2 * n + t] = 1
        if t > 0:
            a_eq[t, 2 * n + t - 1] = -1
    a_ub = np.zeros((1, 3 * n))
    a_ub[0, n:2 * n] = dt
    bounds = [(0, power)] * (2 * n) + [(0, energy)] * n
    bounds[-1] = (0, 0)
    res = linprog(cost, A_ub=a_ub, b_ub=[energy], A_eq=a_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(n), np.zeros(n)
    return res.x[:n], res.x[n:2 * n]


def fixed_rule(hours, power, energy, eff, dt=0.25):
    ec = ed = np.sqrt(eff)
    charge_win, discharge_win = (hours >= 10) & (hours < 15), (hours >= 18) & (hours < 21)
    c, d = np.zeros(len(hours)), np.zeros(len(hours))
    if charge_win.any() and discharge_win.any():
        target = min(energy, power * discharge_win.sum() * dt / ed)
        c[charge_win] = min(power, target / ec / (charge_win.sum() * dt))
        stored = c.sum() * dt * ec
        d[discharge_win] = min(power, stored * ed / (discharge_win.sum() * dt))
    return c, d


def simulate(preds, power, energy, eff, deg, dt=0.25):
    local = preds.index.tz_convert(TZ)
    preds = preds.assign(day=local.tz_localize(None).normalize(), hour=local.hour)
    rows = []
    for day, g in preds.groupby("day"):
        price = g["price"].to_numpy()
        plans = {
            "perfect foresight": schedule(price, price, power, energy, eff, deg),
            "fixed schedule": fixed_rule(g["hour"].to_numpy(), power, energy, eff),
            "median forecast": schedule(g["q50"].to_numpy(), g["q50"].to_numpy(), power, energy, eff, deg),
            "cautious (buy at q90, sell at q10)": schedule(g["q90"].to_numpy(), g["q10"].to_numpy(), power, energy, eff, deg),
            "cautious, calibrated ranges": schedule(g["q90c"].to_numpy(), g["q10c"].to_numpy(), power, energy, eff, deg),
        }
        for name, (c, d) in plans.items():
            rows.append({
                "day": day, "strategy": name,
                "profit_eur": float(((d - c) * price).sum() * dt - deg * d.sum() * dt),
                "discharged_mwh": float(d.sum() * dt),
            })
    daily = pd.DataFrame(rows)
    daily["period"] = period_label(pd.DatetimeIndex(daily["day"]).tz_localize(TZ))
    return daily


def summarise(daily):
    rows = []
    for period, g in [("all", daily)] + list(daily.groupby("period", sort=False)):
        days = g["day"].nunique()
        best = g.loc[g["strategy"] == "perfect foresight", "profit_eur"].sum()
        for name, s in g.groupby("strategy", sort=False):
            rows.append({
                "period": period, "strategy": name, "days": days,
                "profit_eur": s["profit_eur"].sum(),
                "profit_eur_per_day": s["profit_eur"].sum() / days,
                "capture_of_perfect": s["profit_eur"].sum() / best if best else np.nan,
                "cycles_per_day": s["discharged_mwh"].sum() / days / 2,
            })
    return pd.DataFrame(rows)


def plot_cumulative(daily, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in daily.groupby("strategy", sort=False):
        g = g.sort_values("day")
        ax.plot(g["day"], g["profit_eur"].cumsum() / 1000, label=name)
    ax.set_ylabel("cumulative profit, thousand €")
    ax.set_title("1 MW / 2 MWh battery in the Hungarian day-ahead market")
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
    p.add_argument("--reuse", action="store_true", help="reuse saved price forecasts")
    p.add_argument("--power", type=float, default=1.0, help="MW")
    p.add_argument("--energy", type=float, default=2.0, help="MWh")
    p.add_argument("--eff", type=float, default=0.88, help="round-trip efficiency")
    p.add_argument("--deg", type=float, default=10.0, help="degradation cost, EUR per MWh discharged")
    p.add_argument("--conformal-days", type=int, default=90, help="days of past forecast errors used to widen the ranges")
    a = p.parse_args()

    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    saved = out / "price_forecasts.parquet"
    pd.set_option("display.width", 140)

    if a.reuse and saved.exists():
        preds = pd.read_parquet(saved)
    else:
        df = pd.read_parquet(a.data)
        df = df[df.index >= pd.Timestamp(a.train_start, tz=TZ)]
        cols = FEATURES + [c for c in df.columns if c.startswith("wx_")]
        models = {name: quantile_model(cols, alpha) for name, alpha in QUANTILES.items()}
        models |= {"naive_1d": naive("price_lag1d"), "naive_7d": naive("price_lag7d")}
        print("price forecasts, monthly retraining")
        preds = backtest_prices(df, models, a.first_test)
        q = np.sort(preds[list(QUANTILES)].to_numpy(), axis=1)
        preds[list(QUANTILES)] = q
        preds.to_parquet(saved)

    preds = conformal(preds, a.conformal_days)
    fm = forecast_metrics(preds)
    fm.to_csv(out / "price_forecast_metrics.csv", index=False)
    print("\nprice forecast accuracy (EUR/MWh)")
    print(fm.round(3).to_string(index=False))

    print(f"\nbattery: {a.power} MW / {a.energy} MWh, round trip {a.eff:.0%}, degradation {a.deg} EUR/MWh, max one cycle a day")
    daily = simulate(preds, a.power, a.energy, a.eff, a.deg)
    summary = summarise(daily)
    tag = f"{a.power:g}MW_{a.energy:g}MWh_deg{a.deg:g}"
    daily.to_csv(out / f"battery_daily_{tag}.csv", index=False)
    summary.to_csv(out / f"battery_summary_{tag}.csv", index=False)
    plot_cumulative(daily, figs / f"battery_cumulative_{tag}.png")

    t = summary.pivot(index="strategy", columns="period", values="capture_of_perfect")
    t = t[list(dict.fromkeys(summary["period"]))].loc[list(dict.fromkeys(summary["strategy"]))]
    print("\nshare of the perfect-foresight profit")
    print(t.round(3).to_string())
    t = summary.pivot(index="strategy", columns="period", values="profit_eur_per_day")
    t = t[list(dict.fromkeys(summary["period"]))].loc[list(dict.fromkeys(summary["strategy"]))]
    print("\nprofit per day, EUR")
    print(t.round(0).to_string())
    print(f"\nsaved {out}/battery_summary_{tag}.csv and {figs}/battery_cumulative_{tag}.png")


if __name__ == "__main__":
    main()
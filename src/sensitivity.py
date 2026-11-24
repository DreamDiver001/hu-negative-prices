import argparse
import time
from pathlib import Path

import pandas as pd

from battery import conformal, simulate, summarise

BASE = {"power": 1.0, "energy": 2.0, "eff": 0.88, "deg": 10.0}
RUNS = {
    "baseline (1 MW / 2 MWh, 88%, 10 €/MWh)": {},
    "no degradation cost": {"deg": 0.0},
    "degradation 20 €/MWh": {"deg": 20.0},
    "1 hour battery (1 MWh)": {"energy": 1.0},
    "4 hour battery (4 MWh)": {"energy": 4.0},
    "round trip 80%": {"eff": 0.80},
    "round trip 92%": {"eff": 0.92},
}
SHOW = ["fixed schedule", "median forecast", "cautious (buy at q90, sell at q10)", "cautious, calibrated ranges"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecasts", default="results/price_forecasts.parquet")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if not Path(a.forecasts).exists():
        raise SystemExit("run battery.py first, it saves the price forecasts")
    preds = conformal(pd.read_parquet(a.forecasts))

    rows = []
    for name, change in RUNS.items():
        cfg = BASE | change
        t0 = time.time()
        s = summarise(simulate(preds, **cfg))
        s = s[s["period"] == "all"].set_index("strategy")
        row = {"run": name, "perfect €/day": s.loc["perfect foresight", "profit_eur_per_day"]}
        for strat in SHOW:
            row[f"{strat} share"] = s.loc[strat, "capture_of_perfect"]
        row["median forecast €/day"] = s.loc["median forecast", "profit_eur_per_day"]
        row["forecast gain over fixed €/day"] = row["median forecast €/day"] - s.loc["fixed schedule", "profit_eur_per_day"]
        rows.append(row)
        print(f"done: {name} ({time.time() - t0:.0f}s)", flush=True)

    table = pd.DataFrame(rows).set_index("run")
    table.to_csv(Path(a.out) / "battery_sensitivity.csv")
    pd.set_option("display.width", 200)
    shares = table[[c for c in table if c.endswith("share")]].rename(columns=lambda c: c.replace(" share", ""))
    print("\nshare of the perfect-foresight profit, January 2024 to now")
    print(shares.round(3).to_string())
    print("\nEUR per day")
    print(table[["perfect €/day", "median forecast €/day", "forecast gain over fixed €/day"]].round(0).to_string())
    print(f"\nsaved {a.out}/battery_sensitivity.csv")


if __name__ == "__main__":
    main()
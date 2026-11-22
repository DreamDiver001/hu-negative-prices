import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TZ = "Europe/Budapest"


def load_prices(folder):
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no price files in {folder}, run collect_entsoe.py first")
    s = pd.concat(pd.read_parquet(f).iloc[:, 0] for f in files)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.tz_convert(TZ).rename("price")


def with_duration(prices):
    step = prices.index.to_series().diff().shift(-1)
    step = step.where(step <= pd.Timedelta("1h"), pd.Timedelta("1h")).fillna(pd.Timedelta("15min"))
    return pd.DataFrame({"price": prices, "hours": step.dt.total_seconds() / 3600})


def yearly_table(df):
    df = df.assign(year=df.index.year, nonpos=df["price"] <= 0, neg=df["price"] < 0)
    return df.groupby("year").apply(
        lambda g: pd.Series({
            "hours_negative": (g["hours"] * g["neg"]).sum(),
            "hours_zero_or_negative": (g["hours"] * g["nonpos"]).sum(),
            "min_price": g["price"].min(),
            "mean_price": (g["price"] * g["hours"]).sum() / g["hours"].sum(),
        }),
        include_groups=False,
    ).round(1)


def heatmap(df, path, months=12):
    recent = df[df.index >= df.index.max() - pd.DateOffset(months=months)]
    recent = recent.assign(month=recent.index.strftime("%Y-%m"), hour=recent.index.hour)
    recent["nonpos_hours"] = recent["hours"] * (recent["price"] <= 0)
    share = recent.groupby(["month", "hour"])[["nonpos_hours", "hours"]].sum()
    grid = (share["nonpos_hours"] / share["hours"]).unstack("hour").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(24), range(24))
    ax.set_yticks(range(len(grid)), grid.index)
    ax.set_xlabel("hour of day (Budapest time)")
    ax.set_title("Share of time with day-ahead price ≤ 0 €/MWh, Hungary")
    fig.colorbar(im, ax=ax, label="share of the hour")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/raw/entsoe/prices_HU")
    p.add_argument("--out", default="figures")
    a = p.parse_args()

    df = with_duration(load_prices(a.prices))
    Path(a.out).mkdir(exist_ok=True)
    table = yearly_table(df)
    print(table.to_string())
    table.to_csv(Path(a.out) / "negative_hours_by_year.csv")
    heatmap(df, Path(a.out) / "negative_price_heatmap.png")
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()

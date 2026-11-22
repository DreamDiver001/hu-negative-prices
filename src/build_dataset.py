import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Budapest"


def read_folder(folder):
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no files in {folder}, run collect_entsoe.py --only core first")
    df = pd.concat(pd.read_parquet(f) for f in files)
    return df[~df.index.duplicated(keep="last")].sort_index()


def on_grid(df, grid):
    # hourly values hold for the whole hour, so fill at most the next three quarters
    return df.reindex(df.index.union(grid)).ffill(limit=3).reindex(grid)


def first_match(columns, word):
    hits = [c for c in columns if word.lower() in c.lower()]
    return hits[0] if hits else None


def build(raw, weather_dir=None):
    price = read_folder(raw / "prices_HU").iloc[:, 0]
    grid = pd.date_range(price.index.min(), price.index.max() + pd.Timedelta("45min"), freq="15min")
    df = pd.DataFrame(index=grid)
    df.index.name = "time_utc"
    df["price"] = on_grid(price.to_frame(), grid).iloc[:, 0]

    load = read_folder(raw / "load_HU")
    df["load_fc"] = on_grid(load[[first_match(load.columns, "forecast")]], grid).iloc[:, 0]

    ws = read_folder(raw / "wind_solar_forecast_HU")
    solar_col = first_match(ws.columns, "solar")
    wind_cols = [c for c in ws.columns if c.lower().startswith("wind")]
    df["solar_fc"] = on_grid(ws[[solar_col]], grid).iloc[:, 0] if solar_col else np.nan
    df["wind_fc"] = on_grid(ws[wind_cols], grid).sum(axis=1, min_count=1) if wind_cols else 0.0

    df["residual_fc"] = df["load_fc"] - df["solar_fc"] - df["wind_fc"].fillna(0)
    df["solar_share_fc"] = df["solar_fc"] / df["load_fc"]

    local = df.index.tz_convert(TZ)
    day = pd.DatetimeIndex(local.tz_localize(None).normalize())
    hu = holidays.Hungary(years=range(local.year.min(), local.year.max() + 1))
    df["qh"] = local.hour * 4 + local.minute // 15
    df["hour"] = local.hour
    df["weekday"] = local.weekday
    df["month"] = local.month
    df["doy"] = local.dayofyear
    df["holiday"] = pd.Index(day.date).isin(list(hu)).astype(int)
    df["offday"] = ((df["weekday"] >= 5) | (df["holiday"] == 1)).astype(int)

    by_day = df.groupby(day)
    df["solar_fc_daymax"] = by_day["solar_fc"].transform("max")
    df["residual_fc_daymin"] = by_day["residual_fc"].transform("min")

    p = df["price"]
    df["price_lag1d"] = p.reindex(df.index - pd.Timedelta(days=1)).to_numpy()
    df["price_lag7d"] = p.reindex(df.index - pd.Timedelta(days=7)).to_numpy()
    daily = pd.DataFrame({
        "price_min": p.groupby(day).min(),
        "price_mean": p.groupby(day).mean(),
        "nonpos_share": (p <= 0).groupby(day).mean(),
    })
    prev = daily.reindex(day - pd.Timedelta(days=1))
    for col in daily:
        df[f"{col}_prevday"] = prev[col].to_numpy()

    if weather_dir and Path(weather_dir).exists():
        files = sorted(Path(weather_dir).glob("*.parquet"))
        if files:
            wx = pd.concat(pd.read_parquet(f) for f in files)
            wx_all = wx.groupby(level=0).mean().add_prefix("wx_")
            if "shortwave_radiation" in wx:
                wx_all["wx_spread_radiation"] = wx["shortwave_radiation"].groupby(level=0).std()
            df = df.join(on_grid(wx_all, grid))

    df = df[df["price"].notna()]
    df["y"] = (df["price"] <= 0).astype(int)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--weather", default="data/raw/weather/dayahead")
    p.add_argument("--out", default="data/processed/features.parquet")
    a = p.parse_args()

    df = build(Path(a.raw), a.weather)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)

    local = df.index.tz_convert(TZ)
    print(f"{len(df):,} rows, {local.min():%Y-%m-%d} to {local.max():%Y-%m-%d}, {df.shape[1]} columns")
    print(f"share of 15-min periods with price <= 0: {df['y'].mean():.3%}")
    missing = df.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("\nmissing values (share of rows):")
        print(missing.round(3).to_string())
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()

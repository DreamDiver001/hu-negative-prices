import argparse
import time
from pathlib import Path

import pandas as pd
import requests

LOCATIONS = {
    "budapest": (47.50, 19.04),
    "kecskemet": (46.91, 19.69),
    "szeged": (46.25, 20.15),
    "debrecen": (47.53, 21.63),
    "nyiregyhaza": (47.95, 21.72),
    "miskolc": (48.10, 20.78),
    "pecs": (46.07, 18.23),
    "paks": (46.62, 18.86),
    "gyor": (47.69, 17.63),
}

VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_speed_100m",
]

# stitched: first hours of each run, close to observed weather, only for analysis
# dayahead: forecast made one day earlier, safe to use as a model feature (from 2024)
MODES = {
    "stitched": ("https://historical-forecast-api.open-meteo.com/v1/forecast", "", "2022-01-01"),
    "dayahead": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1", "2024-01-01"),
}


def fetch(url, params, session, tries=5):
    for i in range(tries):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def to_frame(payload, suffix):
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").rename_axis("time_utc")
    if suffix:
        df.columns = [c.removesuffix(suffix) for c in df.columns]
    return df


def run(mode, start, end, out_dir, session=None, pause=1.0):
    url, suffix, earliest = MODES[mode]
    start = max(pd.Timestamp(start), pd.Timestamp(earliest))
    end = pd.Timestamp(end)
    session = session or requests.Session()
    out_dir = Path(out_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.now().year

    for loc, (lat, lon) in LOCATIONS.items():
        for year in range(start.year, end.year + 1):
            target = out_dir / f"{loc}_{year}.parquet"
            if target.exists() and year != this_year:
                continue
            s = max(start, pd.Timestamp(f"{year}-01-01"))
            e = min(end, pd.Timestamp(f"{year}-12-31"))
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": s.strftime("%Y-%m-%d"),
                "end_date": e.strftime("%Y-%m-%d"),
                "hourly": ",".join(v + suffix for v in VARIABLES),
                "timezone": "GMT",
            }
            try:
                to_frame(fetch(url, params, session), suffix).to_parquet(target)
                print(f"ok    {mode} {loc} {year}")
            except Exception as exc:
                print(f"FAIL  {mode} {loc} {year}: {exc!r}"[:240])
            time.sleep(pause)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, default="dayahead")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    p.add_argument("--out", default="data/raw/weather")
    a = p.parse_args()
    run(a.mode, a.start, a.end, a.out)


if __name__ == "__main__":
    main()

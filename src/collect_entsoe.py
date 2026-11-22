import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

TZ = "Europe/Budapest"
PRICE_ZONES = ["HU", "AT", "SK", "RO", "SI", "HR", "DE_LU"]
FLOW_NEIGHBOURS = ["AT", "SK", "RO", "HR", "SI", "RS", "UA"]
_lock = threading.Lock()
CORE = ["prices_HU", "load_HU", "wind_solar_forecast_HU", "generation_HU"]


def log(msg):
    with _lock:
        print(msg, flush=True)


def datasets():
    jobs = {}
    for zone in PRICE_ZONES:
        jobs[f"prices_{zone}"] = lambda c, s, e, z=zone: c.query_day_ahead_prices(z, start=s, end=e)
    jobs["load_HU"] = lambda c, s, e: c.query_load_and_forecast("HU", start=s, end=e)
    jobs["wind_solar_forecast_HU"] = lambda c, s, e: c.query_wind_and_solar_forecast("HU", start=s, end=e)
    jobs["generation_HU"] = lambda c, s, e: c.query_generation("HU", start=s, end=e)
    for n in FLOW_NEIGHBOURS:
        jobs[f"flow_HU_{n}"] = lambda c, s, e, n=n: c.query_crossborder_flows("HU", n, start=s, end=e)
        jobs[f"flow_{n}_HU"] = lambda c, s, e, n=n: c.query_crossborder_flows(n, "HU", start=s, end=e)
    return jobs


def to_frame(obj, name):
    df = obj.to_frame(name) if isinstance(obj, pd.Series) else obj.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(p) for p in col if p) for col in df.columns]
    df.columns = [str(c) for c in df.columns]
    df.index = df.index.tz_convert("UTC")
    df.index.name = "time_utc"
    return df


def windows(start, end, chunk):
    start = pd.Timestamp(start, tz=TZ)
    stop = min(pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1), pd.Timestamp.now(tz=TZ))
    freq, fmt = ("YS", "%Y") if chunk == "year" else ("MS", "%Y-%m")
    for s in pd.date_range(start.normalize().replace(day=1), stop, freq=freq, tz=TZ):
        full_end = s + (pd.offsets.YearBegin(1) if chunk == "year" else pd.offsets.MonthBegin(1))
        s2, e = max(s, start), min(full_end, stop)
        if s2 < e:
            yield s.strftime(fmt), s2, e, s2 == s and e == full_end


def already_done(folder, label):
    def have(x):
        return (folder / f"{x}.parquet").exists() or (folder / f"{x}.empty").exists()
    if have(label):
        return True
    if len(label) == 4:
        return all(have(f"{label}-{m:02d}") for m in range(1, 13))
    return have(label[:4])


def download(name, fetch, key, start, end, out_dir, chunk):
    client = EntsoePandasClient(api_key=key, retry_count=3, retry_delay=10, timeout=120)
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    failed = []
    for label, s, e, complete in windows(start, end, chunk):
        if complete and already_done(folder, label):
            continue
        stem = folder / (label if complete else f"{label}-partial")
        t0 = time.time()
        try:
            to_frame(fetch(client, s, e), name).to_parquet(stem.with_suffix(".parquet"))
            if complete:
                (folder / f"{label}-partial.parquet").unlink(missing_ok=True)
            log(f"ok    {name} {label} ({time.time() - t0:.0f}s)")
        except NoMatchingDataError:
            if complete:
                stem.with_suffix(".empty").touch()
            log(f"empty {name} {label}")
        except Exception as exc:
            failed.append((name, label))
            log(f"FAIL  {name} {label}: {exc!r}"[:240])
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d"))
    p.add_argument("--out", default="data/raw/entsoe")
    p.add_argument("--only", nargs="*", help="dataset names, or 'core' for prices, load, forecasts, generation")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--chunk", choices=["month", "year"], default="month")
    a = p.parse_args()

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("set ENTSOE_API_KEY first (see .env.example)")

    jobs = datasets()
    if a.only:
        wanted = set(CORE if a.only == ["core"] else a.only)
        jobs = {k: v for k, v in jobs.items() if k in wanted}

    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(download, n, f, key, a.start, a.end, Path(a.out), a.chunk) for n, f in jobs.items()]
        for fut in as_completed(futures):
            failed += fut.result()

    if failed:
        print(f"\n{len(failed)} downloads failed, run the same command again to retry them:")
        for name, label in failed:
            print(f"  {name} {label}")
    else:
        print("\nall done")


if __name__ == "__main__":
    main()
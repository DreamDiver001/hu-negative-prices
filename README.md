# Negative electricity prices in Hungary

Forecasting when Hungarian day-ahead electricity prices go to zero or below, and measuring what those forecasts are worth to a battery.

## Setup

Needs Python 3.12 or newer (entsoe-py 1.0 requires it).

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ENTSO-E token:

1. Register at https://transparency.entsoe.eu
2. Email transparency@entsoe.eu with the subject "Restful API access" and your account email in the body
3. Once approved, generate the token under My Account Settings
4. Copy `.env.example` to `.env`, paste the token, then `export ENTSOE_API_KEY=...` (PowerShell: `$env:ENTSOE_API_KEY="..."`)

Open-Meteo needs no key.

## Getting the data

```
python src/collect_entsoe.py --start 2020-01-01 --only core
python src/collect_entsoe.py --start 2020-01-01
python src/collect_weather.py --mode dayahead --start 2024-01-01
python src/collect_weather.py --mode stitched --start 2022-01-01
python src/first_look.py
```

## Features and baselines

```
python src/build_dataset.py
python src/baselines.py
```

`build_dataset.py` merges the core ENTSO-E data, and the `dayahead` weather if it is there, into one table at 15-minute resolution (`data/processed/features.parquet`). Before October 2025 the hourly price is repeated over its four quarters. `y` is 1 when the price is at or below 0 €/MWh.

`baselines.py` retrains every model at the start of each month from 2024 on, using everything since 2023, and forecasts that month. It prints Brier skill (against climatology) and PR-AUC for 2024, January to September 2025, and the 15-minute period, and saves the predictions, metrics, LightGBM feature importance and a reliability plot. Expect a few minutes.

`experiments.py` runs the same monthly backtest with a regularised LightGBM (monotone in residual load and solar share), a version of it calibrated on the last 6 months of training data, and the RQ2 transfer test: models trained once on data before 1 October 2025 against the monthly retrained ones, by quarter after the switch. It also reports how often the four quarters of an hour disagree on the sign of the price.

`drift_analysis.py` looks at what changed in 2026. For each year it finds the residual load at which the chance of a price at or below 0 reaches 50%, and compares actual solar output with the day-ahead solar forecast at different price levels. If output falls well below the forecast only when prices are negative, solar plants are switching off.

`curtailment_checks.py` tests that reading three ways: the output gap at the same month, hour and forecast level (and, from 2024, the same irradiance), with a placebo between two normal price bands (20–50 and above 50 €/MWh) that should show no gap; the solar energy missing at prices at or below 0 in GWh per year; and the tipping point fitted separately for spring, summer and autumn (only where a season has at least 100 such periods). The 95% intervals come from resampling whole days 200 times (`--reps`).

`--only core` gets the Hungarian prices, load, wind/solar forecasts and generation first; the second command adds neighbour prices and border flows. Downloads run 3 datasets at a time (`--workers`), one month per request (`--chunk year` is faster but ENTSO-E often answers it with HTTP 599).

Both collectors save one parquet file per month (ENTSO-E) or per location and year (weather) under `data/raw/`. Finished files are skipped on the next run, so if something fails or you stop it halfway, run the same command again. The current month or year is always fetched again.

`first_look.py` prints the number of negative-price hours per year and saves a month × hour heatmap to `figures/`.

## Data notes

- Timestamps are stored in UTC. Convert to Europe/Budapest for anything about time of day.
- Day-ahead prices are hourly until 30 Sep 2025 and 15-minute from 1 Oct 2025, when the European day-ahead market switched to a 15-minute market time unit.
- `wind_solar_forecast_HU` and the forecast column in `load_HU` are published the day before, so they can be used as features. One catch: the rules only require the wind and solar forecast by 18:00 the day before, after the 12:00 auction. Worth a robustness run without it, using the weather forecast instead. `generation_HU` and actual load are only known afterwards, so use them only for analysis.
- Weather `dayahead` mode is the forecast made one day earlier (available from 2024). `stitched` mode is close to the observed weather, so using it as a feature would leak information from the future.

## Battery (RQ3)

```
python src/battery.py
python src/battery.py --reuse --deg 0
python src/battery.py --reuse --energy 4
```

First it forecasts the price for every period as three quantiles (10%, 50%, 90%) with LightGBM, retrained each month, and compares the median with two naive forecasts (the price 1 and 7 days earlier). The forecasts are saved, so `--reuse` skips this slow step when you only change the battery.

Then it runs a 1 MW / 2 MWh battery (88% round trip, at most one full cycle a day, empty at midnight, 10 €/MWh degradation cost) through every test day with four strategies: perfect foresight (the upper bound), a fixed schedule (charge 10:00-15:00, discharge 18:00-21:00), a schedule optimised on the median forecast, and a cautious one that plans with the 90% quantile as the buying price and the 10% quantile as the selling price. Each day is a small linear program. Profit is always counted at the real prices.
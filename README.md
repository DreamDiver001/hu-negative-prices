# Negative electricity prices in Hungary

Forecasting when Hungarian day-ahead electricity prices go to zero or below, and measuring what those forecasts are worth to a battery.

## Setup

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
python src/collect_entsoe.py --start 2020-01-01
python src/collect_weather.py --mode dayahead --start 2024-01-01
python src/collect_weather.py --mode stitched --start 2022-01-01
python src/first_look.py
```

Both collectors save one parquet file per month (ENTSO-E) or per location and year (weather) under `data/raw/`. Finished files are skipped on the next run, so if something fails or you stop it halfway, run the same command again. The current month or year is always fetched again.

`first_look.py` prints the number of negative-price hours per year and saves a month × hour heatmap to `figures/`.

## Data notes

- Timestamps are stored in UTC. Convert to Europe/Budapest for anything about time of day.
- Day-ahead prices are hourly until 30 Sep 2025 and 15-minute from 1 Oct 2025, when the European day-ahead market switched to a 15-minute market time unit.
- `wind_solar_forecast_HU` and the forecast column in `load_HU` are published the day before, so they can be used as features. `generation_HU` and actual load are only known afterwards, so use them only for analysis.
- Weather `dayahead` mode is the forecast made one day earlier (available from 2024). `stitched` mode is close to the observed weather, so using it as a feature would leak information from the future.

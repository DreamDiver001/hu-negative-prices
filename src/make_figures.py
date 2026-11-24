import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "perfect foresight": "#2a78d6",
    "fixed schedule": "#eb6834",
    "median forecast": "#1baf7a",
    "cautious (buy at q90, sell at q10)": "#eda100",
}
LABELS = {
    "perfect foresight": "Perfect foresight",
    "fixed schedule": "Fixed schedule",
    "median forecast": "Median forecast",
    "cautious (buy at q90, sell at q10)": "Cautious (q90 buy, q10 sell)",
}
INK, MUTED, GRID, SHADE = "#2b2b29", "#6b6b66", "#e6e5e0", "#f1f0ec"
YEAR_COLORS = {2023: "#86b6ef", 2024: "#3987e5", 2025: "#1c5cab", 2026: "#0d366b"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MODEL_COLORS = {"logistic": "#2a78d6", "lightgbm": "#eb6834", "lgbm_reg": "#1baf7a"}
MODEL_LABELS = {"logistic": "Logistic regression", "lightgbm": "LightGBM, default", "lgbm_reg": "LightGBM, regularised"}


def winters(start, end):
    for year in range(start.year - 1, end.year + 1):
        a, b = pd.Timestamp(f"{year}-11-01"), pd.Timestamp(f"{year + 1}-03-01")
        if b > start and a < end:
            yield max(a, start), min(b, end)


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def battery_figure(daily_csv, out):
    d = pd.read_csv(daily_csv, parse_dates=["day"])
    wide = d.pivot(index="day", columns="strategy", values="profit_eur").sort_index()
    order = [s for s in COLORS if s in wide]
    start, end = wide.index.min(), wide.index.max() + pd.Timedelta(days=1)

    monthly = wide.groupby(wide.index.to_period("M")).sum()
    capture = monthly.div(monthly["perfect foresight"], axis=0) * 100
    capture.index = capture.index.to_timestamp() + pd.Timedelta(days=14)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.28})
    for ax in (top, bottom):
        style(ax)
        for i, (a, b) in enumerate(winters(start, end)):
            ax.axvspan(a, b, color=SHADE, zorder=0, lw=0)

    cum = wide[order].cumsum() / 1000
    for s in order:
        top.plot(cum.index, cum[s], color=COLORS[s], lw=2, label=LABELS[s])
        top.annotate(f"€{cum[s].iloc[-1]:.0f}k", (cum.index[-1], cum[s].iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK)
    top.set_ylabel("cumulative profit, thousand €", color=INK, fontsize=9.5)
    top.set_title("Cumulative profit", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    top.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    top.text(pd.Timestamp("2024-12-31"), top.get_ylim()[1] * 0.97, "Nov–Feb", ha="center", va="top", fontsize=8.5, color=MUTED)

    for s in [x for x in order if x != "perfect foresight"]:
        bottom.plot(capture.index, capture[s], color=COLORS[s], lw=2, marker="o", markersize=4, label=LABELS[s])
    bottom.set_ylim(min(-5, capture[[x for x in order if x != "perfect foresight"]].min().min() - 4), 105)
    bottom.set_yticks([0, 25, 50, 75, 100])
    bottom.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    bottom.set_ylabel("share of perfect-foresight profit", color=INK, fontsize=9.5)
    bottom.set_title("Share of the maximum profit, by month", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    bottom.set_xlim(start, end + pd.Timedelta(days=55))

    fig.suptitle("A forecast-driven battery keeps most of the possible profit all year;\na fixed solar-hours schedule collapses in winter",
                 x=0.075, ha="left", fontsize=12.5, color=INK, fontweight="bold", y=0.995)
    fig.text(0.075, 0.012, "1 MW / 2 MWh, 88% round trip, at most one cycle a day, 10 €/MWh degradation, Hungarian day-ahead prices,\n"
             f"{start:%d %b %Y} – {wide.index.max():%d %b %Y}. Shaded: November–February. Colours as in the top panel.", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.9, top=0.89, bottom=0.11)
    return save(fig, out)


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def reliability_figure(pred_path, out):
    preds = pd.read_parquet(pred_path)
    names = [n for n in MODEL_COLORS if n in preds]
    bins = np.linspace(0, 1, 11)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.8, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    for ax in (top, bottom):
        style(ax)
    top.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfectly calibrated")
    for n in names:
        b = pd.cut(preds[n], bins, include_lowest=True)
        g = preds.groupby(b, observed=True).agg(p=(n, "mean"), obs=("y", "mean"), size=("y", "size"))
        top.plot(g["p"], g["obs"], color=MODEL_COLORS[n], lw=2, marker="o", markersize=5, label=MODEL_LABELS[n])
        bottom.plot(bins[:-1] + 0.05, g["size"].reindex(b.cat.categories, fill_value=0).to_numpy(),
                    color=MODEL_COLORS[n], lw=2, marker="o", markersize=4)
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_ylabel("observed share with price ≤ 0", color=INK, fontsize=9.5)
    top.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    top.set_title("Predicted against observed, all test months", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    bottom.set_yscale("log")
    bottom.set_ylabel("periods", color=INK, fontsize=9.5)
    bottom.set_xlabel("predicted probability of price ≤ 0", color=INK, fontsize=9.5)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.09)
    return save(fig, out)


def heatmap_years(prices_folder, out, years=(2023, 2024, 2025, 2026)):
    from first_look import load_prices, with_duration
    df = with_duration(load_prices(prices_folder))
    df = df[df.index.year.isin(years)]
    df = df.assign(year=df.index.year, month=df.index.month, hour=df.index.hour,
                   nonpos=df["hours"] * (df["price"] <= 0))
    share = df.groupby(["year", "month", "hour"])[["nonpos", "hours"]].sum()
    share = share["nonpos"] / share["hours"]
    fig, axes = plt.subplots(1, len(years), figsize=(11, 4.4), sharey=True)
    for ax, year in zip(axes, years):
        grid = share.loc[year].unstack("hour").reindex(index=range(1, 13), columns=range(24))
        im = ax.imshow(grid.values, aspect="auto", cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{year}" + (f", to {df.index.max():%d %b}" if year == df.index.max().year else ""),
                     loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("hour (Budapest time)", color=INK, fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.13, wspace=0.08)
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.13, 0.012, 0.77]))
    cb.set_label("share of the hour with price ≤ 0", color=INK, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5)
    return save(fig, out)


def drift_figures(features, raw, tip_out, ratio_out):
    from drift_analysis import daylight, solar_ratio, tipping_point
    df = daylight(pd.read_parquet(features))
    _, curves = tipping_point(df)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style(ax)
    for year, c in curves.items():
        ax.plot(c.index / 1000, c.values, color=YEAR_COLORS.get(year, MUTED), lw=2, label=str(year))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("residual load forecast, GW (load minus wind minus solar)", color=INK, fontsize=9.5)
    ax.set_ylabel("share of periods with price ≤ 0", color=INK, fontsize=9.5)
    ax.set_title("Price ≤ 0 against residual load, 9:00–17:00", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="year", title_fontsize=9)
    fig.tight_layout()
    tip = save(fig, tip_out)

    table, _ = solar_ratio(df, Path(raw))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style(ax)
    x = range(len(table.columns))
    for year, row in table.iterrows():
        ax.plot(x, row.values, color=YEAR_COLORS.get(year, MUTED), lw=2, marker="o", markersize=5, label=str(year))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="year", title_fontsize=9, loc="lower right")
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(list(x), table.columns)
    ax.set_xlabel("day-ahead price, €/MWh", color=INK, fontsize=9.5)
    ax.set_ylabel("actual / day-ahead forecast solar (median)", color=INK, fontsize=9.5)
    ax.set_title("Solar output against its forecast, sunny daylight periods", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.tight_layout()
    return tip, save(fig, ratio_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--battery", default="results/battery_daily_1MW_2MWh_deg10.csv")
    p.add_argument("--predictions", default="results/experiment_predictions.parquet")
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--raw", default="data/raw/entsoe")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()
    figs = Path(a.figures)
    jobs = [
        (a.battery, lambda: battery_figure(a.battery, figs / "paper_battery")),
        (a.predictions, lambda: reliability_figure(a.predictions, figs / "paper_reliability")),
        (Path(a.raw) / "prices_HU", lambda: heatmap_years(Path(a.raw) / "prices_HU", figs / "paper_heatmap_years")),
        (a.features, lambda: drift_figures(a.features, a.raw, figs / "paper_tipping_point", figs / "paper_solar_ratio")),
    ]
    for source, job in jobs:
        if not Path(source).exists():
            print(f"skipped, {source} not found")
            continue
        print("saved", job())


if __name__ == "__main__":
    main()
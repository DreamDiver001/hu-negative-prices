import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    return out.with_suffix(".png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--battery", default="results/battery_daily_1MW_2MWh_deg10.csv")
    p.add_argument("--out", default="figures/paper_battery")
    a = p.parse_args()
    print(f"saved {battery_figure(a.battery, Path(a.out))}")


if __name__ == "__main__":
    main()
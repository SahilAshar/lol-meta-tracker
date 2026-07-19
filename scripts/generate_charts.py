"""Generate cross-league meta charts from processed data.

Produces (to charts/):
  pick_rate_by_region.png    top champions' pick rates, grouped by league
  ban_rate_by_region.png     top champions' ban rates, grouped by league
  week_over_week_delta.png   biggest pick-rate movers, latest week vs prior
                             (skipped until the window has 2+ weeks of data)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import CHARTS_DIR, DATA_PROCESSED, MAJOR_LEAGUES

TOP_N = 12
LEAGUE_PALETTE = {"LCK": "#1f77b4", "LPL": "#d62728", "LEC": "#2ca02c", "LCS": "#9467bd"}

sns.set_theme(style="whitegrid", context="talk")


def _load(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/analyze_meta.py first")
    return pd.read_csv(path)


def _grouped_rate_chart(df: pd.DataFrame, rate_col: str, title: str, outfile: str) -> None:
    top = (
        df.groupby("champion")[rate_col].mean().nlargest(TOP_N).index.tolist()
    )
    plot_df = df[df["champion"].isin(top)].copy()
    plot_df["champion"] = pd.Categorical(plot_df["champion"], categories=top, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(
        data=plot_df.sort_values("champion"),
        y="champion", x=rate_col, hue="league",
        hue_order=[lg for lg in MAJOR_LEAGUES if lg in plot_df["league"].unique()],
        palette=LEAGUE_PALETTE, ax=ax,
    )
    ax.set_title(title, pad=16)
    ax.set_xlabel(rate_col.replace("_", " ").title())
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="", loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / outfile, dpi=150)
    plt.close(fig)
    print(f"Wrote charts/{outfile}")


def week_over_week_chart(weekly: pd.DataFrame) -> bool:
    """Pick-rate movers between the two most recent weeks. Returns False if <2 weeks."""
    weeks = sorted(weekly["iso_week"].unique())
    if len(weeks) < 2:
        print("Skipping week-over-week chart: need 2+ weeks of data "
              f"(have {len(weeks)})")
        return False
    prev_w, last_w = weeks[-2], weeks[-1]

    def rates_for(week: str) -> pd.Series:
        sub = weekly[weekly["iso_week"] == week]
        return sub.groupby("champion")["pick_rate"].mean()

    delta = (rates_for(last_w) - rates_for(prev_w).reindex(rates_for(last_w).index).fillna(0.0))
    movers = pd.concat([delta.nlargest(8), delta.nsmallest(8).sort_values(ascending=False)])
    movers = movers[movers != 0].drop_duplicates()
    if movers.empty:
        print("Skipping week-over-week chart: no pick-rate movement")
        return False

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in movers.values]
    ax.barh(movers.index[::-1], movers.values[::-1], color=colors[::-1])
    ax.set_title(f"Pick-rate movers: {prev_w} → {last_w}\n(mean across LCK/LPL/LEC/LCS)", pad=16)
    ax.set_xlabel("Pick rate change")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "week_over_week_delta.png", dpi=150)
    plt.close(fig)
    print("Wrote charts/week_over_week_delta.png")
    return True


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA_PROCESSED / "meta.json").read_text())
    window = f"since {meta['since']}" + (" (pre-summer fallback window)" if meta["fallback"] else "")

    pick_rates = _load("pick_rates.csv")
    ban_rates = _load("ban_rates.csv")
    weekly = _load("weekly_pick_rates.csv")

    _grouped_rate_chart(
        pick_rates, "pick_rate",
        f"Champion pick rates by region — {window}",
        "pick_rate_by_region.png",
    )
    _grouped_rate_chart(
        ban_rates, "ban_rate",
        f"Champion ban rates by region — {window}",
        "ban_rate_by_region.png",
    )
    week_over_week_chart(weekly)


if __name__ == "__main__":
    sys.exit(main())

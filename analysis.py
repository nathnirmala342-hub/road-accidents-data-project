"""Road accident analysis for India.

Usage:
    python analysis.py                      # uses data/accidents.csv
    python analysis.py --data path/to.csv   # any CSV with the same columns

Required columns: Date, Hour, State, Road_Type, Vehicle_Type, Cause, Weather, Killed, Injured
Saves charts to outputs/ and prints key findings computed from the data.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RED, BLUE = "#d9442b", "#3b6fb6"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date", "State", "Cause"]).drop_duplicates()
    df["Killed"] = df["Killed"].fillna(0).astype(int)
    df["Injured"] = df["Injured"].fillna(0).astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.dayofweek
    print(f"Loaded {before:,} rows, kept {len(df):,} after cleaning "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    return df


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(f"{out}/{name}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/accidents.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.data)

    # 1. Accidents by hour of day
    by_hour = df.groupby("Hour").size()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(by_hour.index, by_hour.values, color=RED)
    ax.set_xlabel("Hour of day (0-23)"); ax.set_ylabel("Accidents"); ax.set_title("When do accidents happen?")
    save(fig, args.out, "1_accidents_by_hour.png")

    # 2. Causes
    causes = df["Cause"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(causes.index, causes.values, color=BLUE)
    ax.set_xlabel("Accidents"); ax.set_title("Main causes of accidents")
    save(fig, args.out, "2_causes.png")

    # 3. Deaths by state
    deaths = df.groupby("State")["Killed"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(deaths.index, deaths.values, color=RED)
    ax.set_xlabel("People killed"); ax.set_title("Deaths by state")
    save(fig, args.out, "3_deaths_by_state.png")

    # 4. Fatality rate by road type and vehicle type
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["Road_Type", "Vehicle_Type"]):
        rate = (df.groupby(col)["Killed"].sum() / df.groupby(col).size() * 100).sort_values()
        ax.barh(rate.index, rate.values, color=RED)
        ax.set_xlabel("Deaths per 100 accidents"); ax.set_title(f"Fatality rate by {col.replace('_', ' ').lower()}")
    save(fig, args.out, "4_fatality_rates.png")

    # 5. Monthly trend
    monthly = df.set_index("Date").resample("MS").size()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly.values, color=BLUE, linewidth=1.8)
    ax.set_ylabel("Accidents per month"); ax.set_title("Monthly accidents over time")
    save(fig, args.out, "5_monthly_trend.png")

    # 6. Weekday x hour heatmap
    heat = df.pivot_table(index="Weekday", columns="Hour", values="State", aggfunc="count").reindex(range(7))
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7)); ax.set_yticklabels(DAYS)
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour"); ax.set_title("Accidents by weekday and hour")
    fig.colorbar(im, ax=ax, label="Accidents")
    save(fig, args.out, "6_weekday_hour_heatmap.png")

    # ---- Key findings, computed from the loaded data ----
    total = len(df)
    top_cause = df["Cause"].value_counts()
    peak = by_hour.rolling(3, center=True).sum().idxmax()
    night = df["Hour"].between(21, 23) | df["Hour"].between(0, 4)
    rate = lambda d: d["Killed"].sum() / len(d) * 100
    road_rate = df.groupby("Road_Type").apply(rate, include_groups=False).sort_values()
    veh_share = df["Vehicle_Type"].value_counts(normalize=True) * 100
    print("\n===== KEY FINDINGS =====")
    print(f"Total accidents: {total:,} | killed: {df['Killed'].sum():,} | injured: {df['Injured'].sum():,}")
    print(f"Top cause: {top_cause.index[0]} ({top_cause.iloc[0] / total:.0%} of accidents)")
    print(f"Peak accident period centers on hour {peak}:00 (the busiest 3-hour window)")
    print(f"Most vehicles involved: {veh_share.index[0]} ({veh_share.iloc[0]:.0f}%)")
    print(f"Deadliest road type: {road_rate.index[-1]} ({road_rate.iloc[-1]:.0f} deaths per 100 accidents)")
    print(f"State with most deaths: {deaths.index[-1]} ({deaths.iloc[-1]:,})")
    print(f"Fatality rate at night (9pm-4am) vs daytime: {rate(df[night]):.0f} vs {rate(df[~night]):.0f} deaths per 100 accidents")
    fog, clear = df[df["Weather"] == "Fog"], df[df["Weather"] == "Clear"]
    if len(fog) and len(clear):
        f = (fog["Cause"] == "Poor visibility").mean()
        c = (clear["Cause"] == "Poor visibility").mean()
        print(f"'Poor visibility' as the cause: {f:.0%} of fog accidents vs {c:.0%} of clear-weather accidents")
    lock = df[(df["Year"] == 2020) & df["Month"].isin([4, 5])]
    prev = df[(df["Year"] == 2019) & df["Month"].isin([4, 5])]
    if len(prev):
        print(f"Apr-May 2020 vs Apr-May 2019 accidents: {len(lock):,} vs {len(prev):,} "
              f"({(len(lock) / len(prev) - 1):+.0%})")
    print(f"\nCharts saved in ./{args.out}/")


if __name__ == "__main__":
    main()

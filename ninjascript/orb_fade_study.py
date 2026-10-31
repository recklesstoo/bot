"""Wyckoff-style fade of a failed opening range breakout.

Same input as orb_volume_study.py (5-minute bars, UTC start times). For each session:
  1. Opening range = the 09:30 bar. Break = first later bar that closes outside it.
  2. Signal = first later bar that closes back inside the range (before --entry-until).
  3. Entry = open of the next bar, against the break. Stop = extreme of the failed move.
  4. Target = range midpoint ("mid") or the opposite side of the range ("opp").
  5. Exit at the stop, at the target, or at the last bar available. If a bar touches both
     stop and target, the stop is assumed to fill first.

Results are in index points, gross and net of --cost (round-trip slippage + commission, in
points). Usage:
    python orb_fade_study.py bars.csv [--cost 0.6] [--entry-until 10:30]
"""
import argparse

import numpy as np
import pandas as pd

from orb_volume_study import load

TICK = 0.25


def rel_volume_table(df, lookback):
    vol = df.pivot(index="date", columns="slot", values="volume").sort_index()
    return vol, vol.shift(1).rolling(lookback, min_periods=max(5, lookback // 2)).mean()


def simulate(after, start, side, stop, target):
    """side=-1 short, +1 long. Returns exit price and reason."""
    for k in range(start, len(after)):
        hi, lo = after["high"].iloc[k], after["low"].iloc[k]
        if side == -1:
            if hi >= stop:
                return stop, "stop"
            if lo <= target:
                return target, "target"
        else:
            if lo <= stop:
                return stop, "stop"
            if hi >= target:
                return target, "target"
    return after["close"].iloc[-1], "time"


def run(df, lookback, entry_until, min_day_ratio):
    df = df.between_time("09:30", "11:00", inclusive="left")
    df = df.assign(date=df.index.date, slot=df.index.strftime("%H:%M"))
    vol, base = rel_volume_table(df, lookback)
    day_vol = vol.sum(axis=1)
    thin = day_vol < min_day_ratio * day_vol.median()

    trades = []
    for date, day in df.groupby("date"):
        if thin.get(date, False) or len(day) < 12:
            continue
        day = day.sort_index()
        orb = day.iloc[0]
        hi, lo = orb["high"], orb["low"]
        if hi - lo <= 0:
            continue
        after = day.iloc[1:]
        closes = after["close"].to_numpy()

        i = next((n for n, c in enumerate(closes) if c > hi or c < lo), None)
        if i is None:
            continue
        direction = 1 if closes[i] > hi else -1

        j = next((n for n in range(i + 1, len(after) - 1) if lo <= closes[n] <= hi), None)
        if j is None or after.index[j].strftime("%H:%M") >= entry_until:
            continue

        side = -direction
        entry = after["open"].iloc[j + 1]
        window = after.iloc[i : j + 1]
        stop = window["high"].max() + TICK if direction == 1 else window["low"].min() - TICK
        risk = abs(stop - entry)
        if risk <= 0:
            continue

        b_slot = after.index[i].strftime("%H:%M")
        r_slot = after.index[j].strftime("%H:%M")
        b_base, r_base = base.loc[date, b_slot], base.loc[date, r_slot]
        rel_break = after["volume"].iloc[i] / b_base if pd.notna(b_base) and b_base > 0 else np.nan
        rel_reentry = after["volume"].iloc[j] / r_base if pd.notna(r_base) and r_base > 0 else np.nan

        mid = (hi + lo) / 2
        for name, target in (("mid", mid), ("opp", lo if side == -1 else hi)):
            if (side == -1 and target >= entry) or (side == 1 and target <= entry):
                continue
            exit_px, why = simulate(after, j + 1, side, stop, target)
            trades.append(
                {
                    "date": date,
                    "target": name,
                    "side": side,
                    "risk": risk,
                    "pts": side * (exit_px - entry),
                    "why": why,
                    "rel_break": rel_break,
                    "rel_reentry": rel_reentry,
                }
            )
    return pd.DataFrame(trades)


def summarize(t, cost):
    net = t["pts"] - cost
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return pd.Series(
        {
            "n": len(t),
            "win%": round((net > 0).mean() * 100),
            "gross_pts": round(t["pts"].mean(), 2),
            "net_pts": round(net.mean(), 2),
            "median_R": round((net / t["risk"]).median(), 2),
            "PF": round(wins / losses, 2) if losses > 0 else np.inf,
            "total_net": round(net.sum(), 1),
        }
    )


def report(trades, cost):
    if trades.empty:
        print("no trades")
        return
    for name, t in trades.groupby("target"):
        print(f"=== target: {name} ===")
        print(summarize(t, cost).to_string())
        t = t.sort_values("date")
        half = len(t) // 2
        print("first half :", summarize(t.iloc[:half], cost)[["n", "win%", "net_pts", "PF"]].to_dict())
        print("second half:", summarize(t.iloc[half:], cost)[["n", "win%", "net_pts", "PF"]].to_dict())
        print("exits:", t["why"].value_counts().to_dict())
        for col in ("rel_break", "rel_reentry"):
            ok = t.dropna(subset=[col])
            if len(ok) >= 20:
                ok = ok.assign(q=pd.qcut(ok[col], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]))
                g = ok.groupby("q", observed=True).apply(lambda x: summarize(x, cost), include_groups=False)
                print(f"by {col} quartile:")
                print(g[["n", "win%", "net_pts", "PF"]].to_string())
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--entry-until", default="10:30")
    ap.add_argument("--cost", type=float, default=0.6)
    ap.add_argument("--min-day-ratio", type=float, default=0.5)
    a = ap.parse_args()
    report(run(load(a.csv), a.lookback, a.entry_until, a.min_day_ratio), a.cost)

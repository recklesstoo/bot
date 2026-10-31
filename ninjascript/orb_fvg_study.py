"""Opening range breakout that leaves a fair value gap (FVG), entered on the retest of the gap.

Same input as orb_volume_study.py (5-minute bars, UTC start times). For each session:
  1. Opening range = the 09:30 bar. Break = first later bar that closes outside it.
  2. FVG = three-candle gap in the break direction whose middle candle is the break bar or one
     of the next two bars. Bullish: low[k+1] > high[k-1]. Bearish: high[k+1] < low[k-1].
  3. Entry = limit at the gap edge nearest to price ("edge") or at its midpoint ("mid"), placed
     after the gap is confirmed (close of candle k+1). Cancelled if a bar closes beyond the far
     side of the gap before it fills.
  4. Stop = one tick beyond the far side of the gap. Target = 1R or 2R. Exit at the last bar if
     neither is hit. On the fill bar only the stop is checked; if a bar touches both stop and
     target the stop is assumed to fill first.

Results are in index points, gross and net of --cost. Usage:
    python orb_fvg_study.py bars.csv [--cost 0.6] [--min-gap 1.0] [--entry-until 10:30]
"""
import argparse

import numpy as np
import pandas as pd

from orb_fade_study import TICK, rel_volume_table, summarize
from orb_volume_study import load


def find_gap(high, low, k, d):
    """Gap (lo, hi) whose middle candle is k, in direction d; None if there is no gap."""
    if d == 1:
        gap_lo, gap_hi = high[k - 1], low[k + 1]
    else:
        gap_lo, gap_hi = high[k + 1], low[k - 1]
    return (gap_lo, gap_hi) if gap_hi > gap_lo else None


def run(df, lookback, entry_until, min_day_ratio, min_gap):
    df = df.between_time("09:30", "11:00", inclusive="left")
    df = df.assign(date=df.index.date, slot=df.index.strftime("%H:%M"))
    vol, base = rel_volume_table(df, lookback)
    day_vol = vol.sum(axis=1)
    thin = day_vol < min_day_ratio * day_vol.median()

    trades, with_gap, broke = [], 0, 0
    for date, day in df.groupby("date"):
        if thin.get(date, False) or len(day) < 12:
            continue
        day = day.sort_index()
        high, low = day["high"].to_numpy(), day["low"].to_numpy()
        close, open_ = day["close"].to_numpy(), day["open"].to_numpy()
        slots = day["slot"].to_numpy()
        volume = day["volume"].to_numpy()
        hi, lo = high[0], low[0]
        if hi - lo <= 0:
            continue

        b = next((n for n in range(1, len(day)) if close[n] > hi or close[n] < lo), None)
        if b is None:
            continue
        broke += 1
        d = 1 if close[b] > hi else -1

        gap = k = None
        for kk in range(b, min(b + 3, len(day) - 1)):
            g = find_gap(high, low, kk, d)
            if g and g[1] - g[0] >= min_gap:
                gap, k = g, kk
                break
        if gap is None:
            continue
        with_gap += 1
        gap_lo, gap_hi = gap

        mid_base = base.loc[date, slots[k]]
        rel_mid = volume[k] / mid_base if pd.notna(mid_base) and mid_base > 0 else np.nan

        near, far = (gap_hi, gap_lo) if d == 1 else (gap_lo, gap_hi)
        stop = far - d * TICK
        for entry_name, entry in (("edge", near), ("mid", (gap_lo + gap_hi) / 2)):
            f = None
            for n in range(k + 2, len(day)):
                if slots[n] >= entry_until:
                    break
                if (d == 1 and close[n] < gap_lo) or (d == -1 and close[n] > gap_hi):
                    break
                touched = low[n] <= entry if d == 1 else high[n] >= entry
                if touched:
                    f = n
                    break
            if f is None:
                continue
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            for r_mult in (1, 2):
                target = entry + d * r_mult * risk
                exit_px, why = day["close"].iloc[-1], "time"
                for n in range(f, len(day)):
                    stopped = low[n] <= stop if d == 1 else high[n] >= stop
                    hit = (high[n] >= target if d == 1 else low[n] <= target) and n > f
                    if stopped:
                        exit_px, why = stop, "stop"
                        break
                    if hit:
                        exit_px, why = target, "target"
                        break
                trades.append(
                    {
                        "date": date,
                        "variant": f"{entry_name}/{r_mult}R",
                        "risk": risk,
                        "pts": d * (exit_px - entry),
                        "why": why,
                        "rel_mid": rel_mid,
                    }
                )
    return pd.DataFrame(trades), broke, with_gap


def report(trades, broke, with_gap, cost):
    print(f"sessions with a break: {broke}; with an FVG in the break direction: {with_gap}")
    if trades.empty:
        print("no filled trades")
        return
    for name, t in trades.groupby("variant"):
        print(f"=== {name} ===")
        print(summarize(t, cost).to_string())
        t = t.sort_values("date")
        half = len(t) // 2
        print("first half :", summarize(t.iloc[:half], cost)[["n", "win%", "net_pts", "PF"]].to_dict())
        print("second half:", summarize(t.iloc[half:], cost)[["n", "win%", "net_pts", "PF"]].to_dict())
        print("exits:", t["why"].value_counts().to_dict())
        ok = t.dropna(subset=["rel_mid"])
        if len(ok) >= 20:
            ok = ok.assign(q=pd.qcut(ok["rel_mid"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]))
            g = ok.groupby("q", observed=True).apply(lambda x: summarize(x, cost), include_groups=False)
            print("by volume of the gap's middle candle:")
            print(g[["n", "win%", "net_pts", "PF"]].to_string())
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--entry-until", default="10:30")
    ap.add_argument("--cost", type=float, default=0.6)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--min-day-ratio", type=float, default=0.5)
    a = ap.parse_args()
    res, broke, with_gap = run(load(a.csv), a.lookback, a.entry_until, a.min_day_ratio, a.min_gap)
    report(res, broke, with_gap, a.cost)

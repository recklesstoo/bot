"""Opening range breakout: does breakout-bar volume separate real breaks from failed ones?

Input: CSV with columns time,open,high,low,close,volume. 5-minute bars, bar start time as an
ISO timestamp in UTC, covering at least 09:30-10:30 New York for each session.

Usage:
    python orb_volume_study.py bars.csv [--lookback 20] [--fail-bars 6] [--hold-bars 12]

For each session the opening range is the 09:30 bar. The break is the first later bar that
closes outside it. Relative volume = break-bar volume / mean volume of the same time slot over
the previous `lookback` sessions. A break "fails" if a later bar within `fail-bars` closes back
inside the range. Moves are in multiples of the opening range width, measured from the break
close over the next `hold-bars` bars. Moves are summarised with the median because days with a
tiny opening range inflate the mean.
"""
import argparse

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    return df[["open", "high", "low", "close", "volume"]]


def study(df, lookback, fail_bars, hold_bars, min_day_ratio):
    df = df.between_time("09:30", "10:30", inclusive="left")
    df = df.assign(date=df.index.date, slot=df.index.strftime("%H:%M"))

    vol = df.pivot(index="date", columns="slot", values="volume").sort_index()
    base = vol.shift(1).rolling(lookback, min_periods=max(5, lookback // 2)).mean()

    # Drop thin sessions (holidays, expiry-week volume that sits in the other contract).
    day_vol = vol.sum(axis=1)
    thin = day_vol < min_day_ratio * day_vol.median()
    skipped = int(thin.sum())

    rows = []
    for date, day in df.groupby("date"):
        if thin.get(date, False) or len(day) < 12:
            continue
        day = day.sort_index()
        orb = day.iloc[0]
        hi, lo = orb["high"], orb["low"]
        width = hi - lo
        if width <= 0:
            continue
        after = day.iloc[1:]

        brk = None
        for i in range(len(after)):
            close = after["close"].iloc[i]
            if close > hi:
                brk = (i, 1)
                break
            if close < lo:
                brk = (i, -1)
                break
        if brk is None:
            continue

        i, direction = brk
        bar = after.iloc[i]
        slot = after.index[i].strftime("%H:%M")
        b = base.loc[date, slot]
        rel_vol = bar["volume"] / b if pd.notna(b) and b > 0 else np.nan

        nxt = after.iloc[i + 1 : i + 1 + hold_bars]
        if nxt.empty:
            continue
        early = nxt.iloc[:fail_bars]
        inside = (early["close"] >= lo) & (early["close"] <= hi)

        if direction == 1:
            mfe = (nxt["high"].max() - bar["close"]) / width
            mae = (bar["close"] - nxt["low"].min()) / width
        else:
            mfe = (bar["close"] - nxt["low"].min()) / width
            mae = (nxt["high"].max() - bar["close"]) / width
        final = direction * (nxt["close"].iloc[-1] - bar["close"]) / width

        rows.append(
            {
                "date": date,
                "dir": "up" if direction == 1 else "down",
                "slot": slot,
                "rel_vol": rel_vol,
                "failed": bool(inside.any()),
                "mfe": mfe,
                "mae": mae,
                "final": final,
            }
        )

    return pd.DataFrame(rows), skipped


def report(res, skipped):
    print(f"sessions with a break: {len(res)}  (thin sessions skipped: {skipped})")
    if res.empty:
        return
    print(f"overall fail rate: {res['failed'].mean():.0%}")
    print(f"overall median final move: {res['final'].median():+.2f} range widths\n")

    ok = res.dropna(subset=["rel_vol"]).copy()
    if len(ok) < 20:
        print("fewer than 20 breaks with a volume baseline: too few for quartiles")
        return
    ok["rv_bucket"] = pd.qcut(ok["rel_vol"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    out = ok.groupby("rv_bucket", observed=True).agg(
        n=("failed", "size"),
        rel_vol_mean=("rel_vol", "mean"),
        fail_rate=("failed", "mean"),
        mfe=("mfe", "median"),
        mae=("mae", "median"),
        final=("final", "median"),
    )
    print(out.round(2).to_string())
    print("\nby slot of the break:")
    by_slot = ok.groupby("slot").agg(n=("failed", "size"), fail_rate=("failed", "mean"), final=("final", "median"))
    print(by_slot.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--fail-bars", type=int, default=6)
    ap.add_argument("--hold-bars", type=int, default=12)
    ap.add_argument("--min-day-ratio", type=float, default=0.5)
    a = ap.parse_args()
    result, skipped = study(load(a.csv), a.lookback, a.fail_bars, a.hold_bars, a.min_day_ratio)
    report(result, skipped)

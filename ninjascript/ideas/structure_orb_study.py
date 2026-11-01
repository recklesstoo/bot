"""Opening range defined by structure instead of a fixed number of bars.

Structural range: from the 09:30 open, wait until a swing high and a swing low have both been
confirmed (a pivot is a bar whose high/low is beyond the `k` bars on each side; it is confirmed
`k` bars later). The range is the high/low of every bar up to the confirmation bar. The break is
the first later close outside it. Days where no structure forms before 10:30 are skipped.

Fixed ranges of 1, 3 and 6 bars (5, 15, 30 minutes) are run with the same trade rules for comparison:
entry at the next bar's open, stop on the opposite side of the range, target 2R, flat at the end
of the data (11:00). Net of costs (0.6 points ES, 0.7 points NQ), in R.

    python structure_orb_study.py --data-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orb_volume_study import load  # noqa: E402

COST = {"es": 0.6, "nq": 0.7}
LAST_BREAK_BAR = 11  # signal bar must start before 10:30


def pivot_confirm_index(x, k, high):
    """Index of the first confirmed pivot (confirmed at bar i+k), or None."""
    for i in range(k, len(x) - k):
        left, right = x[i - k:i], x[i + 1:i + k + 1]
        ok = (x[i] > left.max() and x[i] >= right.max()) if high else (x[i] < left.min() and x[i] <= right.min())
        if ok:
            return i + k
    return None


def trade(d, m, cost):
    """Range = bars 0..m. First close outside after m. Returns (date, net R, failed) or None."""
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    hi, lo = h[:m + 1].max(), l[:m + 1].min()
    width = hi - lo
    if width <= 0:
        return None
    b = next((i for i in range(m + 1, len(c) - 1) if c[i] > hi or c[i] < lo), None)
    if b is None or b > LAST_BREAK_BAR:
        return None
    side = 1 if c[b] > hi else -1
    entry, stop = o[b + 1], (lo if side == 1 else hi)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    tgt = entry + side * 2 * risk
    exit_px = c[-1]
    for j in range(b + 1, len(c)):
        if (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop):
            exit_px = stop
            break
        if (side == 1 and h[j] >= tgt) or (side == -1 and l[j] <= tgt):
            exit_px = tgt
            break
    failed = any(lo <= c[j] <= hi for j in range(b + 1, min(b + 7, len(c))))
    return (d["date"], (side * (exit_px - entry) - cost) / risk, failed, m, width)


def days(df):
    df = df.between_time("09:30", "11:00", inclusive="left")
    out = []
    for date, g in df.groupby(df.index.date):
        if len(g) < 18:
            continue
        g = g.sort_index()
        out.append(dict(date=date, o=g["open"].to_numpy(), h=g["high"].to_numpy(),
                        l=g["low"].to_numpy(), c=g["close"].to_numpy()))
    return out


def run(all_days, cost, variant):
    res = []
    for d in all_days:
        if variant[0] == "fixed":
            m = variant[1] - 1
        else:
            k = variant[1]
            a, b = pivot_confirm_index(d["h"], k, True), pivot_confirm_index(d["l"], k, False)
            if a is None or b is None:
                continue
            m = max(a, b)
            if m >= LAST_BREAK_BAR:
                continue
        t = trade(d, m, cost)
        if t:
            res.append(t)
    return res


def summarize(rows):
    if not rows:
        return dict(n=0)
    r = np.array([x[1] for x in rows])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 2 else np.nan
    return dict(n=len(r), meanR=round(r.mean(), 3), t=round(r.mean() / se, 2) if se else np.nan,
                win=round((r > 0).mean(), 3), failed=round(np.mean([x[2] for x in rows]), 3),
                range_bars=round(np.mean([x[3] + 1 for x in rows]), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    variants = [("fixed", 1, "fixed 5 min"), ("fixed", 3, "fixed 15 min"), ("fixed", 6, "fixed 30 min"),
                ("struct", 1, "structure k=1"), ("struct", 2, "structure k=2"), ("struct", 3, "structure k=3")]
    data = {}
    for sym in ("es", "nq"):
        for y in ("2024", "2025", "2026"):
            data[(sym, y)] = days(load(os.path.join(a.data_dir, f"{sym}{y}.csv")))
    for v in variants:
        print(f"=== {v[2]} ===")
        allr = []
        for y in ("2024", "2025", "2026"):
            rows = []
            for sym in ("es", "nq"):
                rows += run(data[(sym, y)], COST[sym], v)
            allr += rows
            s = summarize(rows)
            print(f"  {y}: {s}")
        print(f"  ALL : {summarize(allr)}")


if __name__ == "__main__":
    main()

"""First-candle breakout (my version of the Zarattini-Aziz opening candle idea, not their code).

Rule fixed before running: at 09:35 go long if the 09:30 bar closed above its open, short if below
(doji skipped). Entry at the open of the 09:35 bar. Stop at the opposite extreme of the first bar.
Variants of the exit: target T x R (T = 1, 2, 5, 10) or none, always flat at 11:00 (last data bar).
Also a no-stop variant that just holds to 11:00. Net of costs (0.6 pts ES, 0.7 pts NQ).
R = net profit / initial risk. Pts = net profit in points, which does not blow up on tiny candles.

    python opening_candle_study.py --data-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orb_volume_study import load  # noqa: E402
from structure_orb_study import COST, days  # noqa: E402


def run(d, cost, tmult, use_stop=True):
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    if c[0] == o[0]:
        return None
    side = 1 if c[0] > o[0] else -1
    entry = o[1]
    stop = l[0] if side == 1 else h[0]
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    tgt = entry + side * tmult * risk if tmult else None
    exit_px = c[-1]
    for j in range(1, len(c)):
        if use_stop and ((side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop)):
            exit_px = stop
            break
        if tgt is not None and ((side == 1 and h[j] >= tgt) or (side == -1 and l[j] <= tgt)):
            exit_px = tgt
            break
    pts = side * (exit_px - entry) - cost
    return dict(R=pts / risk, pts=pts, side=side, risk=risk)


def summ(x):
    if len(x) < 3:
        return f"n {len(x)}"
    r, p = x["R"].to_numpy(), x["pts"].to_numpy()
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    return f"n {len(r):3d} win {np.mean(p > 0):.2f} meanR {r.mean():+.3f} medianR {np.median(r):+.2f} pts {p.mean():+.2f} t(pts) {t:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    data = {(s, y): days(load(os.path.join(a.data_dir, f"{s}{y}.csv"))) for s in ("es", "nq") for y in ("2024", "2025", "2026")}
    variants = [("stop, hold to 11:00", 0, True), ("stop, target 1R", 1, True), ("stop, target 2R", 2, True),
                ("stop, target 5R", 5, True), ("stop, target 10R", 10, True), ("no stop, hold to 11:00", 0, False)]
    for name, tm, us in variants:
        print(f"=== {name} ===")
        allr = []
        for y in ("2024", "2025", "2026"):
            rows = []
            for s in ("es", "nq"):
                for d in data[(s, y)]:
                    t = run(d, COST[s], tm, us)
                    if t:
                        rows.append(t)
            allr += rows
            print(f"  {y}: {summ(pd.DataFrame(rows))}")
        df = pd.DataFrame(allr)
        print(f"  ALL : {summ(df)}")
        print(f"    long {summ(df[df.side == 1])} | short {summ(df[df.side == -1])}")


if __name__ == "__main__":
    main()

"""Gamma flip and call/put wall distance vs opening range breakout results.

Inputs as gamma_orb_study.py, but gex CSVs need gamma_flip_price, call_wall_price, put_wall_price.
The flip is only used when it lies within 10% of the open (TraderOS often returns values near 40% of
price when the regime is negative; those are treated as missing).

Tests (fixed before running), 15- and 30-min ORB, same trade as structure_orb_study:
  A. wall in the way: a wall between entry and the 2R target in the trade direction ("blocked")
     versus none ("clear").
  B. pinning: the open is within 0.25% of the nearest wall versus further away; failure rate and R.
  C. flip side: the open above versus below a valid gamma flip.

    python gex_levels_orb_study.py --data-dir DIR --gex-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orb_volume_study import load  # noqa: E402
from structure_orb_study import COST, LAST_BREAK_BAR, days  # noqa: E402


def trade(d, m, cost, levels):
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    hi, lo = h[:m + 1].max(), l[:m + 1].min()
    if hi - lo <= 0:
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
    blocked = any((v - entry) * side > 0 and abs(v - entry) < 2 * risk for v in levels)
    failed = any(lo <= c[j] <= hi for j in range(b + 1, min(b + 7, len(c))))
    return dict(R=(side * (exit_px - entry) - cost) / risk, blocked=blocked, failed=failed)


def summ(x):
    if len(x) < 3:
        return f"n {len(x)}"
    r = x["R"].to_numpy()
    return f"n {len(r):3d} meanR {r.mean():+.3f} t {r.mean() / (r.std(ddof=1) / np.sqrt(len(r))):+.2f} fail {x['failed'].mean():.2f}"


def diff(a, b):
    if len(a) < 3 or len(b) < 3:
        return "n/a"
    se = np.sqrt(a.R.var(ddof=1) / len(a) + b.R.var(ddof=1) / len(b))
    return f"{a.R.mean() - b.R.mean():+.3f} R (t {(a.R.mean() - b.R.mean()) / se:+.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gex-dir", required=True)
    a = ap.parse_args()
    rows = []
    for sym in ("es", "nq"):
        g = pd.read_csv(os.path.join(a.gex_dir, f"gex_{sym}.csv"))
        g["date"] = pd.to_datetime(g["time"], utc=True).dt.tz_localize(None).dt.normalize().dt.date
        g = g[pd.to_datetime(g["date"]).dt.weekday < 5].drop_duplicates("date").set_index("date")
        for y in ("2024", "2025", "2026"):
            for d in days(load(os.path.join(a.data_dir, f"{sym}{y}.csv"))):
                if d["date"] not in g.index:
                    continue
                r = g.loc[d["date"]]
                op = d["o"][0]
                cw, pw, fl = r["call_wall_price"], r["put_wall_price"], r["gamma_flip_price"]
                flip_ok = pd.notna(fl) and abs(fl / op - 1) < 0.10
                lv = [v for v in (cw, pw) if pd.notna(v)]
                near = min(abs(v - op) / op for v in lv) * 100 if lv else np.nan
                for bars in (3, 6):
                    t = trade(d, bars - 1, COST[sym], lv)
                    if t:
                        t.update(sym=sym, year=y, bars=bars * 5, near=near,
                                 above_flip=(op > fl) if flip_ok else np.nan)
                        rows.append(t)
    df = pd.DataFrame(rows)
    for mins in (15, 30):
        s = df[df.bars == mins]
        print(f"=== ORB {mins} min: {len(s)} breakout trades ===")
        print("A. wall within 2R in trade direction")
        for label, sub in (("ALL", s),) + tuple((y, s[s.year == y]) for y in ("2024", "2025", "2026")):
            print(f"  {label}: clear {summ(sub[~sub.blocked])} | blocked {summ(sub[sub.blocked])} | diff {diff(sub[~sub.blocked], sub[sub.blocked])}")
        print("B. open near a wall (<0.25%) vs further")
        for label, sub in (("ALL", s),) + tuple((y, s[s.year == y]) for y in ("2024", "2025", "2026")):
            print(f"  {label}: near {summ(sub[sub.near < 0.25])} | far {summ(sub[sub.near >= 0.25])}")
        print("C. open above vs below valid flip")
        v = s.dropna(subset=["above_flip"])
        for label, sub in (("ALL", v),) + tuple((y, v[v.year == y]) for y in ("2024", "2025", "2026")):
            print(f"  {label}: above {summ(sub[sub.above_flip == True])} | below {summ(sub[sub.above_flip == False])}")


if __name__ == "__main__":
    main()

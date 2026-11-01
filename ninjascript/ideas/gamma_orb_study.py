"""Does negative gamma make an opening range breakout more likely / better?

GEX regime comes from TraderOS gex_levels_15m at 13:15 UTC (09:15 New York in summer time), saved as
CSV by --gex-dir (gex_es.csv, gex_nq.csv with columns time, net_gamma, regime). Weekend rows are dropped.
Bars are the 5-minute CSVs used by the other studies (09:30-11:00 New York).

Measured per day for the 15-min and 30-min opening range (fixed before running):
  broke      : a close outside the range before 10:30
  failed     : after the break, a close back inside within 6 bars
  follow     : max excursion beyond the range in the break direction until 11:00, in range widths
  R          : same trade as structure_orb_study (entry next open, stop far side, 2R, flat 11:00, net costs)
Groups: negative vs positive gamma. Reported by year so each year acts as its own check.

    python gamma_orb_study.py --data-dir DIR --gex-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orb_volume_study import load  # noqa: E402
from structure_orb_study import COST, LAST_BREAK_BAR, days, trade  # noqa: E402


def follow(d, m):
    h, l, c = d["h"], d["l"], d["c"]
    hi, lo = h[:m + 1].max(), l[:m + 1].min()
    w = hi - lo
    b = next((i for i in range(m + 1, len(c) - 1) if c[i] > hi or c[i] < lo), None)
    if b is None or b > LAST_BREAK_BAR or w <= 0:
        return None
    ex = (h[b:].max() - hi) / w if c[b] > hi else (lo - l[b:].min()) / w
    return ex


def tstat(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


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
                neg = g.loc[d["date"], "net_gamma"] < 0
                for bars in (3, 6):
                    m = bars - 1
                    wid = d["h"][:m + 1].max() - d["l"][:m + 1].min()
                    t = trade(d, m, COST[sym])
                    rows.append(dict(sym=sym, year=y, bars=bars * 5, neg=neg, broke=t is not None,
                                     failed=t[2] if t else np.nan, R=t[1] if t else np.nan,
                                     follow=follow(d, m) if t else np.nan,
                                     width_pct=wid / d["o"][0] * 100))
    df = pd.DataFrame(rows)
    for mins in (15, 30):
        s = df[df.bars == mins]
        print(f"=== opening range {mins} min, {s.groupby(['year','sym']).ngroups} year-symbol blocks, {len(s)} days ===")
        for label, sub in (("ALL", s),) + tuple((y, s[s.year == y]) for y in ("2024", "2025", "2026")):
            n, p = sub[sub.neg], sub[~sub.neg]
            br = sub[sub.broke]
            print(f"  {label}: days neg {len(n)} / pos {len(p)}")
            print(f"     broke    neg {n.broke.mean():.3f}  pos {p.broke.mean():.3f}")
            bn, bp = n[n.broke], p[p.broke]
            print(f"     failed   neg {bn.failed.astype(float).mean():.3f} (n {len(bn)})  pos {bp.failed.astype(float).mean():.3f} (n {len(bp)})")
            print(f"     follow   neg {bn.follow.median():.2f}  pos {bp.follow.median():.2f} range widths (median); t of mean diff {tstat(bn.follow, bp.follow):+.2f}")
            print(f"     meanR    neg {bn.R.mean():+.3f}  pos {bp.R.mean():+.3f}; t {tstat(bn.R, bp.R):+.2f}")
            print(f"     range width % of price  neg {n.width_pct.median():.3f}  pos {p.width_pct.median():.3f}")


if __name__ == "__main__":
    main()

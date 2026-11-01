"""15-minute ORB split by the sign of the 09:30-09:45 aggressor delta (ES, 2026, sampled days).

delta_data/delta_es_0930_0945.csv: buy and sell aggressor volume for 09:30-09:45 ET from TraderOS
market_trades (summary=flow), one row per sampled day (every 5th trading day from 2026-03-09).
Trade: same as structure_orb_study.trade (first 5-minute close outside the 15-minute range before 10:30,
entry next open, stop opposite side, 2R, flat 11:00, 0.6 points cost). Variants: Off (all breaks),
Confirm (delta sign == break side), Fade (opposite). Reports n, mean R, win rate, failed-break rate.
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


def break_side(d, m):
    hi, lo = d["h"][:m + 1].max(), d["l"][:m + 1].min()
    c = d["c"]
    b = next((i for i in range(m + 1, len(c) - 1) if c[i] > hi or c[i] < lo), None)
    if b is None or b > LAST_BREAK_BAR:
        return None
    return 1 if c[b] > hi else -1


def stats(x):
    if len(x) == 0:
        return "n 0"
    r = x["R"].to_numpy()
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else float("nan")
    return f"n {len(r):2d} meanR {r.mean():+.3f} t {t:+.2f} win {np.mean(r > 0):.2f} failed {x['failed'].mean():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--delta", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_data", "delta_es_0930_0945.csv"))
    a = ap.parse_args()
    dl = pd.read_csv(a.delta)
    dl["date"] = pd.to_datetime(dl["date"]).dt.date
    dl["delta"] = dl["buy"] - dl["sell"]
    dl["vol"] = dl["buy"] + dl["sell"]
    dl["pct"] = dl["delta"] / dl["vol"] * 100
    dd = dl.set_index("date")
    rows = []
    for d in days(load(os.path.join(a.data_dir, "es2026.csv"))):
        if d["date"] not in dd.index:
            continue
        side = break_side(d, 2)
        t = trade(d, 2, COST["es"])
        if side is None or t is None:
            continue
        ds = int(np.sign(dd.loc[d["date"], "delta"]))
        rows.append(dict(date=d["date"], side=side, dside=ds, pct=dd.loc[d["date"], "pct"], vol=dd.loc[d["date"], "vol"], R=t[1], failed=t[2]))
    df = pd.DataFrame(rows)
    print(f"{len(dl)} days with delta, {len(df)} with a 15-minute ORB break before 10:30")
    for name, sub in (("all days", df), ("volume >= 30k (drops two roll-week days)", df[df.vol >= 30000])):
        print(f"--- {name} ---")
        print("  Off     ", stats(sub))
        print("  Confirm ", stats(sub[sub.side == sub.dside]))
        print("  Fade    ", stats(sub[(sub.side == -sub.dside) & (sub.dside != 0)]))
        big = sub[sub.pct.abs() >= 2]
        print("  |delta| >= 2% of volume:")
        print("    Confirm", stats(big[big.side == big.dside]), "| Fade", stats(big[big.side == -big.dside]))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

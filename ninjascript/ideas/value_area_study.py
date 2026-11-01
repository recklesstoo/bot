"""Open versus the previous day's value area (small sample, TraderOS value_levels).

va_es.csv / va_nq.csv hold the previous-session RTH VAH and VAL as returned by TraderOS value_levels
(30 sessions; the file date is the session the profile belongs to, so it is used for the NEXT day).
Bars are the 5-minute CSVs (09:30-11:00 New York). Questions, fixed before running:
  1. Where does the day open relative to the previous value area (above, inside, below)?
  2. If outside: does price get back to the edge of the value area by 11:00? After two consecutive
     5-minute closes inside, does it reach the opposite edge by 11:00 (80% rule, cut at 11:00)?
  3. If inside: does it leave the value area (touch VAH or VAL) by 11:00?
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_orb_study import days  # noqa: E402
from orb_volume_study import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--va-dir", required=True)
    a = ap.parse_args()
    out = []
    for sym in ("es", "nq"):
        va = pd.read_csv(os.path.join(a.va_dir, f"va_{sym}.csv"))
        va["next"] = va["date"].shift(-1)
        lv = {pd.Timestamp(r.next).date(): (r.vah, r.val) for r in va.itertuples() if isinstance(r.next, str)}
        for d in days(load(os.path.join(a.data_dir, f"{sym}2026.csv"))):
            if d["date"] not in lv:
                continue
            vah, val = lv[d["date"]]
            o, h, l, c = d["o"], d["h"], d["l"], d["c"]
            op = o[0]
            where = "above" if op > vah else ("below" if op < val else "inside")
            row = dict(sym=sym, date=d["date"], where=where, touch=None, rule80=None, left=None)
            if where == "above":
                row["touch"] = bool((l <= vah).any())
                idx = next((i for i in range(1, len(c)) if c[i] <= vah and c[i - 1] <= vah), None)
                row["rule80"] = None if idx is None else bool((l[idx:] <= val).any())
            elif where == "below":
                row["touch"] = bool((h >= val).any())
                idx = next((i for i in range(1, len(c)) if c[i] >= val and c[i - 1] >= val), None)
                row["rule80"] = None if idx is None else bool((h[idx:] >= vah).any())
            else:
                row["left"] = bool((h >= vah).any() or (l <= val).any())
            out.append(row)
    df = pd.DataFrame(out)
    print(df.groupby(["sym", "where"]).size().unstack(fill_value=0))
    for w in ("above", "below"):
        s = df[df["where"] == w]
        print(f"open {w}: n {len(s)}, back to VA edge by 11:00 {s.touch.mean():.2f}, two closes inside {s.rule80.notna().sum()}, reached opposite edge {s.rule80.dropna().astype(float).mean():.2f}")
    s = df[df["where"] == "inside"]
    print(f"open inside: n {len(s)}, left the VA by 11:00 {s.left.mean():.2f}")


if __name__ == "__main__":
    main()

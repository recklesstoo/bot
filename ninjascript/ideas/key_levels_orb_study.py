"""Do key levels in the way of an opening range breakout change its result?

Key levels for a session: previous trading day high/low/close, previous week high/low and previous
month high/low, from the daily bars (UTC-day buckets, the same definition TraderOS uses for its
"prev_day_*" levels). Weekend rows are skipped.

For every breakout of a fixed opening range (15 and 30 minutes) the trade is the same as in
structure_orb_study.py (entry next open, stop on the other side of the range, target 2R, flat at 11:00,
net of costs, in R). The question, fixed before running: is a trade with no key level between the entry
and the 2R target ("clear path") better than one with at least one level in the way ("blocked")?

    python key_levels_orb_study.py --data-dir DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orb_volume_study import load  # noqa: E402
from structure_orb_study import COST, LAST_BREAK_BAR, days  # noqa: E402


def level_table(path):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None).dt.normalize()
    d = d[d["date"].dt.weekday < 5].sort_values("date").reset_index(drop=True)
    d["week"] = d["date"].dt.to_period("W-SUN")
    d["month"] = d["date"].dt.to_period("M")
    return d


def levels_for(daily, date):
    ts = pd.Timestamp(date)
    prev = daily[daily["date"] < ts]
    if prev.empty:
        return {}
    p = prev.iloc[-1]
    out = {"PDH": p["high"], "PDL": p["low"], "PDC": p["close"]}
    wk = ts.to_period("W-SUN")
    pw = daily[daily["week"] == wk - 1]
    if len(pw):
        out["PWH"], out["PWL"] = pw["high"].max(), pw["low"].min()
    pm = daily[daily["month"] == ts.to_period("M") - 1]
    if len(pm):
        out["PMH"], out["PML"] = pm["high"].max(), pm["low"].min()
    return out


def breakout_trade(d, bars, levels, cost):
    h, l, c, o = d["h"], d["l"], d["c"], d["o"]
    m = bars - 1
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
    ahead = [abs(v - entry) / risk for v in levels.values() if (v - entry) * side > 0]
    nearest = min(ahead) if ahead else np.inf
    return dict(date=d["date"], R=(side * (exit_px - entry) - cost) / risk, nearest=nearest,
                blocked=nearest < 2.0, close_block=nearest < 1.0)


def summarize(x):
    if len(x) < 3:
        return dict(n=len(x))
    r = x["R"].to_numpy()
    se = r.std(ddof=1) / np.sqrt(len(r))
    return dict(n=len(r), meanR=round(r.mean(), 3), t=round(r.mean() / se, 2), win=round((r > 0).mean(), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    daily = {s: level_table(os.path.join(a.data_dir, f"daily_{s}.csv")) for s in ("es", "nq")}
    for bars, name in ((3, "15 min"), (6, "30 min")):
        rows = []
        for sym in ("es", "nq"):
            for y in ("2024", "2025", "2026"):
                for d in days(load(os.path.join(a.data_dir, f"{sym}{y}.csv"))):
                    lv = levels_for(daily[sym], d["date"])
                    t = breakout_trade(d, bars, lv, COST[sym])
                    if t:
                        t.update(sym=sym, year=y)
                        rows.append(t)
        df = pd.DataFrame(rows)
        print(f"=== fixed opening range {name}: {len(df)} breakout trades ===")
        for label, mask in (("all", df.index == df.index), ("clear path (no level within 2R)", ~df.blocked),
                            ("blocked (level within 2R)", df.blocked), ("level within 1R", df.close_block)):
            print(f"  {label:34s} {summarize(df[mask])}")
        c, b = df[~df.blocked]["R"], df[df.blocked]["R"]
        se = np.sqrt(c.var(ddof=1) / len(c) + b.var(ddof=1) / len(b))
        print(f"  difference clear - blocked: {c.mean() - b.mean():+.3f} R, t {(c.mean() - b.mean()) / se:+.2f}")
        for y in ("2024", "2025", "2026"):
            s = df[df.year == y]
            print(f"  {y}: clear {summarize(s[~s.blocked])} | blocked {summarize(s[s.blocked])}")


if __name__ == "__main__":
    main()

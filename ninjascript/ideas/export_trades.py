"""List the trades the Python engine takes for one idea, to check a NinjaTrader port against.

    python export_trades.py --idea 10335 --csv es2026.csv --symbol es [--out trades.csv]

--idea is the id used by top_ideas.py (position in idea_engine.all_combos(), starting at 1).
Times are the New York start time of the signal bar; entry is the next bar's open.
"""
import argparse

import pandas as pd

import idea_engine as ie
from orb_volume_study import load


def detail(d, n, side, stop_kind, target):
    o, h, l, c, atrp = d["o"], d["h"], d["l"], d["c"], d["atrp"]
    N, entry, atr = len(c), o[n + 1], atrp[n]
    if not atr > 0:
        return None
    if stop_kind == "bar":
        dist = (entry - (l[n] - ie.TICK)) if side == 1 else ((h[n] + ie.TICK) - entry)
        dist = max(dist, 0.5 * atr)
    else:
        dist = atr
    stop = entry - side * dist
    tgt = None if target == "time" else entry + side * (1 if target == "1R" else 2) * dist
    exit_px, why, exit_slot = c[N - 1], "time", d["slots"][N - 1]
    for k in range(n + 1, N):
        if (side == 1 and l[k] <= stop) or (side == -1 and h[k] >= stop):
            exit_px, why, exit_slot = stop, "stop", d["slots"][k]
            break
        if tgt is not None and ((side == 1 and h[k] >= tgt) or (side == -1 and l[k] <= tgt)):
            exit_px, why, exit_slot = tgt, "target", d["slots"][k]
            break
    return dict(entry=entry, stop=stop, target=tgt, exit=exit_px, why=why, exit_bar=exit_slot,
                dist=dist, pts=side * (exit_px - entry))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", type=int, required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", choices=["es", "nq"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    combos = ie.all_combos()
    zone, trigger, stop_kind, target, window, vol, trend = combos[a.idea - 1]
    idea = pd.Series(dict(zone=zone, trigger=trigger, stop=stop_kind, target=target,
                          window=window, vol=vol, trend=trend))
    print(f"idea {a.idea}: {ie.describe(combos[a.idea - 1])}")

    rows = []
    for d in ie.prep_days(load(a.csv)):
        for n, side in ie.signals(d, zone, trigger):
            s = d["slots"][n]
            if window == "early" and s >= "10:00":
                continue
            if window == "late" and not ("10:00" <= s < "10:30"):
                continue
            if window == "all" and s >= "10:30":
                continue
            rv = d["rv"][n]
            if vol == "high" and not rv >= 1.2:
                continue
            if vol == "low" and not rv < 0.8:
                continue
            with_trend = (side == 1) == (d["c"][n] >= d["vwap"][n])
            if trend == "with_vwap" and not with_trend:
                continue
            if trend == "against_vwap" and with_trend:
                continue
            res = detail(d, n, side, stop_kind, target)
            if res is None:
                continue
            res.update(date=d["date"], side="long" if side == 1 else "short", signal_bar_start=s)
            res["net_pts"] = res["pts"] - ie.COST[a.symbol]
            rows.append(res)
            break

    df = pd.DataFrame(rows)[["date", "side", "signal_bar_start", "entry", "stop", "target", "exit",
                             "exit_bar", "why", "pts", "net_pts"]]
    print(f"{len(df)} trades, gross {df['pts'].sum():+.2f} pts, net {df['net_pts'].sum():+.2f} pts "
          f"(costs {ie.COST[a.symbol]} pts per trade)")
    print(df.head(12).round(2).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()

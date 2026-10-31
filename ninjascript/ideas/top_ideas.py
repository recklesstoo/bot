"""Measure every combination of idea_engine components and keep the 10 best, with a held-out year.

Selection uses 2025 and 2026 (ES and NQ pooled, in R per trade net of costs). An idea is eligible
if it has at least MIN_TRADES trades in each selection year; ideas are ranked by the lower of the
two years' t-statistics, so only ideas that work in both years rank high. The top 10 are then
measured on 2024, which plays no part in the selection.

    python top_ideas.py --data-dir DIR --out OUT
DIR must contain es/nq 2024, 2025 and 2026 CSVs (es2024.csv, nq2024.csv, ...).
"""
import argparse
import os

import numpy as np
import pandas as pd

import idea_engine as ie
from orb_volume_study import load

MIN_TRADES = 40


def measure(R):
    """R: ideas x days. Returns per-idea n, mean R, win rate, profit factor, t."""
    n = (~np.isnan(R)).sum(axis=1)
    x = np.where(np.isnan(R), 0.0, R)
    mean = np.divide(x.sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
    sd = np.sqrt(np.divide(((np.where(np.isnan(R), mean[:, None], R) - mean[:, None]) ** 2).sum(axis=1),
                           np.maximum(n - 1, 1)))
    t = np.divide(mean * np.sqrt(n), sd, out=np.full(len(n), np.nan), where=sd > 0)
    win = np.divide((np.nan_to_num(R, nan=-9) > 0).sum(axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
    pos, neg = np.where(x > 0, x, 0).sum(axis=1), -np.where(x < 0, x, 0).sum(axis=1)
    pf = np.divide(pos, neg, out=np.full(len(n), np.nan), where=neg > 0)
    return pd.DataFrame({"n": n, "meanR": mean, "win": win, "pf": pf, "t": t})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    combos = pd.DataFrame(ie.all_combos(), columns=ie.DIMS)
    combos.insert(0, "id", range(1, len(combos) + 1))
    print(f"combinations measured: {len(combos)}", flush=True)

    pooled = {}
    for year in ("2024", "2025", "2026"):
        parts = []
        for sym in ("es", "nq"):
            days = ie.prep_days(load(os.path.join(a.data_dir, f"{sym}{year}.csv")))
            parts.append(np.vstack([ie.run_idea(days, idea, ie.COST[sym]) for idea in combos.itertuples()]))
            if sym == "es" and year == "2025":
                es25 = parts[-1]
        pooled[year] = np.hstack(parts)
        print(f"{year}: {pooled[year].shape[1]} pooled sessions measured", flush=True)

    stats = {y: measure(pooled[y]).add_suffix(f"_{y}") for y in pooled}
    full = pd.concat([combos] + [stats[y] for y in ("2025", "2026", "2024")], axis=1)
    elig = (full["n_2025"] >= MIN_TRADES) & (full["n_2026"] >= MIN_TRADES)
    full["score"] = np.where(elig, np.fmin(full["t_2025"], full["t_2026"]), np.nan)
    ranked = full.dropna(subset=["score"]).sort_values("score", ascending=False)
    top = ranked.head(10)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print(f"\neligible (>= {MIN_TRADES} trades in 2025 and 2026): {int(elig.sum())}")
    print(f"ideas positive in both selection years: {int(((ranked.meanR_2025 > 0) & (ranked.meanR_2026 > 0)).sum())}")
    print("\n=== TOP 10 (ranked by the lower of the 2025 and 2026 t-statistics) ===")
    cols = ["id", "zone", "trigger", "stop", "target", "window", "vol", "trend"]
    print(top[cols].to_string(index=False))
    print("\n   id | 2025: n  meanR  win  PF  t | 2026: n  meanR  win  PF  t | 2024 (held out): n  meanR  win  PF  t")
    for r in top.itertuples():
        print(f"{r.id:>5} | {r.n_2025:>3} {r.meanR_2025:+.3f} {r.win_2025:.0%} {r.pf_2025:.2f} {r.t_2025:+.1f} "
              f"| {r.n_2026:>3} {r.meanR_2026:+.3f} {r.win_2026:.0%} {r.pf_2026:.2f} {r.t_2026:+.1f} "
              f"| {r.n_2024:>3} {r.meanR_2024:+.3f} {r.win_2024:.0%} {r.pf_2024:.2f} {r.t_2024:+.1f}")

    h = top["meanR_2024"]
    base = ranked[ranked.n_2024 >= 20]["meanR_2024"]
    print(f"\nTOP 10 on the held-out year: mean R {h.mean():+.3f}, positive {int((h > 0).sum())} of 10, "
          f"best t {top['t_2024'].max():+.1f}")
    print(f"all eligible ideas on the held-out year: mean R {base.mean():+.3f}, positive {(base > 0).mean():.0%}")
    print(f"distinct (zone, trigger) pairs among the top 10: {len(top[['zone', 'trigger']].drop_duplicates())}")

    # How good does the best of N look by chance? Same selection on the held-out year itself.
    best_in = ranked.sort_values("t_2024", ascending=False).head(10)
    print(f"for scale: the 10 best ideas inside 2024 alone reach t {best_in['t_2024'].min():+.1f} to "
          f"{best_in['t_2024'].max():+.1f} (best-of-{len(ranked)} look-back, no validation)")

    sample = np.random.default_rng(1).choice(np.where((~np.isnan(es25)).sum(axis=1) >= 15)[0], 2000, replace=False)
    C = np.corrcoef(np.nan_to_num(es25[sample]))
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(C, nan=1.0)), 0, None)
    print(f"effective independent ideas among a 2000-idea sample (ES 2025): {eig.sum() ** 2 / (eig ** 2).sum():.0f}")

    os.makedirs(a.out, exist_ok=True)
    top.to_csv(os.path.join(a.out, "top10.csv"), index=False)
    ranked.head(200).to_csv(os.path.join(a.out, "top200.csv"), index=False)


if __name__ == "__main__":
    main()

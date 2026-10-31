"""Generate a large set of intraday ideas from components and compare their rules and triggers.

An idea = zone x trigger x stop x target x window x volume filter x trend filter, all computed
from 5-minute bars of the first 90 minutes of the NY session (09:30-11:00). One trade per idea
per day: the first signal that passes the filters; entry at the next bar's open.

    python idea_engine.py generate ideas_2000.csv
    python idea_engine.py run ideas_2000.csv --data-dir DIR --out results

DIR must contain es2025.csv, es2026.csv, nq2025.csv, nq2026.csv (same format as
orb_volume_study.py). Costs: 0.6 points ES, 0.7 points NQ, converted to R per trade.
"""
import argparse
import warnings
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orb_volume_study import load  # noqa: E402

warnings.simplefilter("ignore")
TICK = 0.25
ZONES = ["orb5", "orb15", "orb30", "run_hl", "open", "vwap", "vwap_band", "ema9", "ema21"]
TRIGGERS = [
    "close_break", "break_2close", "break_retest", "failed_break_fade", "wick_sweep",
    "fvg_retest", "engulf_at_level", "pin_rejection", "volume_break", "thrust_break",
]
STOPS = ["bar", "atr"]
TARGETS = ["1R", "2R", "time"]
WINDOWS = ["all", "early", "late"]
VOLS = ["any", "high", "low"]
TRENDS = ["none", "with_vwap", "against_vwap"]
DIMS = ["zone", "trigger", "stop", "target", "window", "vol", "trend"]
COST = {"es": 0.6, "nq": 0.7}

ZONE_TXT = {
    "orb5": "the 5-minute opening range", "orb15": "the 15-minute opening range",
    "orb30": "the 30-minute opening range", "run_hl": "the running high/low of the session",
    "open": "the session open", "vwap": "session VWAP", "vwap_band": "the VWAP +/-1 sigma band",
    "ema9": "the 9-bar EMA", "ema21": "the 21-bar EMA",
}
TRIG_TXT = {
    "close_break": "first close beyond {z}", "break_2close": "two consecutive closes beyond {z}",
    "break_retest": "retest of {z} after a break, entering when it holds",
    "failed_break_fade": "fade a break of {z} once price closes back inside",
    "wick_sweep": "wick through {z} that closes back, trade the reversal",
    "fvg_retest": "fair value gap left by a break of {z}, enter on the retest of the gap",
    "engulf_at_level": "engulfing candle rejecting {z}", "pin_rejection": "pin bar rejecting {z}",
    "volume_break": "close beyond {z} on 1.5x relative volume",
    "thrust_break": "close beyond {z} on a bar of 1.5x average range",
}
WIN_TXT = {"all": "any time before 10:30", "early": "before 10:00", "late": "10:00-10:30"}
VOL_TXT = {"any": "no volume filter", "high": "signal bar volume >= 1.2x normal",
           "low": "signal bar volume < 0.8x normal"}
TREND_TXT = {"none": "no trend filter", "with_vwap": "only in the direction of price vs VWAP",
             "against_vwap": "only against price vs VWAP"}
STOP_TXT = {"bar": "stop beyond the signal bar", "atr": "stop 1 ATR away"}
TARGET_TXT = {"1R": "target 1R", "2R": "target 2R", "time": "no target, exit at 10:55"}


def all_combos():
    return list(itertools.product(ZONES, TRIGGERS, STOPS, TARGETS, WINDOWS, VOLS, TRENDS))


def describe(row):
    z, t, s, tg, w, v, tr = row
    return (f"{TRIG_TXT[t].format(z=ZONE_TXT[z])}; {WIN_TXT[w]}; {VOL_TXT[v]}; "
            f"{TREND_TXT[tr]}; {STOP_TXT[s]}; {TARGET_TXT[tg]}")


def generate(path, n=2000, seed=7):
    combos = all_combos()
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(combos), size=n, replace=False))
    rows = [combos[i] for i in pick]
    df = pd.DataFrame(rows, columns=DIMS)
    df.insert(0, "id", range(1, n + 1))
    df["description"] = [describe(r) for r in rows]
    df.to_csv(path, index=False)
    print(f"{len(combos)} possible combinations, {n} sampled -> {path}")


# ---------------------------------------------------------------- data preparation
def ema(x, span):
    a, out = 2 / (span + 1), np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def prep_days(df, lookback=20, min_day_ratio=0.5):
    df = df.between_time("09:30", "11:00", inclusive="left")
    df = df.assign(date=df.index.date, slot=df.index.strftime("%H:%M"))
    vol = df.pivot(index="date", columns="slot", values="volume").sort_index()
    base = vol.shift(1).rolling(lookback, min_periods=max(5, lookback // 2)).mean()
    day_vol = vol.sum(axis=1)
    thin = day_vol < min_day_ratio * day_vol.median()

    days = []
    for date, d in df.groupby("date"):
        if thin.get(date, False) or len(d) < 18:
            continue
        d = d.sort_index()
        o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        n = len(d)
        tp = (h + l + c) / 3
        cum_v = np.cumsum(v)
        vwap = np.cumsum(tp * v) / cum_v
        sig = np.sqrt(np.maximum(np.cumsum(v * tp ** 2) / cum_v - vwap ** 2, 0))
        rng = h - l
        atrp = np.full(n, np.nan)
        for i in range(1, n):
            atrp[i] = rng[max(0, i - 10):i].mean()
        slots = d["slot"].to_numpy()
        rv = np.array([v[i] / base.loc[date, slots[i]] if pd.notna(base.loc[date, slots[i]])
                       and base.loc[date, slots[i]] > 0 else np.nan for i in range(n)])

        zones = {}
        for name in ZONES:
            U, L = np.full(n, np.nan), np.full(n, np.nan)
            if name in ("orb5", "orb15", "orb30"):
                k = {"orb5": 1, "orb15": 3, "orb30": 6}[name]
                U[k:], L[k:] = h[:k].max(), l[:k].min()
            elif name == "run_hl":
                U[1:], L[1:] = np.maximum.accumulate(h)[:-1], np.minimum.accumulate(l)[:-1]
            elif name == "open":
                U[1:] = L[1:] = o[0]
            elif name == "vwap":
                U[1:] = L[1:] = vwap[:-1]
            elif name == "vwap_band":
                U[1:], L[1:] = vwap[:-1] + sig[:-1], vwap[:-1] - sig[:-1]
            else:
                e = ema(c, 9 if name == "ema9" else 21)
                U[3:] = L[3:] = e[2:-1]
            zones[name] = (U, L)
        days.append(dict(date=date, o=o, h=h, l=l, c=c, rng=rng, atrp=atrp, rv=rv,
                         vwap=vwap, slots=slots, zones=zones, memo={}, sims={}))
    return days


# ---------------------------------------------------------------- triggers
def signals(d, zone, trig):
    key = (zone, trig)
    if key in d["memo"]:
        return d["memo"][key]
    U, L = d["zones"][zone]
    o, h, l, c, rv, rng, atrp = (d[k] for k in ("o", "h", "l", "c", "rv", "rng", "atrp"))
    N, last, out = len(c), len(c) - 2, []
    up = [False] * N
    dn = [False] * N
    for n in range(1, N):
        up[n] = bool(c[n] > U[n] and c[n - 1] <= U[n])
        dn[n] = bool(c[n] < L[n] and c[n - 1] >= L[n])

    if trig == "close_break":
        out = [(n, s) for n in range(1, last + 1) for s, f in ((1, up), (-1, dn)) if f[n]]
    elif trig == "volume_break":
        out = [(n, s) for n in range(1, last + 1) for s, f in ((1, up), (-1, dn))
               if f[n] and rv[n] >= 1.5]
    elif trig == "thrust_break":
        out = [(n, s) for n in range(1, last + 1) for s, f in ((1, up), (-1, dn))
               if f[n] and rng[n] >= 1.5 * atrp[n]]
    elif trig == "break_2close":
        for n in range(2, last + 1):
            if c[n] > U[n] and c[n - 1] > U[n] and c[n - 2] <= U[n]:
                out.append((n, 1))
            if c[n] < L[n] and c[n - 1] < L[n] and c[n - 2] >= L[n]:
                out.append((n, -1))
    elif trig == "break_retest":
        bl = bs = False
        for n in range(1, last + 1):
            if up[n]:
                bl = True
            elif bl:
                if c[n] <= U[n]:
                    bl = False
                elif l[n] <= U[n]:
                    out.append((n, 1))
                    bl = False
            if dn[n]:
                bs = True
            elif bs:
                if c[n] >= L[n]:
                    bs = False
                elif h[n] >= L[n]:
                    out.append((n, -1))
                    bs = False
    elif trig == "failed_break_fade":
        bl = bs = False
        for n in range(1, last + 1):
            if up[n]:
                bl = True
            elif bl and c[n] <= U[n]:
                out.append((n, -1))
                bl = False
            if dn[n]:
                bs = True
            elif bs and c[n] >= L[n]:
                out.append((n, 1))
                bs = False
    elif trig == "wick_sweep":
        for n in range(1, last + 1):
            if h[n] > U[n] and c[n] < U[n] and c[n - 1] <= U[n]:
                out.append((n, -1))
            if l[n] < L[n] and c[n] > L[n] and c[n - 1] >= L[n]:
                out.append((n, 1))
    elif trig == "fvg_retest":
        for m in range(1, N - 3):
            for side, brk in ((1, up[m]), (-1, dn[m])):
                if not brk:
                    continue
                for k in (m, m + 1, m + 2):
                    if k < 1 or k + 1 >= N:
                        continue
                    if side == 1 and l[k + 1] > h[k - 1]:
                        g_lo, g_hi = h[k - 1], l[k + 1]
                    elif side == -1 and h[k + 1] < l[k - 1]:
                        g_lo, g_hi = h[k + 1], l[k - 1]
                    else:
                        continue
                    for n in range(k + 2, last + 1):
                        if (side == 1 and c[n] < g_lo) or (side == -1 and c[n] > g_hi):
                            break
                        if (side == 1 and l[n] <= g_hi) or (side == -1 and h[n] >= g_lo):
                            out.append((n, side))
                            break
                    break
    elif trig == "engulf_at_level":
        for n in range(1, last + 1):
            if (l[n] <= L[n] and c[n] > L[n] and c[n] > o[n] and c[n - 1] < o[n - 1]
                    and c[n] > o[n - 1] and o[n] <= c[n - 1]):
                out.append((n, 1))
            if (h[n] >= U[n] and c[n] < U[n] and c[n] < o[n] and c[n - 1] > o[n - 1]
                    and c[n] < o[n - 1] and o[n] >= c[n - 1]):
                out.append((n, -1))
    elif trig == "pin_rejection":
        for n in range(1, last + 1):
            body = max(abs(c[n] - o[n]), TICK)
            lw, uw = min(o[n], c[n]) - l[n], h[n] - max(o[n], c[n])
            if l[n] <= L[n] and c[n] > L[n] and lw >= 2 * body and lw >= 0.5 * rng[n]:
                out.append((n, 1))
            if h[n] >= U[n] and c[n] < U[n] and uw >= 2 * body and uw >= 0.5 * rng[n]:
                out.append((n, -1))
    out = sorted(set(out))
    d["memo"][key] = out
    return out


def simulate(d, n, side, stop_kind, target):
    key = (n, side, stop_kind, target)
    if key in d["sims"]:
        return d["sims"][key]
    o, h, l, c, atrp = d["o"], d["h"], d["l"], d["c"], d["atrp"]
    N, entry, atr = len(c), o[n + 1], d["atrp"][n]
    if not atr > 0:
        d["sims"][key] = None
        return None
    if stop_kind == "bar":
        dist = (entry - (l[n] - TICK)) if side == 1 else ((h[n] + TICK) - entry)
        dist = max(dist, 0.5 * atr)
    else:
        dist = atr
    stop = entry - side * dist
    tgt = None if target == "time" else entry + side * (1 if target == "1R" else 2) * dist
    exit_px = c[N - 1]
    for k in range(n + 1, N):
        if (side == 1 and l[k] <= stop) or (side == -1 and h[k] >= stop):
            exit_px = stop
            break
        if tgt is not None and ((side == 1 and h[k] >= tgt) or (side == -1 and l[k] <= tgt)):
            exit_px = tgt
            break
    res = (side * (exit_px - entry), dist)
    d["sims"][key] = res
    return res


def run_idea(days, idea, cost):
    out = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        for n, side in signals(d, idea.zone, idea.trigger):
            s = d["slots"][n]
            if idea.window == "early" and s >= "10:00":
                continue
            if idea.window == "late" and not ("10:00" <= s < "10:30"):
                continue
            if idea.window == "all" and s >= "10:30":
                continue
            rv = d["rv"][n]
            if idea.vol == "high" and not rv >= 1.2:
                continue
            if idea.vol == "low" and not rv < 0.8:
                continue
            above = d["c"][n] >= d["vwap"][n]
            with_trend = (side == 1) == above
            if idea.trend == "with_vwap" and not with_trend:
                continue
            if idea.trend == "against_vwap" and with_trend:
                continue
            res = simulate(d, n, side, idea.stop, idea.target)
            if res is not None:
                out[i] = (res[0] - cost) / res[1]
                break
    return out


# ---------------------------------------------------------------- analysis
def tstat(x):
    x = x[~np.isnan(x)]
    return np.nan if len(x) < 3 or x.std() == 0 else x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def pooled(r_a, r_b):
    return np.concatenate([r_a, r_b])


def analyse(ideas, R, out_dir):
    """R[name] -> array (ideas x days) of net R per day, NaN when no trade."""
    pools = {"2026": pooled(R["es2026"].T, R["nq2026"].T).T, "2025": pooled(R["es2025"].T, R["nq2025"].T).T}
    stats = pd.DataFrame({"id": ideas["id"]})
    for k, m in pools.items():
        stats[f"n_{k}"] = (~np.isnan(m)).sum(axis=1)
        stats[f"meanR_{k}"] = np.nanmean(np.where(np.isnan(m), np.nan, m), axis=1)
        stats[f"t_{k}"] = [tstat(row) for row in m]
    full = ideas.merge(stats, on="id")
    full.to_csv(os.path.join(out_dir, "idea_stats.csv"), index=False)
    print(f"ideas: {len(ideas)}; ideas that never trade in 2026: {(stats['n_2026'] == 0).sum()}\n")

    print("== SELECTION WITH A HELD-OUT PERIOD ==")
    for dev, hold in (("2026", "2025"), ("2025", "2026")):
        ok = full[(full[f"n_{dev}"] >= 40) & (full[f"meanR_{dev}"] > 0) & (full[f"t_{dev}"] >= 2.0)]
        elig = full[full[f"n_{dev}"] >= 40]
        hm = ok[ok[f"n_{hold}"] >= 20][f"meanR_{hold}"]
        allm = full[full[f"n_{hold}"] >= 20][f"meanR_{hold}"]
        pairs = full[(full["n_2026"] >= 30) & (full["n_2025"] >= 30)]
        rho = pairs[["meanR_2026", "meanR_2025"]].corr(method="spearman").iloc[0, 1]
        print(f"dev {dev}: eligible {len(elig)}, picked (n>=40, meanR>0, t>=2) {len(ok)}; "
              f"expected by chance alone about {0.023 * len(elig):.0f}")
        print(f"   hold-out {hold}: picked ideas mean R {hm.mean():+.3f} "
              f"(positive {int((hm > 0).sum())}/{len(hm)}); all ideas mean R {allm.mean():+.3f}")
        print(f"   rank correlation dev vs hold-out across ideas: {rho:+.2f}")
    print()

    # ---- coincidence in rules and triggers, on the 2026 ES days
    m = R["es2026"]
    keep = (~np.isnan(m)).sum(axis=1) >= 15
    X = np.nan_to_num(m[keep])
    sub = ideas[keep].reset_index(drop=True)
    C = np.corrcoef(X)
    np.fill_diagonal(C, np.nan)
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(C, nan=1.0)), 0, None)
    pr = eig.sum() ** 2 / (eig ** 2).sum()
    print(f"== COINCIDENCE (ES 2026 daily results, {keep.sum()} ideas with >=15 trades) ==")
    print(f"effective number of independent ideas (participation ratio): {pr:.0f} of {keep.sum()}")
    print(f"mean pairwise correlation: {np.nanmean(C):+.3f}\n")
    print("mean correlation between ideas that share a component vs all pairs:")
    rows = []
    for dim in DIMS:
        v = sub[dim].to_numpy()
        same = v[:, None] == v[None, :]
        mask = same & ~np.eye(len(v), dtype=bool)
        rows.append((dim, np.nanmean(C[mask]), np.nanmean(C[~same])))
    for dim, a, b in sorted(rows, key=lambda r: -r[1]):
        print(f"  same {dim:8s}: {a:+.3f}   different {dim:8s}: {b:+.3f}   gap {a - b:+.3f}")

    print("\nmost coincident (zone, trigger) pairs by shared entries (Jaccard on every signal, ES 2026):")
    days = prep_cache["es2026"]
    sets = {}
    for z in ZONES:
        for t in TRIGGERS:
            sets[(z, t)] = {(i, n, s) for i, d in enumerate(days) for n, s in signals(d, z, t)}
    keys = [k for k, v in sets.items() if len(v) >= 20]
    pairs = []
    for a, b in itertools.combinations(keys, 2):
        u = len(sets[a] | sets[b])
        pairs.append((len(sets[a] & sets[b]) / u, a, b))
    for j, a, b in sorted(pairs, reverse=True)[:12]:
        print(f"  {j:.2f}  {a[0]}/{a[1]}  ~  {b[0]}/{b[1]}")
    print(f"\nshare of (zone,trigger) pairs with Jaccard >= 0.5: "
          f"{np.mean([p[0] >= 0.5 for p in pairs]):.0%} of {len(pairs)}")


prep_cache = {}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("path")
    g.add_argument("--n", type=int, default=2000)
    r = sub.add_parser("run")
    r.add_argument("ideas")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.path, a.n)
        return
    ideas = pd.read_csv(a.ideas)
    R = {}
    for name in ("es2025", "es2026", "nq2025", "nq2026"):
        days = prep_days(load(os.path.join(a.data_dir, f"{name}.csv")))
        prep_cache[name] = days
        cost = COST[name[:2]]
        R[name] = np.vstack([run_idea(days, idea, cost) for idea in ideas.itertuples()])
        print(f"{name}: {len(days)} sessions, {int((~np.isnan(R[name])).sum())} trades", flush=True)
    os.makedirs(a.out, exist_ok=True)
    analyse(ideas, R, a.out)


if __name__ == "__main__":
    main()

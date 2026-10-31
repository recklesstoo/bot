# 2,000 intraday ideas: comparison of rules and triggers

Ideas are combinations of 9 zones x 10 triggers x 2 stops x 3 targets x 3 time windows x 3 volume
filters x 3 trend filters (14,580 possible, 2,000 sampled with seed 7). Data: ES and NQ 5-minute
bars, 09:30-11:00 New York, March-October 2025 and 2026 (about 630 sessions per instrument-year pair).
Costs: 0.6 points ES, 0.7 points NQ. One trade per idea per day, entry at the next open.

Run: `python idea_engine.py run ideas_2000.csv --data-dir DIR --out OUT` (about 5 seconds).

## Held-out selection
| Selected on | Picked (n>=40, mean R>0, t>=2) | Expected by chance | Picked ideas in the other period | All ideas in the other period |
|---|---|---|---|---|
| 2026 | 17 | ~15 | -0.053 R (6 of 17 positive) | -0.014 R |
| 2025 | 6 | ~15 | -0.009 R (4 of 6 positive) | -0.078 R |

Rank correlation of mean R between periods across ideas: +0.01 in both directions.

## Coincidence
- Effective number of independent ideas (participation ratio of the daily-result correlation
  matrix, ES 2026, 896 ideas with >=15 trades): 45.
- Mean correlation between ideas that share a component, against ideas that do not:
  trigger +0.131 vs +0.045, zone +0.091 vs +0.049, volume filter +0.077 vs +0.040,
  window +0.077 vs +0.042, trend filter +0.074 vs +0.041, target +0.063 vs +0.049,
  stop +0.057 vs +0.051.
- Most overlapping (zone, trigger) pairs: EMA9 / EMA21 / VWAP zones with the same trigger
  (Jaccard 0.6-0.76), and close_break vs failed_break_fade on moving levels (0.64-0.68).

## Limits
Only the first 90 minutes of the session. 460 of the 2,000 ideas never trade in 2026. ES and NQ
results on the same days are strongly correlated, so pooled t-statistics are optimistic.

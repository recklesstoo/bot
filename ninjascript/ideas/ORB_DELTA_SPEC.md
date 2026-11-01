# ORB + opening delta: the strategy to hand to TraderOS

Hypothesis (fixed before any data is looked at): the aggressor delta of the first 15 minutes tells whether a
15-minute opening range break holds. Two opposite variants, because either could be true:

- **Confirm**: take the first break only when the sign of the delta matches the break direction.
- **Fade**: take it only when the sign is opposite.
- **Off**: plain 15-minute ORB, the benchmark.

Rules (ES and NQ, 09:30 New York)
- Range: high/low of 09:30-09:45. Delta: aggressor buy volume minus sell volume, 09:30-09:45.
- Signal: the first 5-minute close outside the range before 10:30. Skip the day if delta is exactly 0.
- Entry: next bar open. Stop: opposite side of the range. Target: 2R. Flat at 11:00. One trade per day.
- Costs: 0.6 points ES, 0.7 points NQ per trade.
- Success criterion: Confirm or Fade must beat Off by a margin the data can see, in the same direction in
  every year available, and survive `quant_validate_walk_forward` and `quant_validate_deflated_sharpe`
  with 3 variants counted as trials.

Data limits (checked on 2026-11-01)
- TraderOS holds time and sales for ES only from 2026-01-29 to 2026-10-30, so this is a one-year test.
  Longer history needs tick data from elsewhere (NinjaTrader tick replay does it: `OrbDelta.cs`).
- market_trades returns at most 60 minutes per call, so each day costs one call.

Files
- `ninjascript/OrbDelta.cs`: the strategy for NinjaTrader (not compiled or tested yet).
- Reference trade logic: `structure_orb_study.py` (`trade`, 3 bars = 15 minutes) in this folder.

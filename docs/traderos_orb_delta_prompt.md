# Paste into the TraderOS app chat

I want to test one hypothesis with fixed rules. Please do not tune anything; run exactly this and report.

**Hypothesis.** The aggressor delta of the first 15 minutes (09:30-09:45 New York) tells whether a
15-minute opening range break holds.

**Three variants, same trades otherwise**
1. Off: plain 15-minute opening range breakout (benchmark).
2. Confirm: take the break only if the sign of the 09:30-09:45 delta equals the break direction.
3. Fade: take the break only if the sign of that delta is opposite to the break direction.

**Rules (ES and NQ, futures, regular hours)**
- Range: high and low of 09:30-09:45 ET. Delta = aggressor buy volume minus sell volume in that window
  (from time and sales). Skip the day if delta is exactly 0.
- Signal: first 5-minute bar that closes outside the range, starting 09:45, no later than 10:30.
- Entry: open of the next 5-minute bar. One trade per day.
- Stop: opposite side of the opening range. Target: 2R. Flat at 11:00 ET.
- Costs: 0.6 points per trade on ES, 0.7 on NQ.

**What I need back**
- Number of trades, win rate, mean R and total points per variant, split by year and by instrument.
- Whether Confirm or Fade beats Off with the same sign in every year available.
- `quant_validate_walk_forward` and `quant_validate_deflated_sharpe` for the best variant, counting
  3 variants as trials.
- Say plainly which data you did not have. Your time and sales for ES start 2026-01-29, so this is a one-year
  test; do not extend it with other data.

**If you cannot run the delta filter natively**
The filter catalog's flow family reads prior-day options flow (`flow_net_premium_prior`), not intraday
futures delta. If the opening-range delta is not available as a trade feature, tell me, and run only the
Off benchmark so I can compare it with my own number: 15-minute ORB, 2R, flat 11:00, March to October 2026,
ES and NQ together, net of costs.

**Success criterion, set before the test:** Confirm or Fade must beat Off in every available year and pass
both validators. Anything less counts as no edge.

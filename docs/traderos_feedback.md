# TraderOS feedback from a research session (MCP client)

Context: a long session testing intraday opening-range ideas on ES and NQ through the TraderOS MCP
tools (`query_data`, `market_ohlcv`, `market_orb_study`, `market_coverage`, `list_strategies`,
`run_pine_script`, `run_code`). Everything below was observed, and the evidence is in the session
logs. Nothing here depends on a particular strategy.

## Bugs and inconsistencies

1. **`query_data` repeats every row in the response.** A query returning 396 rows came back as
   95,163 characters: the 396 rows sit once in `artifacts[0].payload.rows` and again in
   `frames[].payload.rows`, and 50 more in `data.rows`. The rows alone are about 47,000 characters.
   Any query above roughly 150 rows exceeds the client token limit and is spilled to a file.
   Rows should be returned once.
2. **`market_ohlcv` returns 10 bars, not the window.** For ES, 5m, `days=120`:
   `summary.bars = 23568` but `recent_bars` has 10 entries. The description says the tool is for
   "raw bars", and there is no way to page through the rest.
3. **Continuous futures volume drops in the weeks before quarterly expiry.** ES 09:30-11:00 volume
   per day (median 362,602): 2026-03-17/18/19 = 52k/37k/47k, 2026-06-15..19 = 169k/41k/33k/-/27k,
   2026-09-14/15 = 172k/56k. The series appears to follow the expiring contract until the roll
   date while volume has moved to the next one. Holidays also show (2026-05-25, 07-03, 09-07).
   Volume-based studies on these days are wrong without a manual filter. A roll rule by volume or
   open interest, and a `contract` or `roll_week` marker, would fix it. The rule used should be
   documented next to `contract_stitching`.
4. **`market_coverage` readiness contradicts the depth block.** MES `ohlcv.history_start` says
   `2026-10-17`, while `bar_depth.MES.1m.earliest` says `2019-05-05`. The note explains it, but the
   field name invites the wrong reading. The call also took about 29 s, and the options-chain
   probe returned "unverified".
5. **Boilerplate on every `query_data` response.** `volume_scope` (Databento US equities feed,
   EQUS.MINI), `ranking_basis`, `intraday_history` and `paging` add about 1,900 characters per
   call, including for futures where `applies_to` says the note does not apply.
6. **`market_orb_study` definitions are not stated.** For the standard report: what `held` means,
   how direction is assigned on `double_break` days, the time zone of the session, and that the
   default break criterion is `by-wick`. `by_day` has no time of the break, so the question "when
   does the breakout happen" cannot be answered from it.
7. **Intermittent disconnects.** Within one session the `traderos` tool list was reported as
   unavailable and available again several times. This may be the client or proxy, not the server.

## Improvements, by how much time they would have saved

1. **Let `run_code` read TraderOS data.** The sandbox has no data access, so analysing 3,000
   five-minute bars meant copying rows through `input_data` (2 MB cap) or working outside
   TraderOS. A `data` argument (ticker, timeframe, range, session) that mounts the bars as a file
   would make `run_code` usable for real studies. A CSV/parquet export of a query result would do
   the same job.
2. **Session and time-zone aware queries.** `query_data` works in UTC, so selecting 09:30-11:00
   New York needs hand-written hour filters that break across the DST changes. A `session`
   parameter (RTH, ETH, or an exchange-local time range) would remove this.
3. **More SQL.** No window functions or joins: opening-range logic (first close outside the range,
   then the next bar) cannot be expressed. `lag`/`lead`/`row_number` would cover most of it. Also
   a higher row cap with file output.
4. **Summary mode for studies.** `market_orb_study` returned 84,417 characters (ES) and 86,051
   (NQ) because of the `by_day` log. A `summary_only` or `limit` option, plus time of first break,
   break-and-retest and a volume bucket, would make it directly useful.
5. **Strategy creation from MCP clients.** `backtest_run` needs a session, but no tool in the MCP
   list creates or selects one. `list_strategies` returns `strategy_id`, `saved_strategy_id` and
   `lineage_id`, and it is not stated which one `backtest_run(session_id)` expects.
6. **`run_pine_script` bar cap.** 10,000 bars is about 36 sessions at 5m when extended hours are
   included. RTH-only staging, or chunked runs, would allow multi-year intraday tests. Returning
   the trade list would allow checking a port against another engine.
7. **Multiple-testing by default in sweeps.** A search over 14,580 combinations found 10
   "best" ideas with t of 1.7 to 3.1 in two years that averaged -0.033 R in a third. The sweep tools
   could report the effective number of independent trials and keep a held-out period by default.

## What worked well
`query_data` is fast (15-50 ms per call), data goes back to 2016 for ES and NQ, the data-honesty
notes are clear, and artifacts make large results recoverable.

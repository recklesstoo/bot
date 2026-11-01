#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

// Opening range breakout filtered by the order flow delta of the opening range.
//
// Delta = aggressor buy volume minus aggressor sell volume between RangeStartTime and RangeEndTime.
// A last trade at or above the ask counts as a buy, at or below the bid as a sell.
// DeltaMode:
//   Off      plain ORB (for comparison)
//   Confirm  trade the break only when the sign of the delta matches the break direction
//   Fade     trade the break only when the sign of the delta is opposite to the break direction
// Entry: first bar close beyond the range after RangeEndTime, market order. Stop: opposite side of the
// range. Target: RewardRisk x risk. Flat at ExitTime. One trade per day.
//
// Needs Tick Replay for backtests (Strategy Analyzer: tick replay on, or a tick data series), because
// bid/ask are needed for every trade. Use a 5-minute chart. Not compiled or tested yet.
namespace NinjaTrader.NinjaScript.Strategies
{
	public enum OrbDeltaMode { Off, Confirm, Fade }

	public class OrbDelta : Strategy
	{
		private double rangeHigh, rangeLow;
		private bool rangeSet, tradedToday;
		private DateTime currentDate = DateTime.MinValue;
		private double bid, ask;
		private double buyVol, sellVol;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = "Opening range breakout with order flow delta filter.";
				Name = "OrbDelta";
				Calculate = Calculate.OnBarClose;
				EntriesPerDirection = 1;
				EntryHandling = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds = 30;
				StartBehavior = StartBehavior.WaitUntilFlat;
				TimeInForce = TimeInForce.Gtc;
				RealtimeErrorHandling = RealtimeErrorHandling.StopCancelCloseIgnoreRejects;
				StopTargetHandling = StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade = 20;

				RangeStartTime = 93000;
				RangeEndTime = 94500;
				LastEntryTime = 103000;
				ExitTime = 110000;
				RewardRisk = 2.0;
				Quantity = 1;
				MinDeltaPct = 0;
				Mode = OrbDeltaMode.Confirm;
			}
		}

		protected override void OnMarketData(MarketDataEventArgs e)
		{
			if (e.MarketDataType == MarketDataType.Bid) { bid = e.Price; return; }
			if (e.MarketDataType == MarketDataType.Ask) { ask = e.Price; return; }
			if (e.MarketDataType != MarketDataType.Last)
				return;

			int t = ToTime(e.Time);
			if (t <= RangeStartTime || t > RangeEndTime)
				return;
			if (ask > 0 && e.Price >= ask)
				buyVol += e.Volume;
			else if (bid > 0 && e.Price <= bid)
				sellVol += e.Volume;
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBar < BarsRequiredToTrade)
				return;

			if (Time[0].Date != currentDate)
			{
				currentDate = Time[0].Date;
				rangeHigh = double.MinValue;
				rangeLow = double.MaxValue;
				rangeSet = false;
				tradedToday = false;
				buyVol = 0;
				sellVol = 0;
			}

			int t = ToTime(Time[0]);
			if (t > RangeStartTime && t <= RangeEndTime)
			{
				rangeHigh = Math.Max(rangeHigh, High[0]);
				rangeLow = Math.Min(rangeLow, Low[0]);
				return;
			}
			if (!rangeSet && t > RangeEndTime && rangeHigh > rangeLow)
				rangeSet = true;

			if (t >= ExitTime)
			{
				if (Position.MarketPosition == MarketPosition.Long)
					ExitLong("ExitTime", "ORB Long");
				else if (Position.MarketPosition == MarketPosition.Short)
					ExitShort("ExitTime", "ORB Short");
				return;
			}
			if (!rangeSet || tradedToday || Position.MarketPosition != MarketPosition.Flat || t > LastEntryTime)
				return;

			int side = Close[0] > rangeHigh ? 1 : (Close[0] < rangeLow ? -1 : 0);
			if (side == 0)
				return;

			double total = buyVol + sellVol;
			double delta = buyVol - sellVol;
			double pct = total > 0 ? Math.Abs(delta) / total * 100.0 : 0;
			int deltaSide = delta > 0 ? 1 : (delta < 0 ? -1 : 0);

			bool ok = true;
			if (Mode != OrbDeltaMode.Off)
			{
				if (total <= 0 || pct < MinDeltaPct)
					ok = false;
				else if (Mode == OrbDeltaMode.Confirm)
					ok = deltaSide == side;
				else
					ok = deltaSide == -side;
			}
			if (!ok)
			{
				tradedToday = true; // first break only, one decision per day
				return;
			}

			if (side == 1)
			{
				double risk = Close[0] - rangeLow;
				SetStopLoss("ORB Long", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(rangeLow), false);
				SetProfitTarget("ORB Long", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(Close[0] + RewardRisk * risk));
				EnterLong(Quantity, "ORB Long");
			}
			else
			{
				double risk = rangeHigh - Close[0];
				SetStopLoss("ORB Short", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(rangeHigh), false);
				SetProfitTarget("ORB Short", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(Close[0] - RewardRisk * risk));
				EnterShort(Quantity, "ORB Short");
			}
			tradedToday = true;
		}

		#region Properties
		[NinjaScriptProperty]
		[Display(Name = "Delta mode", Order = 1, GroupName = "Parameters")]
		public OrbDeltaMode Mode { get; set; }

		[NinjaScriptProperty]
		[Range(0, 100)]
		[Display(Name = "Min |delta| as % of volume", Order = 2, GroupName = "Parameters")]
		public double MinDeltaPct { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Range start (HHmmss)", Order = 3, GroupName = "Parameters")]
		public int RangeStartTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Range end (HHmmss)", Order = 4, GroupName = "Parameters")]
		public int RangeEndTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Last entry (HHmmss)", Order = 5, GroupName = "Parameters")]
		public int LastEntryTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Exit time (HHmmss)", Order = 6, GroupName = "Parameters")]
		public int ExitTime { get; set; }

		[NinjaScriptProperty]
		[Range(0.1, 10)]
		[Display(Name = "Reward / risk", Order = 7, GroupName = "Parameters")]
		public double RewardRisk { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Quantity", Order = 8, GroupName = "Parameters")]
		public int Quantity { get; set; }
		#endregion
	}
}

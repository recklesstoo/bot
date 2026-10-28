#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

// Opening Range Breakout, one trade per day.
// Range = high/low of the bars between RangeStartTime and RangeEndTime (chart time zone).
// Entry: first bar close beyond the range after RangeEndTime (market order on the next bar).
// Stop: opposite side of the range. Target: RewardRisk x initial risk. Flat at ExitTime.
// Meant for a 1-5 minute chart. Times are HHmmss in the chart's time zone.
namespace NinjaTrader.NinjaScript.Strategies
{
	public class SimpleORB : Strategy
	{
		private double rangeHigh;
		private double rangeLow;
		private bool rangeSet;
		private bool tradedToday;
		private DateTime currentDate = DateTime.MinValue;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = "Simple opening range breakout, one trade per day.";
				Name = "SimpleORB";
				Calculate = Calculate.OnBarClose;
				EntriesPerDirection = 1;
				EntryHandling = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds = 30;
				IsFillLimitOnTouch = false;
				MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution = OrderFillResolution.Standard;
				Slippage = 0;
				StartBehavior = StartBehavior.WaitUntilFlat;
				TimeInForce = TimeInForce.Gtc;
				TraceOrders = false;
				RealtimeErrorHandling = RealtimeErrorHandling.StopCancelCloseIgnoreRejects;
				StopTargetHandling = StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade = 20;

				RangeStartTime = 93000;
				RangeEndTime = 93500;
				LastEntryTime = 110000;
				ExitTime = 155500;
				RewardRisk = 2.0;
				Quantity = 1;
				MinRangeTicks = 8;
				MaxRangeTicks = 80;
			}
			else if (State == State.Configure)
			{
				// Opening range needs intraday bars.
			}
			else if (State == State.DataLoaded)
			{
				if (BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value > 5)
					Print("SimpleORB: use a 1-5 minute chart. Current: " + BarsPeriod.Value + " " + BarsPeriod.BarsPeriodType);
			}
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
			}

			// Bars are stamped with their close time, so a 09:30-09:35 range
			// is made of bars with a timestamp in (09:30:00, 09:35:00].
			int t = ToTime(Time[0]);

			if (t > RangeStartTime && t <= RangeEndTime)
			{
				rangeHigh = Math.Max(rangeHigh, High[0]);
				rangeLow = Math.Min(rangeLow, Low[0]);
				return;
			}

			if (!rangeSet && t > RangeEndTime && rangeHigh > rangeLow)
			{
				double rangeTicks = (rangeHigh - rangeLow) / TickSize;
				rangeSet = true;
				if (rangeTicks < MinRangeTicks || rangeTicks > MaxRangeTicks)
					tradedToday = true; // range outside limits: skip the day
			}

			if (t >= ExitTime)
			{
				if (Position.MarketPosition == MarketPosition.Long)
					ExitLong("ExitTime", "ORB Long");
				else if (Position.MarketPosition == MarketPosition.Short)
					ExitShort("ExitTime", "ORB Short");
				return;
			}

			if (!rangeSet || tradedToday || Position.MarketPosition != MarketPosition.Flat)
				return;
			if (t > LastEntryTime)
				return;

			if (Close[0] > rangeHigh)
			{
				double risk = Close[0] - rangeLow;
				SetStopLoss("ORB Long", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(rangeLow), false);
				SetProfitTarget("ORB Long", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(Close[0] + RewardRisk * risk));
				EnterLong(Quantity, "ORB Long");
				tradedToday = true;
			}
			else if (Close[0] < rangeLow)
			{
				double risk = rangeHigh - Close[0];
				SetStopLoss("ORB Short", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(rangeHigh), false);
				SetProfitTarget("ORB Short", CalculationMode.Price, Instrument.MasterInstrument.RoundToTickSize(Close[0] - RewardRisk * risk));
				EnterShort(Quantity, "ORB Short");
				tradedToday = true;
			}
		}

		#region Properties
		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Range start (HHmmss)", Order = 1, GroupName = "Parameters")]
		public int RangeStartTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Range end (HHmmss)", Order = 2, GroupName = "Parameters")]
		public int RangeEndTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Last entry (HHmmss)", Order = 3, GroupName = "Parameters")]
		public int LastEntryTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Exit time (HHmmss)", Order = 4, GroupName = "Parameters")]
		public int ExitTime { get; set; }

		[NinjaScriptProperty]
		[Range(0.1, 10)]
		[Display(Name = "Reward / risk", Order = 5, GroupName = "Parameters")]
		public double RewardRisk { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Quantity", Order = 6, GroupName = "Parameters")]
		public int Quantity { get; set; }

		[NinjaScriptProperty]
		[Range(1, 10000)]
		[Display(Name = "Min range (ticks)", Order = 7, GroupName = "Parameters")]
		public int MinRangeTicks { get; set; }

		[NinjaScriptProperty]
		[Range(1, 10000)]
		[Display(Name = "Max range (ticks)", Order = 8, GroupName = "Parameters")]
		public int MaxRangeTicks { get; set; }
		#endregion
	}
}

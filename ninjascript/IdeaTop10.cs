#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

// The 10 best ideas of ideas/top_ideas.py as one NinjaScript strategy (unmanaged orders).
// Port of ideas/idea_engine.py: 5-minute bars, first 90 minutes of the NY session (09:30-11:00),
// one trade per day, entry at the open of the bar after the signal, flat at 11:00.
//
// Chart: ES, MES, NQ or MNQ, 5 minute bars, regular trading hours (first bar closes 09:35 NY).
// TimeOffsetHours converts chart time to New York time (0 if the chart already shows NY time).
//
// Preset  zone        trigger            stop  target  window  trend
//   1     VwapBand    FailedBreakFade    Atr   TwoR    Late    AgainstVwap   (idea 10335)
//   2     VwapBand    FailedBreakFade    Atr   Time    Late    AgainstVwap   (idea 10362)
//   3     VwapBand    FailedBreakFade    Atr   Time    Late    None          (idea 10360)
//   4     VwapBand    FailedBreakFade    Atr   OneR    Late    AgainstVwap   (idea 10308)
//   5     VwapBand    FailedBreakFade    Atr   TwoR    Late    None          (idea 10333)
//   6     VwapBand    FailedBreakFade    Bar   Time    Late    None          (idea 10279)
//   7     VwapBand    FailedBreakFade    Bar   TwoR    Late    None          (idea 10252)
//   8     Ema9        CloseBreak         Atr   Time    Late    None          (idea 11494)
//   9     VwapBand    FailedBreakFade    Bar   TwoR    Late    AgainstVwap   (idea 10254)
//  10     VwapBand    FailedBreakFade    Atr   OneR    Late    None          (idea 10306)
// Preset 0 uses the individual parameters below.
//
// Rules, all evaluated at the close of a 5-minute bar n (bar 0 is the 09:30 bar):
//  - VwapBand levels: session VWAP +/- 1 volume-weighted sigma of the typical price, as of bar n-1.
//    Ema9 level: 9-bar EMA of closes restarted each session, as of bar n-1 (from bar 3).
//  - Break up: close > upper level and previous close <= upper level. Break down: mirror.
//  - FailedBreakFade: after a break up, the first close <= the upper level is a short signal
//    (after a break down, the first close >= the lower level is a long signal).
//  - CloseBreak: a break up is a long signal, a break down is a short signal.
//  - Window: Early = signal bar starts before 10:00, Late = 10:00 to 10:30, Any = before 10:30.
//  - Trend filter compares the signal bar close with VWAP.
//  - ATR = mean range of the previous 10 bars (fewer at the start of the session).
//  - Stop: Atr = 1 ATR from the fill; SignalBar = beyond the signal bar extreme by one tick, but
//    never closer than 0.5 ATR. Target = 1R or 2R of that distance, or none (exit at 11:00).
namespace NinjaTrader.NinjaScript.Strategies
{
	public enum IdeaZone { VwapBand, Ema9 }
	public enum IdeaTrigger { FailedBreakFade, CloseBreak }
	public enum IdeaStop { SignalBar, Atr }
	public enum IdeaTarget { OneR, TwoR, TimeExit }
	public enum IdeaWindow { Any, Early, Late }
	public enum IdeaTrend { None, WithVwap, AgainstVwap }

	public class IdeaTop10 : Strategy
	{
		private const int SessionBars = 18;   // 09:30 to 11:00 in 5-minute bars

		private double[] hi = new double[SessionBars];
		private double[] lo = new double[SessionBars];
		private double[] cl = new double[SessionBars];
		private double[] rg = new double[SessionBars];
		private double[] vwapA = new double[SessionBars];
		private double[] sigA = new double[SessionBars];
		private double[] emaA = new double[SessionBars];
		private double cumV, cumTPV, cumTP2V, emaVal;
		private bool tradedToday, brokeUp, brokeDown, exitSent, badPeriod;

		private int pendSide;
		private double pendHigh, pendLow, pendAtr;

		private IdeaZone zoneE;
		private IdeaTrigger triggerE;
		private IdeaStop stopE;
		private IdeaTarget targetE;
		private IdeaWindow windowE;
		private IdeaTrend trendE;

		private Order entryOrder, stopOrder, targetOrder;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = "Top 10 ideas of the idea study, 5-minute bars, first 90 minutes of the NY session.";
				Name = "IdeaTop10";
				Calculate = Calculate.OnBarClose;
				IsUnmanaged = true;
				IsExitOnSessionCloseStrategy = false;
				BarsRequiredToTrade = 0;
				TraceOrders = false;

				Preset = 1;
				Quantity = 1;
				TimeOffsetHours = 0;
				ZoneSel = IdeaZone.VwapBand;
				TriggerSel = IdeaTrigger.FailedBreakFade;
				StopSel = IdeaStop.Atr;
				TargetSel = IdeaTarget.TwoR;
				WindowSel = IdeaWindow.Late;
				TrendSel = IdeaTrend.AgainstVwap;
			}
			else if (State == State.DataLoaded)
			{
				badPeriod = BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 5;
				if (badPeriod)
					Print("IdeaTop10: use a 5 minute chart. Current: " + BarsPeriod.Value + " " + BarsPeriod.BarsPeriodType);

				zoneE = ZoneSel; triggerE = TriggerSel; stopE = StopSel;
				targetE = TargetSel; windowE = WindowSel; trendE = TrendSel;
				if (Preset >= 1 && Preset <= 10)
					ApplyPreset(Preset);
			}
		}

		private void ApplyPreset(int p)
		{
			zoneE = IdeaZone.VwapBand;
			triggerE = IdeaTrigger.FailedBreakFade;
			windowE = IdeaWindow.Late;
			switch (p)
			{
				case 1:  stopE = IdeaStop.Atr;       targetE = IdeaTarget.TwoR;      trendE = IdeaTrend.AgainstVwap; break;
				case 2:  stopE = IdeaStop.Atr;       targetE = IdeaTarget.TimeExit;  trendE = IdeaTrend.AgainstVwap; break;
				case 3:  stopE = IdeaStop.Atr;       targetE = IdeaTarget.TimeExit;  trendE = IdeaTrend.None;        break;
				case 4:  stopE = IdeaStop.Atr;       targetE = IdeaTarget.OneR;      trendE = IdeaTrend.AgainstVwap; break;
				case 5:  stopE = IdeaStop.Atr;       targetE = IdeaTarget.TwoR;      trendE = IdeaTrend.None;        break;
				case 6:  stopE = IdeaStop.SignalBar; targetE = IdeaTarget.TimeExit;  trendE = IdeaTrend.None;        break;
				case 7:  stopE = IdeaStop.SignalBar; targetE = IdeaTarget.TwoR;      trendE = IdeaTrend.None;        break;
				case 8:
					zoneE = IdeaZone.Ema9; triggerE = IdeaTrigger.CloseBreak;
					stopE = IdeaStop.Atr; targetE = IdeaTarget.TimeExit; trendE = IdeaTrend.None;
					break;
				case 9:  stopE = IdeaStop.SignalBar; targetE = IdeaTarget.TwoR;      trendE = IdeaTrend.AgainstVwap; break;
				default: stopE = IdeaStop.Atr;       targetE = IdeaTarget.OneR;      trendE = IdeaTrend.None;        break;
			}
		}

		protected override void OnBarUpdate()
		{
			if (BarsInProgress != 0 || badPeriod)
				return;

			DateTime ny = Time[0].AddHours(TimeOffsetHours);
			int n = (int)Math.Round((ny.TimeOfDay.TotalMinutes - 5.0 - 570.0) / 5.0);
			if (n < 0 || n >= SessionBars)
				return;

			if (n == 0)
			{
				tradedToday = false;
				brokeUp = false;
				brokeDown = false;
				exitSent = false;
				cumV = 0; cumTPV = 0; cumTP2V = 0;
				if (Position.MarketPosition != MarketPosition.Flat)
					FlattenAll();
			}

			hi[n] = High[0];
			lo[n] = Low[0];
			cl[n] = Close[0];
			rg[n] = High[0] - Low[0];
			double tp = (High[0] + Low[0] + Close[0]) / 3.0;
			double v = Volume[0];
			cumV += v; cumTPV += tp * v; cumTP2V += tp * tp * v;
			double vw = cumV > 0 ? cumTPV / cumV : Close[0];
			double variance = cumV > 0 ? cumTP2V / cumV - vw * vw : 0;
			vwapA[n] = vw;
			sigA[n] = Math.Sqrt(Math.Max(variance, 0));
			emaVal = (n == 0) ? Close[0] : 0.2 * Close[0] + 0.8 * emaVal;
			emaA[n] = emaVal;

			if (n == SessionBars - 1)
			{
				if (Position.MarketPosition != MarketPosition.Flat && !exitSent)
					FlattenAll();
				return;
			}
			if (n < 1)
				return;

			double upper, lower;
			if (zoneE == IdeaZone.VwapBand)
			{
				upper = vwapA[n - 1] + sigA[n - 1];
				lower = vwapA[n - 1] - sigA[n - 1];
			}
			else
			{
				if (n < 3)
					return;
				upper = emaA[n - 1];
				lower = emaA[n - 1];
			}

			bool up = cl[n] > upper && cl[n - 1] <= upper;
			bool down = cl[n] < lower && cl[n - 1] >= lower;
			bool signalShort = false;
			bool signalLong = false;

			if (triggerE == IdeaTrigger.FailedBreakFade)
			{
				if (up)
					brokeUp = true;
				else if (brokeUp && cl[n] <= upper)
				{
					signalShort = true;
					brokeUp = false;
				}
				if (down)
					brokeDown = true;
				else if (brokeDown && cl[n] >= lower)
				{
					signalLong = true;
					brokeDown = false;
				}
			}
			else
			{
				signalLong = up;
				signalShort = down;
			}

			if (tradedToday || !InWindow(n))
				return;

			if (signalShort && TrendOk(n, -1) && TryEnter(-1, n))
				return;
			if (signalLong && TrendOk(n, 1) && TryEnter(1, n))
				return;
		}

		private bool InWindow(int n)
		{
			if (windowE == IdeaWindow.Early)
				return n <= 5;
			if (windowE == IdeaWindow.Late)
				return n >= 6 && n <= 11;
			return n <= 11;
		}

		private bool TrendOk(int n, int side)
		{
			bool above = cl[n] >= vwapA[n];
			bool withTrend = (side == 1) == above;
			if (trendE == IdeaTrend.WithVwap)
				return withTrend;
			if (trendE == IdeaTrend.AgainstVwap)
				return !withTrend;
			return true;
		}

		private bool TryEnter(int side, int n)
		{
			double sum = 0;
			int count = 0;
			for (int i = Math.Max(0, n - 10); i < n; i++)
			{
				sum += rg[i];
				count++;
			}
			double atr = count > 0 ? sum / count : 0;
			if (!(atr > 0))
				return false;

			pendSide = side;
			pendHigh = hi[n];
			pendLow = lo[n];
			pendAtr = atr;
			tradedToday = true;
			exitSent = false;
			SubmitOrderUnmanaged(0, side == 1 ? OrderAction.Buy : OrderAction.SellShort,
				OrderType.Market, Quantity, 0, 0, "", "IdeaEntry");
			return true;
		}

		protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity,
			int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
		{
			if (order.Name == "IdeaEntry")
				entryOrder = order;
			else if (order.Name == "IdeaStop")
				stopOrder = order;
			else if (order.Name == "IdeaTarget")
				targetOrder = order;
		}

		protected override void OnExecutionUpdate(Execution execution, string executionId, double price,
			int quantity, MarketPosition marketPosition, string orderId, DateTime time)
		{
			if (execution.Order == null || execution.Order.Name != "IdeaEntry"
				|| execution.Order.OrderState != OrderState.Filled)
				return;

			double dist;
			if (stopE == IdeaStop.Atr)
				dist = pendAtr;
			else
			{
				dist = pendSide == 1 ? price - (pendLow - TickSize) : (pendHigh + TickSize) - price;
				dist = Math.Max(dist, 0.5 * pendAtr);
			}

			double mult = targetE == IdeaTarget.OneR ? 1.0 : (targetE == IdeaTarget.TwoR ? 2.0 : 0.0);
			OrderAction exitAction = pendSide == 1 ? OrderAction.Sell : OrderAction.BuyToCover;
			double stopPx = Instrument.MasterInstrument.RoundToTickSize(price - pendSide * dist);
			string oco = mult > 0 ? "IdeaOCO" + Guid.NewGuid().ToString("N") : "";

			SubmitOrderUnmanaged(0, exitAction, OrderType.StopMarket, execution.Order.Filled, 0, stopPx, oco, "IdeaStop");
			if (mult > 0)
			{
				double targetPx = Instrument.MasterInstrument.RoundToTickSize(price + pendSide * mult * dist);
				SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, execution.Order.Filled, targetPx, 0, oco, "IdeaTarget");
			}
		}

		private bool IsLive(Order o)
		{
			return o != null && o.OrderState != OrderState.Filled && o.OrderState != OrderState.Cancelled
				&& o.OrderState != OrderState.Rejected && o.OrderState != OrderState.Unknown;
		}

		private void FlattenAll()
		{
			if (IsLive(stopOrder))
				CancelOrder(stopOrder);
			if (IsLive(targetOrder))
				CancelOrder(targetOrder);
			if (Position.MarketPosition == MarketPosition.Long)
				SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "IdeaTimeExit");
			else if (Position.MarketPosition == MarketPosition.Short)
				SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "IdeaTimeExit");
			exitSent = true;
		}

		#region Properties
		[NinjaScriptProperty]
		[Range(0, 10)]
		[Display(Name = "Preset (0 = use the parameters below)", Order = 1, GroupName = "Idea")]
		public int Preset { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Quantity", Order = 2, GroupName = "Idea")]
		public int Quantity { get; set; }

		[NinjaScriptProperty]
		[Range(-12, 12)]
		[Display(Name = "Hours to add to chart time to get New York time", Order = 3, GroupName = "Idea")]
		public int TimeOffsetHours { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Zone", Order = 4, GroupName = "Custom (Preset 0)")]
		public IdeaZone ZoneSel { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Trigger", Order = 5, GroupName = "Custom (Preset 0)")]
		public IdeaTrigger TriggerSel { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Stop", Order = 6, GroupName = "Custom (Preset 0)")]
		public IdeaStop StopSel { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Target", Order = 7, GroupName = "Custom (Preset 0)")]
		public IdeaTarget TargetSel { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Window", Order = 8, GroupName = "Custom (Preset 0)")]
		public IdeaWindow WindowSel { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Trend filter", Order = 9, GroupName = "Custom (Preset 0)")]
		public IdeaTrend TrendSel { get; set; }
		#endregion
	}
}

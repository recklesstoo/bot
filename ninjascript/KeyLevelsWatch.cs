#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

// Key levels as a visual aid with alerts. It does not place orders.
//
// Levels (calculated from the bars the chart contains):
//   PDH PDL PDC   previous trading day high, low, close
//   PWH PWL       previous week high, low
//   PMH PML       previous month high, low
//   ONH ONL       overnight (Globex) high and low, shown from 09:30 New York onwards
//
// Notes
//  - Use a 5-minute chart. With a 24-hour (ETH) session template the "day" is the Globex day
//    (18:00 to 17:00 New York); with a regular-hours template it is 09:30 to 16:00.
//  - Overnight levels need a chart that includes the overnight session.
//  - Monthly levels need at least about 35 days of data loaded on the chart.
//  - TimeOffsetHours converts chart time to New York time (0 if the chart is already in New York time).
//  - The levels are not identical to TraderOS "prev_day_*" levels, which use a UTC day with extended hours.
//  - Alerts fire only in real time. "Approach" fires once when price comes within AlertTicks ticks of a
//    level and re-arms after it moves away to twice that distance. "Cross" fires when price crosses a level.
namespace NinjaTrader.NinjaScript.Indicators
{
	public class KeyLevelsWatch : Indicator
	{
		private static readonly string[] Names = { "PDH", "PDL", "PDC", "PWH", "PWL", "PMH", "PML", "ONH", "ONL" };

		private double[] lv = new double[9];
		private bool[] approachFired = new bool[9];

		private DateTime curTradeDate = DateTime.MinValue;
		private DateTime curWeekStart = DateTime.MinValue;
		private int curMonth = -1;

		private double dayH = double.NaN, dayL = double.NaN, dayC = double.NaN;
		private double weekH = double.NaN, weekL = double.NaN;
		private double monthH = double.NaN, monthL = double.NaN;
		private double pdH = double.NaN, pdL = double.NaN, pdC = double.NaN;
		private double pwH = double.NaN, pwL = double.NaN;
		private double pmH = double.NaN, pmL = double.NaN;
		private double onH = double.NaN, onL = double.NaN;
		private double prevPx = double.NaN;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = "Previous day, week, month and overnight levels with proximity and cross alerts.";
				Name = "KeyLevelsWatch";
				Calculate = Calculate.OnEachTick;
				IsOverlay = true;
				DisplayInDataBox = true;
				PaintPriceMarkers = true;
				DrawOnPricePanel = true;

				ShowDaily = true;
				ShowWeekly = true;
				ShowMonthly = true;
				ShowOvernight = true;
				ShowLabels = true;
				ShowStatus = true;
				AlertOnApproach = true;
				AlertOnCross = true;
				AlertTicks = 10;
				AlertSound = "Alert1.wav";
				TimeOffsetHours = 0;

				AddPlot(new Stroke(Brushes.Red, DashStyleHelper.Solid, 2), PlotStyle.Line, "PDH");
				AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Solid, 2), PlotStyle.Line, "PDL");
				AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Dash, 1), PlotStyle.Line, "PDC");
				AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dash, 2), PlotStyle.Line, "PWH");
				AddPlot(new Stroke(Brushes.SpringGreen, DashStyleHelper.Dash, 2), PlotStyle.Line, "PWL");
				AddPlot(new Stroke(Brushes.Magenta, DashStyleHelper.Dot, 2), PlotStyle.Line, "PMH");
				AddPlot(new Stroke(Brushes.DeepSkyBlue, DashStyleHelper.Dot, 2), PlotStyle.Line, "PML");
				AddPlot(new Stroke(Brushes.Orchid, DashStyleHelper.DashDot, 1), PlotStyle.Line, "ONH");
				AddPlot(new Stroke(Brushes.Turquoise, DashStyleHelper.DashDot, 1), PlotStyle.Line, "ONL");
			}
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBar < 1)
				return;

			DateTime ny = Time[0].AddHours(TimeOffsetHours);
			int hm = ny.Hour * 100 + ny.Minute;
			DateTime tradeDate = ny.Hour >= 18 ? ny.Date.AddDays(1) : ny.Date;

			if (tradeDate != curTradeDate)
			{
				if (curTradeDate != DateTime.MinValue)
				{
					pdH = dayH;
					pdL = dayL;
					pdC = dayC;
				}
				dayH = High[0];
				dayL = Low[0];
				dayC = Close[0];
				onH = double.NaN;
				onL = double.NaN;
				for (int i = 0; i < approachFired.Length; i++)
					approachFired[i] = false;
				curTradeDate = tradeDate;
			}
			else
			{
				dayH = Math.Max(dayH, High[0]);
				dayL = Math.Min(dayL, Low[0]);
				dayC = Close[0];
			}

			DateTime weekStart = tradeDate.AddDays(-(((int)tradeDate.DayOfWeek + 6) % 7));
			if (weekStart != curWeekStart)
			{
				if (curWeekStart != DateTime.MinValue)
				{
					pwH = weekH;
					pwL = weekL;
				}
				weekH = High[0];
				weekL = Low[0];
				curWeekStart = weekStart;
			}
			else
			{
				weekH = Math.Max(weekH, High[0]);
				weekL = Math.Min(weekL, Low[0]);
			}

			int monthKey = tradeDate.Year * 100 + tradeDate.Month;
			if (monthKey != curMonth)
			{
				if (curMonth != -1)
				{
					pmH = monthH;
					pmL = monthL;
				}
				monthH = High[0];
				monthL = Low[0];
				curMonth = monthKey;
			}
			else
			{
				monthH = Math.Max(monthH, High[0]);
				monthL = Math.Min(monthL, Low[0]);
			}

			bool isOvernight = hm > 1800 || hm <= 930;
			if (isOvernight)
			{
				onH = double.IsNaN(onH) ? High[0] : Math.Max(onH, High[0]);
				onL = double.IsNaN(onL) ? Low[0] : Math.Min(onL, Low[0]);
			}

			lv[0] = pdH; lv[1] = pdL; lv[2] = pdC;
			lv[3] = pwH; lv[4] = pwL;
			lv[5] = pmH; lv[6] = pmL;
			lv[7] = isOvernight ? double.NaN : onH;
			lv[8] = isOvernight ? double.NaN : onL;

			for (int i = 0; i < Names.Length; i++)
			{
				if (IsVisible(i) && !double.IsNaN(lv[i]))
				{
					Values[i][0] = lv[i];
					if (ShowLabels)
						Draw.Text(this, "KLW_" + Names[i], Names[i] + " " + lv[i].ToString("F2"), 0, lv[i], Plots[i].Brush);
				}
				else
					Values[i].Reset();
			}

			if (ShowStatus)
				DrawStatus();

			if (State == State.Realtime)
				CheckAlerts();

			prevPx = Close[0];
		}

		private bool IsVisible(int i)
		{
			if (i <= 2)
				return ShowDaily;
			if (i <= 4)
				return ShowWeekly;
			if (i <= 6)
				return ShowMonthly;
			return ShowOvernight;
		}

		private void DrawStatus()
		{
			int best = -1;
			double bestDist = double.MaxValue;
			for (int i = 0; i < Names.Length; i++)
			{
				if (!IsVisible(i) || double.IsNaN(lv[i]))
					continue;
				double d = Math.Abs(Close[0] - lv[i]);
				if (d < bestDist)
				{
					bestDist = d;
					best = i;
				}
			}
			if (best < 0)
				return;
			string side = Close[0] >= lv[best] ? "above" : "below";
			Draw.TextFixed(this, "KLW_status",
				"Nearest level: " + Names[best] + " " + lv[best].ToString("F2") + " (" + (bestDist / TickSize).ToString("F0") + " ticks, price " + side + ")",
				TextPosition.TopLeft);
		}

		private void CheckAlerts()
		{
			double px = Close[0];
			for (int i = 0; i < Names.Length; i++)
			{
				if (!IsVisible(i) || double.IsNaN(lv[i]))
					continue;

				double ticks = Math.Abs(px - lv[i]) / TickSize;
				if (AlertOnApproach)
				{
					if (ticks <= AlertTicks)
					{
						if (!approachFired[i])
						{
							approachFired[i] = true;
							Notify("KLW_A_" + Names[i], Names[i] + " " + lv[i].ToString("F2") + ": price within " + ticks.ToString("F0") + " ticks");
						}
					}
					else if (ticks > 2 * AlertTicks)
						approachFired[i] = false;
				}

				if (AlertOnCross && !double.IsNaN(prevPx) && (prevPx - lv[i]) * (px - lv[i]) < 0)
					Notify("KLW_C_" + Names[i], Names[i] + " " + lv[i].ToString("F2") + ": price crossed " + (px > prevPx ? "up" : "down"));
			}
		}

		private void Notify(string id, string message)
		{
			string sound = string.IsNullOrEmpty(AlertSound) ? "" : NinjaTrader.Core.Globals.InstallDir + @"\sounds\" + AlertSound;
			Alert(id, Priority.High, message, sound, 10, Brushes.Black, Brushes.Yellow);
			Print(Time[0].ToString("HH:mm:ss") + " " + message);
		}

		#region Properties
		[NinjaScriptProperty]
		[Display(Name = "Show previous day levels", Order = 1, GroupName = "Levels")]
		public bool ShowDaily { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show previous week levels", Order = 2, GroupName = "Levels")]
		public bool ShowWeekly { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show previous month levels", Order = 3, GroupName = "Levels")]
		public bool ShowMonthly { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show overnight levels", Order = 4, GroupName = "Levels")]
		public bool ShowOvernight { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show labels", Order = 5, GroupName = "Levels")]
		public bool ShowLabels { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show nearest level (top left)", Order = 6, GroupName = "Levels")]
		public bool ShowStatus { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Alert when price approaches a level", Order = 1, GroupName = "Alerts")]
		public bool AlertOnApproach { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Alert when price crosses a level", Order = 2, GroupName = "Alerts")]
		public bool AlertOnCross { get; set; }

		[NinjaScriptProperty]
		[Range(1, 1000)]
		[Display(Name = "Approach distance (ticks)", Order = 3, GroupName = "Alerts")]
		public int AlertTicks { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Alert sound file (empty = none)", Order = 4, GroupName = "Alerts")]
		public string AlertSound { get; set; }

		[NinjaScriptProperty]
		[Range(-12, 12)]
		[Display(Name = "Hours to add to chart time to get New York time", Order = 1, GroupName = "Time")]
		public int TimeOffsetHours { get; set; }
		#endregion
	}
}

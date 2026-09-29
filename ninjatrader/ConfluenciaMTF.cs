#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

// ConfluenciaMTF - Indicador de confluencia multi-temporalidad para NinjaTrader 8
//
// Combina RSI + MACD + Volumen + ATR en 1, 5 y 15 minutos y muestra el ADR.
//
//  - Cada temporalidad tiene un SESGO:
//      ALCISTA  = histograma MACD > 0  y  RSI > 50
//      BAJISTA  = histograma MACD < 0  y  RSI < 50
//      NEUTRAL  = cualquier otro caso
//  - COMPRA: las 3 temporalidades ALCISTAS + volumen 1m > media * multiplicador
//            + RSI 1m por debajo de sobrecompra + ATR 1m >= minimo + ADR no agotado
//  - VENTA:  lo mismo en sentido bajista.
//  - En cada senal dibuja flecha, texto y lineas de Stop / Objetivo basadas en ATR.
//  - Panel en la esquina con el estado de cada temporalidad, ATR y ADR.
//  - El fondo se colorea verde/rojo cuando las 3 temporalidades estan alineadas.
//
// Recomendado: grafico de 1 minuto, Calculate = On bar close, cargar >= 20 dias.

namespace NinjaTrader.NinjaScript.Indicators
{
	public class ConfluenciaMTF : Indicator
	{
		// Indices de las series de datos (0 = serie del grafico)
		private const int Idx1m  = 1;
		private const int Idx5m  = 2;
		private const int Idx15m = 3;
		private const int IdxDay = 4;

		private RSI  rsi1, rsi5, rsi15;
		private MACD macd1, macd5, macd15;
		private SMA  volSma1, volSma5, volSma15;
		private ATR  atr1, atr5, atr15;
		private CurrentDayOHL dayOhl;

		private Series<double> signalSeries;
		private int lastSignal; // 1 = ultima senal fue compra, -1 = venta, 0 = rearmado

		private Brush bgBull, bgBear;
		private SimpleFont panelFont;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description					= "Confluencia RSI + MACD + Volumen + ATR en 1/5/15 min con senales de COMPRA/VENTA y ADR.";
				Name						= "ConfluenciaMTF";
				Calculate					= Calculate.OnBarClose;
				IsOverlay					= true;
				IsAutoScale					= false;
				DisplayInDataBox			= false;
				DrawOnPricePanel			= true;
				PaintPriceMarkers			= false;
				IsSuspendedWhileInactive	= true;

				RsiPeriod			= 14;
				RsiSmooth			= 3;
				RsiOverbought		= 70;
				RsiOversold			= 30;
				MacdFast			= 12;
				MacdSlow			= 26;
				MacdSmooth			= 9;
				VolumePeriod		= 20;
				VolumeMultiplier	= 1.2;
				AtrPeriod			= 14;
				MinAtrTicks			= 0;
				StopAtrMultiplier	= 1.5;
				RewardRisk			= 2.0;
				AdrPeriod			= 14;
				UseAdrFilter		= true;
				AdrMaxPercent		= 100;

				ShowPanel			= true;
				ShowStopTarget		= true;
				ColorBackground		= true;
				LineLength			= 15;
				ArrowOffsetTicks	= 4;
				PanelFontSize		= 12;
				UseAlerts			= true;
				BuyBrush			= Brushes.LimeGreen;
				SellBrush			= Brushes.Red;
				PanelTextBrush		= Brushes.White;
				PanelBackBrush		= Brushes.Black;
			}
			else if (State == State.Configure)
			{
				AddDataSeries(BarsPeriodType.Minute, 1);	// Idx1m
				AddDataSeries(BarsPeriodType.Minute, 5);	// Idx5m
				AddDataSeries(BarsPeriodType.Minute, 15);	// Idx15m
				AddDataSeries(BarsPeriodType.Day, 1);		// IdxDay (para ADR)
			}
			else if (State == State.DataLoaded)
			{
				rsi1	= RSI(BarsArray[Idx1m],  RsiPeriod, RsiSmooth);
				rsi5	= RSI(BarsArray[Idx5m],  RsiPeriod, RsiSmooth);
				rsi15	= RSI(BarsArray[Idx15m], RsiPeriod, RsiSmooth);

				macd1	= MACD(BarsArray[Idx1m],  MacdFast, MacdSlow, MacdSmooth);
				macd5	= MACD(BarsArray[Idx5m],  MacdFast, MacdSlow, MacdSmooth);
				macd15	= MACD(BarsArray[Idx15m], MacdFast, MacdSlow, MacdSmooth);

				volSma1		= SMA(Volumes[Idx1m],  VolumePeriod);
				volSma5		= SMA(Volumes[Idx5m],  VolumePeriod);
				volSma15	= SMA(Volumes[Idx15m], VolumePeriod);

				atr1	= ATR(BarsArray[Idx1m],  AtrPeriod);
				atr5	= ATR(BarsArray[Idx5m],  AtrPeriod);
				atr15	= ATR(BarsArray[Idx15m], AtrPeriod);

				dayOhl	= CurrentDayOHL();

				signalSeries = new Series<double>(this);

				bgBull = new SolidColorBrush(Color.FromArgb(28, 0, 200, 0));
				bgBull.Freeze();
				bgBear = new SolidColorBrush(Color.FromArgb(28, 220, 0, 0));
				bgBear.Freeze();

				panelFont = new SimpleFont("Consolas", PanelFontSize);
				lastSignal = 0;
			}
		}

		protected override void OnBarUpdate()
		{
			// Toda la logica se ejecuta sobre la serie del grafico
			if (BarsInProgress != 0)
				return;

			int minBars = Math.Max(MacdSlow + MacdSmooth, Math.Max(VolumePeriod, Math.Max(RsiPeriod, AtrPeriod))) + 1;
			if (CurrentBars[0] < 1
				|| CurrentBars[Idx1m]  < minBars
				|| CurrentBars[Idx5m]  < minBars
				|| CurrentBars[Idx15m] < minBars)
			{
				signalSeries[0] = 0;
				return;
			}

			// ---- Sesgo por temporalidad ----
			int b1  = Bias(rsi1,  macd1);
			int b5  = Bias(rsi5,  macd5);
			int b15 = Bias(rsi15, macd15);
			int score = b1 + b5 + b15;

			double vr1  = VolumeRatio(Volumes[Idx1m][0],  volSma1[0]);
			double vr5  = VolumeRatio(Volumes[Idx5m][0],  volSma5[0]);
			double vr15 = VolumeRatio(Volumes[Idx15m][0], volSma15[0]);

			double atrVal = atr1[0];

			// ---- ADR ----
			double adr = ComputeAdr();
			double todayRange = dayOhl.CurrentHigh[0] - dayOhl.CurrentLow[0];
			double adrUsedPct = adr > 0 ? todayRange / adr * 100.0 : 0;

			// ---- Condiciones ----
			bool allBull = b1 == 1 && b5 == 1 && b15 == 1;
			bool allBear = b1 == -1 && b5 == -1 && b15 == -1;
			bool volOk   = vr1 >= VolumeMultiplier;
			bool atrOk   = MinAtrTicks <= 0 || atrVal >= MinAtrTicks * TickSize;
			bool adrOk   = !UseAdrFilter || adr <= 0 || adrUsedPct < AdrMaxPercent;

			bool buyCond  = allBull && volOk && atrOk && adrOk && rsi1[0] < RsiOverbought;
			bool sellCond = allBear && volOk && atrOk && adrOk && rsi1[0] > RsiOversold;

			// Rearmar cuando se pierde la alineacion para permitir nuevas entradas
			if ((lastSignal == 1 && !allBull) || (lastSignal == -1 && !allBear))
				lastSignal = 0;

			signalSeries[0] = 0;

			if (buyCond && lastSignal != 1)
			{
				lastSignal = 1;
				signalSeries[0] = 1;
				DrawSignal(true, atrVal);
			}
			else if (sellCond && lastSignal != -1)
			{
				lastSignal = -1;
				signalSeries[0] = -1;
				DrawSignal(false, atrVal);
			}

			// ---- Fondo ----
			if (ColorBackground)
			{
				if (allBull)
					BackBrush = bgBull;
				else if (allBear)
					BackBrush = bgBear;
			}

			// ---- Panel (solo en la ultima barra para no cargar el historico) ----
			if (ShowPanel && (State == State.Realtime || CurrentBar >= Count - 2))
				DrawPanel(b1, b5, b15, score, vr1, vr5, vr15, adr, todayRange, adrUsedPct, allBull, allBear, volOk, adrOk);
		}

		#region Helpers

		private int Bias(RSI r, MACD m)
		{
			if (m.Diff[0] > 0 && r[0] > 50)
				return 1;
			if (m.Diff[0] < 0 && r[0] < 50)
				return -1;
			return 0;
		}

		private double VolumeRatio(double vol, double avg)
		{
			return avg > 0 ? vol / avg : 0;
		}

		private double ComputeAdr()
		{
			if (CurrentBars[IdxDay] < AdrPeriod - 1)
				return 0;

			double sum = 0;
			for (int i = 0; i < AdrPeriod; i++)
				sum += Highs[IdxDay][i] - Lows[IdxDay][i];
			return sum / AdrPeriod;
		}

		private void DrawSignal(bool isBuy, double atrVal)
		{
			double offset	= ArrowOffsetTicks * TickSize;
			double entry	= Close[0];
			double risk		= atrVal * StopAtrMultiplier;
			string tag		= (isBuy ? "CMTF_Buy_" : "CMTF_Sell_") + CurrentBar;

			if (isBuy)
			{
				Draw.ArrowUp(this, tag + "_A", false, 0, Low[0] - offset, BuyBrush);
				Draw.Text(this, tag + "_T", "COMPRA", 0, Low[0] - offset * 3, BuyBrush);
			}
			else
			{
				Draw.ArrowDown(this, tag + "_A", false, 0, High[0] + offset, SellBrush);
				Draw.Text(this, tag + "_T", "VENTA", 0, High[0] + offset * 3, SellBrush);
			}

			if (ShowStopTarget && risk > 0)
			{
				double stop   = isBuy ? entry - risk : entry + risk;
				double target = isBuy ? entry + risk * RewardRisk : entry - risk * RewardRisk;

				Draw.Line(this, tag + "_E",  false, 0, entry,  -LineLength, entry,  Brushes.Gray,       DashStyleHelper.Dot,  1);
				Draw.Line(this, tag + "_SL", false, 0, stop,   -LineLength, stop,   Brushes.OrangeRed,  DashStyleHelper.Dash, 2);
				Draw.Line(this, tag + "_TP", false, 0, target, -LineLength, target, Brushes.DodgerBlue, DashStyleHelper.Dash, 2);
			}

			if (UseAlerts && State == State.Realtime)
			{
				string msg = string.Format("{0} {1} @ {2}", isBuy ? "COMPRA" : "VENTA",
					Instrument.FullName, Instrument.MasterInstrument.FormatPrice(entry));
				Alert("CMTF_" + CurrentBar, Priority.High, msg,
					NinjaTrader.Core.Globals.InstallDir + @"\sounds\Alert1.wav", 10,
					isBuy ? BuyBrush : SellBrush, Brushes.Black);
			}
		}

		private string BiasText(int b)
		{
			return b == 1 ? "ALCISTA" : (b == -1 ? "BAJISTA" : "neutral");
		}

		private string PanelRow(string tf, RSI r, MACD m, double vr, int bias)
		{
			return string.Format("{0,-4}{1,6:0.0}  {2,-6}{3,5:0.0}x  {4}",
				tf, r[0], m.Diff[0] >= 0 ? "+ alc" : "- baj", vr, BiasText(bias));
		}

		private string FmtPrice(double v)
		{
			return Instrument.MasterInstrument.FormatPrice(v);
		}

		private void DrawPanel(int b1, int b5, int b15, int score, double vr1, double vr5, double vr15,
			double adr, double todayRange, double adrUsedPct, bool allBull, bool allBear, bool volOk, bool adrOk)
		{
			StringBuilder sb = new StringBuilder();
			sb.AppendLine("CONFLUENCIA MTF");
			sb.AppendLine(string.Format("{0,-4}{1,6}  {2,-6}{3,6}  {4}", "TF", "RSI", "MACD", "VOL", "SESGO"));
			sb.AppendLine(PanelRow("1m",  rsi1,  macd1,  vr1,  b1));
			sb.AppendLine(PanelRow("5m",  rsi5,  macd5,  vr5,  b5));
			sb.AppendLine(PanelRow("15m", rsi15, macd15, vr15, b15));
			sb.AppendLine("--------------------------------");
			sb.AppendLine(string.Format("ATR({0})  1m {1} ({2:0} ticks)", AtrPeriod, FmtPrice(atr1[0]), atr1[0] / TickSize));
			sb.AppendLine(string.Format("         5m {0}  15m {1}", FmtPrice(atr5[0]), FmtPrice(atr15[0])));

			if (adr > 0)
			{
				sb.AppendLine(string.Format("ADR({0}d) {1} ({2:0} ticks)", AdrPeriod, FmtPrice(adr), adr / TickSize));
				sb.AppendLine(string.Format("Hoy      {0}  = {1:0}% del ADR", FmtPrice(todayRange), adrUsedPct));
				sb.AppendLine(string.Format("Restante {0}", FmtPrice(Math.Max(0, adr - todayRange))));
			}
			else
			{
				sb.AppendLine(string.Format("ADR({0}d) cargando... (carga mas dias)", AdrPeriod));
			}

			sb.AppendLine("--------------------------------");
			sb.AppendLine(string.Format("Consenso: {0}{1}/3", score > 0 ? "+" : "", score));

			string estado;
			if (allBull && volOk && adrOk)
				estado = ">>> COMPRA <<<";
			else if (allBear && volOk && adrOk)
				estado = ">>> VENTA <<<";
			else if (allBull)
				estado = "ALCISTA - esperar volumen";
			else if (allBear)
				estado = "BAJISTA - esperar volumen";
			else
				estado = "ESPERAR (sin alineacion)";

			if (!adrOk && (allBull || allBear))
				estado = "RANGO DIARIO AGOTADO";

			sb.Append("Estado:   " + estado);

			Draw.TextFixed(this, "CMTF_Panel", sb.ToString(), TextPosition.TopRight,
				PanelTextBrush, panelFont, Brushes.Gray, PanelBackBrush, 70);
		}

		#endregion

		#region Properties

		[Browsable(false)]
		[XmlIgnore]
		public Series<double> Senal
		{
			get { Update(); return signalSeries; }
		}

		[Range(2, int.MaxValue)]
		[Display(Name = "RSI periodo", Order = 1, GroupName = "1. RSI")]
		public int RsiPeriod { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "RSI suavizado", Order = 2, GroupName = "1. RSI")]
		public int RsiSmooth { get; set; }

		[Range(50, 100)]
		[Display(Name = "RSI sobrecompra", Order = 3, GroupName = "1. RSI")]
		public double RsiOverbought { get; set; }

		[Range(0, 50)]
		[Display(Name = "RSI sobreventa", Order = 4, GroupName = "1. RSI")]
		public double RsiOversold { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "MACD rapida", Order = 1, GroupName = "2. MACD")]
		public int MacdFast { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "MACD lenta", Order = 2, GroupName = "2. MACD")]
		public int MacdSlow { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "MACD senal", Order = 3, GroupName = "2. MACD")]
		public int MacdSmooth { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "Volumen periodo media", Order = 1, GroupName = "3. Volumen")]
		public int VolumePeriod { get; set; }

		[Range(0, double.MaxValue)]
		[Display(Name = "Volumen multiplicador", Description = "Volumen de 1m debe ser >= media * este valor", Order = 2, GroupName = "3. Volumen")]
		public double VolumeMultiplier { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "ATR periodo", Order = 1, GroupName = "4. ATR / Stop")]
		public int AtrPeriod { get; set; }

		[Range(0, int.MaxValue)]
		[Display(Name = "ATR minimo (ticks)", Description = "0 = desactivado. Evita senales en mercado muerto.", Order = 2, GroupName = "4. ATR / Stop")]
		public int MinAtrTicks { get; set; }

		[Range(0.1, double.MaxValue)]
		[Display(Name = "Stop = ATR x", Order = 3, GroupName = "4. ATR / Stop")]
		public double StopAtrMultiplier { get; set; }

		[Range(0.1, double.MaxValue)]
		[Display(Name = "Objetivo (R:R)", Order = 4, GroupName = "4. ATR / Stop")]
		public double RewardRisk { get; set; }

		[Range(1, int.MaxValue)]
		[Display(Name = "ADR dias", Order = 1, GroupName = "5. ADR")]
		public int AdrPeriod { get; set; }

		[Display(Name = "Filtrar si ADR agotado", Order = 2, GroupName = "5. ADR")]
		public bool UseAdrFilter { get; set; }

		[Range(1, 1000)]
		[Display(Name = "ADR maximo usado (%)", Order = 3, GroupName = "5. ADR")]
		public double AdrMaxPercent { get; set; }

		[Display(Name = "Mostrar panel", Order = 1, GroupName = "6. Visual")]
		public bool ShowPanel { get; set; }

		[Display(Name = "Mostrar Stop / Objetivo", Order = 2, GroupName = "6. Visual")]
		public bool ShowStopTarget { get; set; }

		[Display(Name = "Colorear fondo", Order = 3, GroupName = "6. Visual")]
		public bool ColorBackground { get; set; }

		[Range(1, 500)]
		[Display(Name = "Largo lineas (barras)", Order = 4, GroupName = "6. Visual")]
		public int LineLength { get; set; }

		[Range(0, 100)]
		[Display(Name = "Separacion flecha (ticks)", Order = 5, GroupName = "6. Visual")]
		public int ArrowOffsetTicks { get; set; }

		[Range(6, 30)]
		[Display(Name = "Tamano letra panel", Order = 6, GroupName = "6. Visual")]
		public int PanelFontSize { get; set; }

		[Display(Name = "Alertas con sonido", Order = 7, GroupName = "6. Visual")]
		public bool UseAlerts { get; set; }

		[XmlIgnore]
		[Display(Name = "Color compra", Order = 8, GroupName = "6. Visual")]
		public Brush BuyBrush { get; set; }

		[Browsable(false)]
		public string BuyBrushSerializable
		{
			get { return Serialize.BrushToString(BuyBrush); }
			set { BuyBrush = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Color venta", Order = 9, GroupName = "6. Visual")]
		public Brush SellBrush { get; set; }

		[Browsable(false)]
		public string SellBrushSerializable
		{
			get { return Serialize.BrushToString(SellBrush); }
			set { SellBrush = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Color texto panel", Order = 10, GroupName = "6. Visual")]
		public Brush PanelTextBrush { get; set; }

		[Browsable(false)]
		public string PanelTextBrushSerializable
		{
			get { return Serialize.BrushToString(PanelTextBrush); }
			set { PanelTextBrush = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Color fondo panel", Order = 11, GroupName = "6. Visual")]
		public Brush PanelBackBrush { get; set; }

		[Browsable(false)]
		public string PanelBackBrushSerializable
		{
			get { return Serialize.BrushToString(PanelBackBrush); }
			set { PanelBackBrush = Serialize.StringToBrush(value); }
		}

		#endregion
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private ConfluenciaMTF[] cacheConfluenciaMTF;
		public ConfluenciaMTF ConfluenciaMTF()
		{
			return ConfluenciaMTF(Input);
		}

		public ConfluenciaMTF ConfluenciaMTF(ISeries<double> input)
		{
			if (cacheConfluenciaMTF != null)
				for (int idx = 0; idx < cacheConfluenciaMTF.Length; idx++)
					if (cacheConfluenciaMTF[idx] != null &&  cacheConfluenciaMTF[idx].EqualsInput(input))
						return cacheConfluenciaMTF[idx];
			return CacheIndicator<ConfluenciaMTF>(new ConfluenciaMTF(), input, ref cacheConfluenciaMTF);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.ConfluenciaMTF ConfluenciaMTF()
		{
			return indicator.ConfluenciaMTF(Input);
		}

		public Indicators.ConfluenciaMTF ConfluenciaMTF(ISeries<double> input )
		{
			return indicator.ConfluenciaMTF(input);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.ConfluenciaMTF ConfluenciaMTF()
		{
			return indicator.ConfluenciaMTF(Input);
		}

		public Indicators.ConfluenciaMTF ConfluenciaMTF(ISeries<double> input )
		{
			return indicator.ConfluenciaMTF(input);
		}
	}
}

#endregion

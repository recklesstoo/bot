#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

// Estrategia NinjaTrader 8 que consulta a una IA local (ai_server + Ollama)
// al cierre de cada vela y ejecuta BUY / SELL / EXIT / HOLD.
// La gestión del riesgo (stop, target, pérdida diaria, horario, nº de trades)
// se hace AQUÍ, no en la IA: la IA solo propone la dirección.
namespace NinjaTrader.NinjaScript.Strategies
{
	public class AIFuturesTrader : Strategy
	{
		private const string LongSignal  = "AI Long";
		private const string ShortSignal = "AI Short";

		private ATR    atr;
		private volatile bool requestInFlight;
		private double sessionStartCumProfit;
		private int    tradesToday;
		private bool   haltedToday;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Name                         = "AIFuturesTrader";
				Description                  = "Opera futuros con decisiones de un LLM local (Ollama) vía ai_server.";
				Calculate                    = Calculate.OnBarClose;
				EntriesPerDirection          = 1;
				EntryHandling                = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds    = 120;
				BarsRequiredToTrade          = 60;
				StartBehavior                = StartBehavior.WaitUntilFlat;
				RealtimeErrorHandling        = RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling           = StopTargetHandling.PerEntryExecution;
				TimeInForce                  = TimeInForce.Gtc;
				TraceOrders                  = false;

				ServerUrl         = "http://127.0.0.1:8000/decide";
				ApiToken          = "";
				RequestTimeoutSec = 20;
				BarsToSend        = 100;

				Quantity          = 1;
				AllowShorts       = true;
				AllowReversal     = false;
				MinConfidence     = 0.65;

				AtrPeriod         = 14;
				StopAtrMult       = 1.5;
				TargetAtrMult     = 3.0;
				MinStopTicks      = 8;
				MaxStopTicks      = 200;

				MaxDailyLoss      = 300;
				MaxTradesPerDay   = 6;
				StartTime         = 93500;
				EndTime           = 154500;
				FlattenOutsideHours = true;
			}
			else if (State == State.DataLoaded)
			{
				atr = ATR(AtrPeriod);
			}
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBar < Math.Max(BarsRequiredToTrade, BarsToSend))
				return;

			if (Bars.IsFirstBarOfSession)
			{
				sessionStartCumProfit = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
				tradesToday = 0;
				haltedToday = false;
			}

			// Nunca se consulta a la IA sobre datos históricos: solo en tiempo real.
			if (State != State.Realtime)
				return;

			double dailyPnl = DailyPnl();
			if (!haltedToday && dailyPnl <= -MaxDailyLoss)
			{
				haltedToday = true;
				Print(string.Format("{0} [AI] Pérdida diaria máxima alcanzada ({1:C}). Cerrando y deteniendo hasta la próxima sesión.", Time[0], dailyPnl));
				Flatten("AI DailyLoss");
			}
			if (haltedToday)
				return;

			int now = ToTime(Time[0]);
			if (now < StartTime || now > EndTime)
			{
				if (FlattenOutsideHours && Position.MarketPosition != MarketPosition.Flat)
				{
					Print(string.Format("{0} [AI] Fuera de horario, cerrando posición.", Time[0]));
					Flatten("AI Hours");
				}
				return;
			}

			if (requestInFlight)
				return;

			string payload     = BuildPayload(dailyPnl);
			string url         = ServerUrl;
			string token       = ApiToken;
			int    timeoutMs   = RequestTimeoutSec * 1000;
			int    barAtRequest = CurrentBar;
			requestInFlight = true;

			Task.Run(() =>
			{
				string response = null, error = null;
				try   { response = Post(url, token, payload, timeoutMs); }
				catch (Exception ex) { error = ex.Message; }

				try   { TriggerCustomEvent(o => HandleResponse(response, error, barAtRequest), null); }
				catch (Exception ex) { requestInFlight = false; Print("[AI] Error despachando respuesta: " + ex.Message); }
			});
		}

		private void HandleResponse(string response, string error, int barAtRequest)
		{
			requestInFlight = false;
			if (State != State.Realtime || haltedToday)
				return;

			if (error != null)
			{
				Print(string.Format("{0} [AI] Sin respuesta del servidor ({1}). No se opera.", Time[0], error));
				return;
			}
			if (CurrentBar != barAtRequest)
			{
				Print(string.Format("{0} [AI] Respuesta llegó tarde (vela ya cerrada). Se descarta.", Time[0]));
				return;
			}

			string action     = JsonString(response, "action").ToUpperInvariant();
			double confidence = JsonNumber(response, "confidence");
			string reason     = JsonString(response, "reason");

			Print(string.Format("{0} [AI] {1} conf={2:0.00} pos={3} | {4}", Time[0], action, confidence, Position.MarketPosition, reason));

			if (action != "EXIT" && confidence < MinConfidence)
				return;

			Execute(action);
		}

		private void Execute(string action)
		{
			MarketPosition mp = Position.MarketPosition;

			switch (action)
			{
				case "BUY":
					if (mp == MarketPosition.Long) return;
					if (mp == MarketPosition.Short && !AllowReversal) { ExitShort(Position.Quantity, "AI Exit", ShortSignal); return; }
					if (!CanOpen()) return;
					SetBracket(LongSignal);
					EnterLong(Quantity, LongSignal);
					tradesToday++;
					break;

				case "SELL":
					if (mp == MarketPosition.Short) return;
					if (mp == MarketPosition.Long && (!AllowReversal || !AllowShorts)) { ExitLong(Position.Quantity, "AI Exit", LongSignal); return; }
					if (!AllowShorts || !CanOpen()) return;
					SetBracket(ShortSignal);
					EnterShort(Quantity, ShortSignal);
					tradesToday++;
					break;

				case "EXIT":
					Flatten("AI Exit");
					break;
			}
		}

		private bool CanOpen()
		{
			if (tradesToday >= MaxTradesPerDay)
			{
				Print(string.Format("{0} [AI] Máximo de trades diarios ({1}) alcanzado.", Time[0], MaxTradesPerDay));
				return false;
			}
			return true;
		}

		// Stop y target en ticks a partir del ATR. Se fijan ANTES de la entrada,
		// así la orden de protección sale en cuanto se llena la entrada.
		private void SetBracket(string signal)
		{
			double atrTicks = atr[0] / TickSize;
			int stopTicks   = Math.Min(MaxStopTicks, Math.Max(MinStopTicks, (int)Math.Round(atrTicks * StopAtrMult)));
			int targetTicks = Math.Max(stopTicks, (int)Math.Round(atrTicks * TargetAtrMult));
			SetStopLoss(signal, CalculationMode.Ticks, stopTicks, false);
			SetProfitTarget(signal, CalculationMode.Ticks, targetTicks);
			Print(string.Format("{0} [AI] {1}: stop={2} ticks, target={3} ticks", Time[0], signal, stopTicks, targetTicks));
		}

		private void Flatten(string name)
		{
			if (Position.MarketPosition == MarketPosition.Long)
				ExitLong(Position.Quantity, name, LongSignal);
			else if (Position.MarketPosition == MarketPosition.Short)
				ExitShort(Position.Quantity, name, ShortSignal);
		}

		private double DailyPnl()
		{
			double realized   = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit - sessionStartCumProfit;
			double unrealized = Position.MarketPosition == MarketPosition.Flat ? 0 : Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);
			return realized + unrealized;
		}

		private string BuildPayload(double dailyPnl)
		{
			CultureInfo ic = CultureInfo.InvariantCulture;
			string pos = Position.MarketPosition == MarketPosition.Long ? "LONG"
			           : Position.MarketPosition == MarketPosition.Short ? "SHORT" : "FLAT";
			double unrealized = Position.MarketPosition == MarketPosition.Flat ? 0 : Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);

			var sb = new StringBuilder(BarsToSend * 110 + 400);
			sb.Append('{');
			sb.AppendFormat(ic, "\"instrument\":\"{0}\",", Escape(Instrument.FullName));
			sb.AppendFormat(ic, "\"timeframe\":\"{0}\",", Escape(BarsPeriod.ToString()));
			sb.AppendFormat(ic, "\"tick_size\":{0},", TickSize);
			sb.AppendFormat(ic, "\"point_value\":{0},", Instrument.MasterInstrument.PointValue);
			sb.AppendFormat(ic, "\"position\":\"{0}\",", pos);
			sb.AppendFormat(ic, "\"position_qty\":{0},", Position.Quantity);
			sb.AppendFormat(ic, "\"avg_price\":{0},", Position.AveragePrice);
			sb.AppendFormat(ic, "\"unrealized_pnl\":{0:0.##},", unrealized);
			sb.AppendFormat(ic, "\"daily_pnl\":{0:0.##},", dailyPnl);
			sb.AppendFormat(ic, "\"trades_today\":{0},", tradesToday);
			sb.AppendFormat(ic, "\"allow_shorts\":{0},", AllowShorts ? "true" : "false");
			sb.Append("\"bars\":[");
			for (int i = BarsToSend - 1; i >= 0; i--)
			{
				sb.AppendFormat(ic, "{{\"t\":\"{0:yyyy-MM-ddTHH:mm:ss}\",\"o\":{1},\"h\":{2},\"l\":{3},\"c\":{4},\"v\":{5}}}",
					Time[i], Open[i], High[i], Low[i], Close[i], Volume[i]);
				if (i > 0) sb.Append(',');
			}
			sb.Append("]}");
			return sb.ToString();
		}

		private static string Post(string url, string token, string json, int timeoutMs)
		{
			var req = (HttpWebRequest)WebRequest.Create(url);
			req.Method           = "POST";
			req.ContentType      = "application/json";
			req.Timeout          = timeoutMs;
			req.ReadWriteTimeout = timeoutMs;
			if (!string.IsNullOrEmpty(token))
				req.Headers.Add("X-Api-Token", token);

			byte[] data = Encoding.UTF8.GetBytes(json);
			req.ContentLength = data.Length;
			using (Stream s = req.GetRequestStream())
				s.Write(data, 0, data.Length);

			using (var resp = (HttpWebResponse)req.GetResponse())
			using (var reader = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
				return reader.ReadToEnd();
		}

		private static string Escape(string s)
		{
			return (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
		}

		private static string JsonString(string json, string key)
		{
			Match m = Regex.Match(json ?? "", "\"" + key + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
			return m.Success ? Regex.Unescape(m.Groups[1].Value) : "";
		}

		private static double JsonNumber(string json, string key)
		{
			Match m = Regex.Match(json ?? "", "\"" + key + "\"\\s*:\\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)");
			double v;
			return m.Success && double.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out v) ? v : 0;
		}

		#region Properties
		[NinjaScriptProperty]
		[Display(Name = "Server URL", Order = 1, GroupName = "1. IA")]
		public string ServerUrl { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Api token", Order = 2, GroupName = "1. IA")]
		public string ApiToken { get; set; }

		[NinjaScriptProperty]
		[Range(2, 120)]
		[Display(Name = "Timeout (seg)", Order = 3, GroupName = "1. IA")]
		public int RequestTimeoutSec { get; set; }

		[NinjaScriptProperty]
		[Range(60, 500)]
		[Display(Name = "Velas a enviar", Order = 4, GroupName = "1. IA")]
		public int BarsToSend { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 1.0)]
		[Display(Name = "Confianza mínima", Order = 5, GroupName = "1. IA")]
		public double MinConfidence { get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "Contratos", Order = 1, GroupName = "2. Órdenes")]
		public int Quantity { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Permitir cortos", Order = 2, GroupName = "2. Órdenes")]
		public bool AllowShorts { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Permitir reversión directa", Order = 3, GroupName = "2. Órdenes")]
		public bool AllowReversal { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Name = "Periodo ATR", Order = 1, GroupName = "3. Riesgo")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(0.1, 20)]
		[Display(Name = "Stop (x ATR)", Order = 2, GroupName = "3. Riesgo")]
		public double StopAtrMult { get; set; }

		[NinjaScriptProperty]
		[Range(0.1, 50)]
		[Display(Name = "Target (x ATR)", Order = 3, GroupName = "3. Riesgo")]
		public double TargetAtrMult { get; set; }

		[NinjaScriptProperty]
		[Range(1, 1000)]
		[Display(Name = "Stop mínimo (ticks)", Order = 4, GroupName = "3. Riesgo")]
		public int MinStopTicks { get; set; }

		[NinjaScriptProperty]
		[Range(1, 5000)]
		[Display(Name = "Stop máximo (ticks)", Order = 5, GroupName = "3. Riesgo")]
		public int MaxStopTicks { get; set; }

		[NinjaScriptProperty]
		[Range(1, double.MaxValue)]
		[Display(Name = "Pérdida diaria máx ($)", Order = 6, GroupName = "3. Riesgo")]
		public double MaxDailyLoss { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Máx trades por día", Order = 7, GroupName = "3. Riesgo")]
		public int MaxTradesPerDay { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Hora inicio (HHmmss)", Order = 1, GroupName = "4. Horario")]
		public int StartTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Hora fin (HHmmss)", Order = 2, GroupName = "4. Horario")]
		public int EndTime { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Cerrar fuera de horario", Order = 3, GroupName = "4. Horario")]
		public bool FlattenOutsideHours { get; set; }
		#endregion
	}
}

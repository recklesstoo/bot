#region Using declarations
using System;
using System.Collections.Generic;
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
		private TimeZoneInfo chartTz;
		private TimeZoneInfo newYorkTz;
		private volatile bool requestInFlight;
		private double sessionStartCumProfit;
		private int    tradesToday;
		private bool   haltedToday;

		// Panel: latido cada segundo hacia ai_server (estado + comandos manuales)
		private System.Threading.Timer heartbeatTimer;
		private volatile bool heartbeatInFlight;
		private readonly object panelLock = new object();
		private readonly List<string> pendingEvents = new List<string>();
		private readonly List<string> pendingAcks   = new List<string>();

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
				BarsToSend        = 1500;   // ~5 días de velas de 5 min: día anterior, noche y volumen relativo

				Quantity          = 1;
				AllowShorts       = true;
				AllowReversal     = false;
				MinConfidence     = 0.65;

				AtrPeriod         = 14;
				StopAtrMult       = 1.5;
				TargetAtrMult     = 3.0;
				MinStopTicks      = 40;   // MNQ: 10 puntos = 20 $ por contrato
				MaxStopTicks      = 160;  // MNQ: 40 puntos = 80 $ por contrato

				MaxDailyLoss      = 200;
				MaxTradesPerDay   = 4;
				StartTime         = 93500;
				EndTime           = 154500;
				FlattenOutsideHours = true;
				AllowManualOrders   = true;
			}
			else if (State == State.DataLoaded)
			{
				atr = ATR(AtrPeriod);
				// Las horas de Time[0] vienen en la zona configurada en NinjaTrader
				// (Tools > Options > General). El horario se evalúa en hora de Nueva York
				// para que el cambio de horario de verano (EE. UU.) se aplique solo.
				try   { chartTz = NinjaTrader.Core.Globals.GeneralOptions.TimeZoneInfo; }
				catch { chartTz = null; }
				if (chartTz == null)
					chartTz = TimeZoneInfo.Local;
				newYorkTz = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
			}
			else if (State == State.Realtime)
			{
				heartbeatTimer = new System.Threading.Timer(o => Heartbeat(), null, 1000, 1000);
				Log("Estrategia en tiempo real. Cuenta: " + Account.Name);
			}
			else if (State == State.Terminated)
			{
				if (heartbeatTimer != null)
				{
					heartbeatTimer.Dispose();
					heartbeatTimer = null;
				}
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
				Log(string.Format("Pérdida diaria máxima alcanzada ({0:C}). Cerrando y deteniendo hasta la próxima sesión.", dailyPnl));
				Flatten("AI DailyLoss");
			}
			if (haltedToday)
				return;

			int now = ToTime(NewYorkTime(Time[0]));
			if (now < StartTime || now > EndTime)
			{
				if (FlattenOutsideHours && Position.MarketPosition != MarketPosition.Flat)
				{
					Log("Fuera de horario, cerrando posición.");
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
				catch (Exception ex) { requestInFlight = false; Log("Error despachando respuesta: " + ex.Message); }
			});
		}

		private void HandleResponse(string response, string error, int barAtRequest)
		{
			requestInFlight = false;
			if (State != State.Realtime || haltedToday)
				return;

			if (error != null)
			{
				Log(string.Format("Sin respuesta del servidor ({0}). No se opera.", error));
				return;
			}
			if (CurrentBar != barAtRequest)
			{
				Log("Respuesta llegó tarde (vela ya cerrada). Se descarta.");
				return;
			}

			string action     = JsonString(response, "action").ToUpperInvariant();
			double confidence = JsonNumber(response, "confidence");
			string reason     = JsonString(response, "reason");

			Log(string.Format("{0} conf={1:0.00} pos={2} | {3}", action, confidence, Position.MarketPosition, reason));

			if (action != "EXIT" && confidence < MinConfidence)
				return;

			// Si la decisión viene de un setup, trae su stop estructural (y objetivo) en ticks
			Execute(action, (int)JsonNumber(response, "stop_ticks"), (int)JsonNumber(response, "target_ticks"));
		}

		// setupStop/setupTarget > 0: stop estructural del setup. 0: stop por ATR (IA libre y órdenes manuales).
		private string Execute(string action, int setupStop = 0, int setupTarget = 0)
		{
			MarketPosition mp = Position.MarketPosition;
			int stopTicks, targetTicks;

			switch (action)
			{
				case "BUY":
					if (mp == MarketPosition.Long) return "Ya hay posición larga";
					if (mp == MarketPosition.Short && !AllowReversal) { ExitShort(Position.Quantity, "AI Exit", ShortSignal); return "Cerrando corto"; }
					if (!Bracket(setupStop, setupTarget, out stopTicks, out targetTicks)) return "Stop del setup mayor que el máximo permitido";
					if (!CanOpen(stopTicks)) return "Bloqueada por reglas de riesgo (ver registro)";
					SetBracket(LongSignal, stopTicks, targetTicks);
					EnterLong(Quantity, LongSignal);
					tradesToday++;
					return "Orden de compra enviada";

				case "SELL":
					if (mp == MarketPosition.Short) return "Ya hay posición corta";
					if (mp == MarketPosition.Long && (!AllowReversal || !AllowShorts)) { ExitLong(Position.Quantity, "AI Exit", LongSignal); return "Cerrando largo"; }
					if (!AllowShorts) return "Cortos deshabilitados";
					if (!Bracket(setupStop, setupTarget, out stopTicks, out targetTicks)) return "Stop del setup mayor que el máximo permitido";
					if (!CanOpen(stopTicks)) return "Bloqueada por reglas de riesgo (ver registro)";
					SetBracket(ShortSignal, stopTicks, targetTicks);
					EnterShort(Quantity, ShortSignal);
					tradesToday++;
					return "Orden de venta enviada";

				case "EXIT":
					if (mp == MarketPosition.Flat) return "No hay posición abierta";
					Flatten("AI Exit");
					return "Cerrando posición";
			}
			return "Sin acción";
		}

		// Calcula stop y objetivo en ticks. Con setup: el stop nunca es más ajustado que el
		// mínimo (se amplía y el objetivo se escala para mantener el R:R) y, si supera el
		// máximo, NO se opera (recortar un stop estructural rompe la idea del setup).
		private bool Bracket(int setupStop, int setupTarget, out int stopTicks, out int targetTicks)
		{
			double atrTicks = atr[0] / TickSize;
			if (setupStop > 0)
			{
				stopTicks = Math.Max(MinStopTicks, setupStop);
				targetTicks = setupTarget > 0 ? (int)Math.Round(setupTarget * (double)stopTicks / setupStop) : stopTicks * 2;
				if (stopTicks > MaxStopTicks)
				{
					Log(string.Format("Setup descartado: stop de {0} ticks > máximo {1}.", stopTicks, MaxStopTicks));
					return false;
				}
				return true;
			}
			stopTicks = Math.Min(MaxStopTicks, Math.Max(MinStopTicks, (int)Math.Round(atrTicks * StopAtrMult)));
			targetTicks = Math.Max(stopTicks, (int)Math.Round(atrTicks * TargetAtrMult));
			return true;
		}

		private bool CanOpen(int stopTicks)
		{
			if (tradesToday >= MaxTradesPerDay)
			{
				Log(string.Format("Máximo de trades diarios ({0}) alcanzado.", MaxTradesPerDay));
				return false;
			}

			// Solo se abre si, tocando el stop, el día no supera la pérdida máxima.
			// Si hay una posición contraria abierta (reversión), su PnL flotante ya cuenta.
			double budget = MaxDailyLoss + DailyPnl();
			double risk   = stopTicks * TickSize * Instrument.MasterInstrument.PointValue * Quantity;
			if (risk > budget)
			{
				Log(string.Format("Entrada bloqueada: riesgo del stop {0:C} > margen restante del día {1:C}.", risk, budget));
				return false;
			}
			return true;
		}

		// Stop y target se fijan ANTES de la entrada, así la orden de protección
		// sale en cuanto se llena la entrada.
		private void SetBracket(string signal, int stopTicks, int targetTicks)
		{
			SetStopLoss(signal, CalculationMode.Ticks, stopTicks, false);
			SetProfitTarget(signal, CalculationMode.Ticks, targetTicks);
			Log(string.Format("{0}: stop={1} ticks, target={2} ticks", signal, stopTicks, targetTicks));
		}

		private DateTime NewYorkTime(DateTime chartTime)
		{
			return TimeZoneInfo.ConvertTime(DateTime.SpecifyKind(chartTime, DateTimeKind.Unspecified), chartTz, newYorkTz);
		}

		// ===================== Panel (ai_server) =====================

		private void Log(string msg)
		{
			Print(DateTime.Now.ToString("HH:mm:ss") + " [AI] " + msg);
			lock (panelLock)
			{
				if (pendingEvents.Count < 200)
					pendingEvents.Add(msg);
			}
		}

		private string HeartbeatUrl()
		{
			string url = ServerUrl ?? "";
			return url.EndsWith("/decide") ? url.Substring(0, url.Length - "/decide".Length) + "/nt/heartbeat" : url.TrimEnd('/') + "/nt/heartbeat";
		}

		// Corre en un hilo del temporizador: el estado se lee en el hilo de la
		// estrategia (TriggerCustomEvent) y la llamada HTTP va en segundo plano.
		private void Heartbeat()
		{
			if (heartbeatInFlight || State != State.Realtime)
				return;
			heartbeatInFlight = true;
			try
			{
				TriggerCustomEvent(o =>
				{
					string payload, url, token;
					try
					{
						payload = BuildHeartbeat();
						url = HeartbeatUrl();
						token = ApiToken;
					}
					catch (Exception ex)
					{
						heartbeatInFlight = false;
						Print("[AI] Error preparando latido: " + ex.Message);
						return;
					}
					Task.Run(() =>
					{
						string response = null;
						try { response = Post(url, token, payload, 3000); }
						catch { }
						try
						{
							if (response != null)
								TriggerCustomEvent(x => HandleCommands(response), null);
						}
						catch { }
						finally { heartbeatInFlight = false; }
					});
				}, null);
			}
			catch { heartbeatInFlight = false; }
		}

		private string BuildHeartbeat()
		{
			CultureInfo ic = CultureInfo.InvariantCulture;
			string pos = Position.MarketPosition == MarketPosition.Long ? "LONG"
			           : Position.MarketPosition == MarketPosition.Short ? "SHORT" : "FLAT";
			int now = ToTime(NewYorkTime(Time[0]));
			double lastPrice = Close[0];
			try { if (Bars.GetClose(Bars.Count - 1) > 0) lastPrice = Bars.GetClose(Bars.Count - 1); } catch { }
			double unrealized = Position.MarketPosition == MarketPosition.Flat ? 0 : Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, lastPrice);

			var sb = new StringBuilder(512);
			sb.Append('{');
			sb.AppendFormat(ic, "\"account\":\"{0}\",", Escape(Account.Name));
			sb.AppendFormat(ic, "\"instrument\":\"{0}\",", Escape(Instrument.FullName));
			sb.AppendFormat(ic, "\"state\":\"{0}\",", State);
			sb.AppendFormat(ic, "\"position\":\"{0}\",", pos);
			sb.AppendFormat(ic, "\"qty\":{0},", Position.Quantity);
			sb.AppendFormat(ic, "\"avg_price\":{0},", Position.AveragePrice);
			sb.AppendFormat(ic, "\"unrealized_pnl\":{0:0.##},", unrealized);
			sb.AppendFormat(ic, "\"daily_pnl\":{0:0.##},", DailyPnl());
			sb.AppendFormat(ic, "\"max_daily_loss\":{0},", MaxDailyLoss);
			sb.AppendFormat(ic, "\"trades_today\":{0},", tradesToday);
			sb.AppendFormat(ic, "\"max_trades\":{0},", MaxTradesPerDay);
			sb.AppendFormat(ic, "\"halted\":{0},", haltedToday ? "true" : "false");
			sb.AppendFormat(ic, "\"in_hours\":{0},", now >= StartTime && now <= EndTime ? "true" : "false");
			sb.AppendFormat(ic, "\"last_price\":{0},", lastPrice);
			sb.AppendFormat(ic, "\"bar_time\":\"{0:yyyy-MM-ddTHH:mm:ss}\",", Time[0]);
			sb.AppendFormat(ic, "\"manual_enabled\":{0},", AllowManualOrders ? "true" : "false");

			List<string> events, acks;
			lock (panelLock)
			{
				events = new List<string>(pendingEvents);
				acks   = new List<string>(pendingAcks);
				pendingEvents.Clear();
				pendingAcks.Clear();
			}
			sb.Append("\"events\":[");
			for (int i = 0; i < events.Count; i++)
				sb.Append(i > 0 ? "," : "").Append('"').Append(Escape(events[i])).Append('"');
			sb.Append("],\"acks\":[").Append(string.Join(",", acks)).Append("]}");
			return sb.ToString();
		}

		private void HandleCommands(string response)
		{
			if (State != State.Realtime)
				return;
			foreach (Match m in Regex.Matches(response, "\"id\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"type\"\\s*:\\s*\"([A-Z]+)\""))
			{
				string id = m.Groups[1].Value, type = m.Groups[2].Value, result;
				if (type == "PING")
					result = string.Format("PONG · cuenta {0} · {1} · posición {2}", Account.Name, Instrument.FullName, Position.MarketPosition);
				else if (!AllowManualOrders)
					result = "Órdenes manuales deshabilitadas en la estrategia";
				else if (haltedToday && type != "EXIT")
					result = "Bloqueado: pérdida diaria máxima alcanzada";
				else
					result = Execute(type);

				Log(string.Format("Panel: {0} → {1}", type, result));
				lock (panelLock)
					pendingAcks.Add(string.Format("{{\"id\":\"{0}\",\"result\":\"{1}\"}}", Escape(id), Escape(result)));
			}
		}

		protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
			MarketPosition marketPosition, string orderId, DateTime time)
		{
			if (execution.Order != null)
				Log(string.Format("Ejecutada: {0} {1} @ {2} ({3})", execution.Order.OrderAction, quantity,
					price.ToString("0.00", CultureInfo.InvariantCulture), execution.Order.Name));
		}

		protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled,
			double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
		{
			if (orderState == OrderState.Rejected)
				Log(string.Format("Orden RECHAZADA: {0} {1} — {2}", order.OrderAction, order.Name, comment));
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
			sb.AppendFormat(ic, "\"start_time\":{0},\"end_time\":{1},", StartTime, EndTime);
			sb.Append("\"bars\":[");
			for (int i = BarsToSend - 1; i >= 0; i--)
			{
				sb.AppendFormat(ic, "{{\"t\":\"{0:yyyy-MM-ddTHH:mm:ss}\",\"o\":{1},\"h\":{2},\"l\":{3},\"c\":{4},\"v\":{5}}}",
					NewYorkTime(Time[i]), Open[i], High[i], Low[i], Close[i], Volume[i]);   // hora de NY, como el backtest
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
			return (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
			                .Replace("\r", " ").Replace("\n", " ").Replace("\t", " ");
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
		[Range(60, 5000)]
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
		[Display(Name = "Hora inicio NY (HHmmss)", Order = 1, GroupName = "4. Horario")]
		public int StartTime { get; set; }

		[NinjaScriptProperty]
		[Range(0, 235959)]
		[Display(Name = "Hora fin NY (HHmmss)", Order = 2, GroupName = "4. Horario")]
		public int EndTime { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Cerrar fuera de horario", Order = 3, GroupName = "4. Horario")]
		public bool FlattenOutsideHours { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Permitir órdenes desde el panel", Order = 1, GroupName = "5. Panel")]
		public bool AllowManualOrders { get; set; }
		#endregion
	}
}

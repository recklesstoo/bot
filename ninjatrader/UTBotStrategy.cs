#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

// Conversión a NinjaTrader 8 (NinjaScript, C#) del indicador Pine Script v4
// "UT Bot Alerts" (autor original: HPotter / QuantNomad), convertido en una
// estrategia que abre y cierra posiciones automáticamente (stop-and-reverse
// sobre el cruce del precio con el ATR Trailing Stop) y que además usa el
// propio ATR Trailing Stop como stop de protección/salida.
namespace NinjaTrader.NinjaScript.Strategies
{
    public class UTBotStrategy : Strategy
    {
        private ATR atr;

        // Series persistente del ATR Trailing Stop (equivalente a xATRTrailingStop en Pine)
        private Series<double> trailStop;

        #region Properties

        [NinjaScriptProperty]
        [Range(0.01, double.MaxValue)]
        [Display(Name = "Key Value", Description = "Sensibilidad del ATR Trailing Stop ('a' en el script original)", Order = 1, GroupName = "Parameters")]
        public double KeyValue { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "ATR Period", Description = "Periodo del ATR ('c' en el script original)", Order = 2, GroupName = "Parameters")]
        public int AtrPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use Heikin Ashi Source", Description = "Usar el cierre Heikin Ashi como precio fuente ('h' en el script original)", Order = 3, GroupName = "Parameters")]
        public bool UseHeikinAshi { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Trade Quantity", Description = "Cantidad de contratos/acciones por operación", Order = 4, GroupName = "Order Management")]
        public int TradeQuantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Long Entries", Description = "Permitir entradas en largo", Order = 5, GroupName = "Order Management")]
        public bool EnableLongEntries { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Short Entries", Description = "Permitir entradas en corto", Order = 6, GroupName = "Order Management")]
        public bool EnableShortEntries { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use ATR Trailing Stop Exit", Description = "Ademas de la reversion por señal contraria, usar el ATR Trailing Stop como stop de salida", Order = 7, GroupName = "Order Management")]
        public bool UseAtrTrailingStopExit { get; set; }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Estrategia basada en el indicador 'UT Bot Alerts' (ATR Trailing Stop). " +
                              "Compra cuando el precio cruza al alza el ATR Trailing Stop y vende/cortocircuita " +
                              "cuando el precio cruza a la baja, replicando las señales buy/sell del script Pine original.";
                Name = "UTBotStrategy";
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
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
                IsInstantiatedOnEachOptimizationIteration = true;
                IsOverlay = true;

                // Valores por defecto equivalentes a los inputs del Pine Script
                KeyValue = 1;
                AtrPeriod = 10;
                UseHeikinAshi = false;
                TradeQuantity = 1;
                EnableLongEntries = true;
                EnableShortEntries = true;
                UseAtrTrailingStopExit = true;

                AddPlot(Brushes.DodgerBlue, "UTBotTrailStop");
            }
            else if (State == State.Configure)
            {
            }
            else if (State == State.DataLoaded)
            {
                atr = ATR(AtrPeriod);
                trailStop = new Series<double>(this);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < Math.Max(AtrPeriod, 1))
                return;

            double atrValue = atr[0];
            double nLoss = KeyValue * atrValue;

            // HA_Close = (Open+High+Low+Close)/4, único valor de la vela Heikin Ashi
            // que no depende de la recursión de HA_Open, así que se calcula directo.
            double src = UseHeikinAshi ? (Open[0] + High[0] + Low[0] + Close[0]) / 4.0 : Close[0];
            double srcPrev = CurrentBar > 0
                ? (UseHeikinAshi ? (Open[1] + High[1] + Low[1] + Close[1]) / 4.0 : Close[1])
                : src;

            double prevTrail = CurrentBar > 0 ? trailStop[1] : 0.0;

            double newTrail;
            if (src > prevTrail && srcPrev > prevTrail)
                newTrail = Math.Max(prevTrail, src - nLoss);
            else if (src < prevTrail && srcPrev < prevTrail)
                newTrail = Math.Min(prevTrail, src + nLoss);
            else if (src > prevTrail)
                newTrail = src - nLoss;
            else
                newTrail = src + nLoss;

            trailStop[0] = newTrail;
            Values[0][0] = newTrail;

            // Señales de cruce equivalentes a "buy"/"sell" del script original
            // (buy = crossover(src, xATRTrailingStop); sell = crossunder(src, xATRTrailingStop))
            bool buySignal = CurrentBar > 0 && srcPrev <= prevTrail && src > newTrail;
            bool sellSignal = CurrentBar > 0 && srcPrev >= prevTrail && src < newTrail;

            if (buySignal)
            {
                Draw.ArrowUp(this, "UTBotBuy" + CurrentBar, true, 0, Low[0] - 2 * TickSize, Brushes.LimeGreen);

                if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("UTBotExitShort", "UTBotShort");

                if (EnableLongEntries && Position.MarketPosition != MarketPosition.Long)
                    EnterLong(TradeQuantity, "UTBotLong");
            }
            else if (sellSignal)
            {
                Draw.ArrowDown(this, "UTBotSell" + CurrentBar, true, 0, High[0] + 2 * TickSize, Brushes.Red);

                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("UTBotExitLong", "UTBotLong");

                if (EnableShortEntries && Position.MarketPosition != MarketPosition.Short)
                    EnterShort(TradeQuantity, "UTBotShort");
            }

            if (UseAtrTrailingStopExit)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLongStopMarket(0, true, Position.Quantity, newTrail, "UTBotTrailStopExit", "UTBotLong");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShortStopMarket(0, true, Position.Quantity, newTrail, "UTBotTrailStopExit", "UTBotShort");
            }
        }
    }
}

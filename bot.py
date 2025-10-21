
import os, time, math, sys, traceback
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone
from dotenv import load_dotenv

# =======================
# Configuración y helpers
# =======================
load_dotenv()

API_KEY   = os.getenv("BINANCE_API_KEY", "")
API_SECRET= os.getenv("BINANCE_API_SECRET", "")
SYMBOL    = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
RISK_PCT  = float(os.getenv("RISK_PCT", "0.0025"))  # 0.25% del capital
TESTNET   = os.getenv("TESTNET", "1") == "1"       # Por defecto, testnet activado
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "30"))       # cada cuánto revisa en segundos (live loop)

if not API_KEY or not API_SECRET:
    print("Faltan BINANCE_API_KEY o BINANCE_API_SECRET en variables de entorno.")
    sys.exit(1)

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {msg}", flush=True)

def init_exchange():
    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    # Modo sandbox (testnet)
    exchange.set_sandbox_mode(TESTNET)
    return exchange

# ===========
# Indicadores
# ===========
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14):
    high = df["high"]
    low  = df["low"]
    close= df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =================
# Mercado y señales
# =================
def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df.dropna()

def generate_signal(df: pd.DataFrame):
    """
    Estrategia 'EMA SmartFlow':
      LONG: cruce ema20>ema50 en la vela actual, RSI 45-60, volumen > 1.2*vol_ma20, close>ema20
      EXIT LONG (señal de venta): ema20<ema50 o RSI>70
    Devuelve: {"signal": "BUY"/"SELL"/"NONE", "reason": str}
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    long_cond = (
        (prev["ema20"] <= prev["ema50"]) and (last["ema20"] > last["ema50"]) and
        (45 < last["rsi14"] < 60) and
        (last["volume"] > 1.2 * last["vol_ma20"]) and
        (last["close"] > last["ema20"])
    )
    exit_cond = (
        (last["ema20"] < last["ema50"]) or
        (last["rsi14"] > 70)
    )

    if long_cond:
        return {"signal": "BUY", "reason": "EMA20>EMA50 + RSI(45-60) + Vol>1.2xMA20 + close>EMA20"}
    elif exit_cond:
        return {"signal": "SELL", "reason": "EMA20<EMA50 o RSI>70"}
    else:
        return {"signal": "NONE", "reason": "Sin condiciones"}

# ======================
# Tamaño de posición SL/TP
# ======================
def get_free_usdt(exchange):
    balance = exchange.fetch_balance()
    # en spot binance, 'USDT' suele estar en 'free'
    return float(balance.get('free', {}).get('USDT', 0.0) or balance['USDT']['free'])

def market_info(exchange, symbol):
    exchange.load_markets()
    mkt = exchange.markets[symbol]
    # Filtros de cantidad/notional
    lot_step = mkt.get("limits", {}).get("amount", {}).get("min", None)
    prec_amt = mkt.get("precision", {}).get("amount", None)
    prec_px  = mkt.get("precision", {}).get("price", None)
    min_notional = None
    for f in mkt.get("info", {}).get("filters", []):
        if f.get("filterType") == "MIN_NOTIONAL":
            min_notional = float(f.get("minNotional"))
    return mkt, lot_step, prec_amt, prec_px, min_notional

def round_amount(exchange, symbol, amount):
    mkt = exchange.market(symbol)
    return exchange.amount_to_lots(symbol, amount) if hasattr(exchange, "amount_to_lots") else float(exchange.amount_to_precision(symbol, amount))

def round_price(exchange, symbol, price):
    return float(exchange.price_to_precision(symbol, price))

def position_size_from_risk(balance_usdt, entry_price, atr_val, risk_pct):
    # Riesgo = 0.25% del capital. Stop = 1*ATR => cantidad = (balance*risk) / ATR
    risk_usdt = balance_usdt * risk_pct
    if atr_val <= 0:
        return 0.0
    qty = risk_usdt / atr_val
    # no superar el balance disponible en notional
    max_qty_by_balance = (balance_usdt * 0.98) / entry_price  # 98% por seguridad
    return max(0.0, min(qty, max_qty_by_balance))

# ==================
# Órdenes en Binance
# ==================
def buy_market(exchange, symbol, amount):
    return exchange.create_order(symbol, type="market", side="buy", amount=amount)

def sell_market(exchange, symbol, amount):
    return exchange.create_order(symbol, type="market", side="sell", amount=amount)

# =====================
# Ciclo principal (loop)
# =====================
def main_loop():
    exchange = init_exchange()
    mkt, lot_step, prec_amt, prec_px, min_notional = market_info(exchange, SYMBOL)
    log(f"Iniciado. SYMBOL={SYMBOL} TF={TIMEFRAME} TESTNET={'ON' if TESTNET else 'OFF'}")

    in_position = False
    entry_price = None
    atr_entry   = None
    pos_qty     = 0.0
    tp_price    = None
    sl_price    = None

    while True:
        try:
            df = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME, limit=200)
            sig = generate_signal(df)
            last = df.iloc[-1]

            ticker = exchange.fetch_ticker(SYMBOL)
            price  = float(ticker["last"])

            # Si estamos dentro, gestionar salida por TP/SL
            if in_position:
                # Cerrar por SL/TP
                if price <= sl_price:
                    log(f"🛑 SL alcanzado @ {price:.2f} (SL {sl_price:.2f}) → VENTA MARKET")
                    sell_market(exchange, SYMBOL, pos_qty)
                    in_position = False
                    continue
                if price >= tp_price:
                    log(f"✅ TP alcanzado @ {price:.2f} (TP {tp_price:.2f}) → VENTA MARKET")
                    sell_market(exchange, SYMBOL, pos_qty)
                    in_position = False
                    continue

                # Cerrar si aparece señal de salida por estrategia
                if sig["signal"] == "SELL":
                    log(f"↘️ Señal de salida: {sig['reason']} → VENTA MARKET @ {price:.2f}")
                    sell_market(exchange, SYMBOL, pos_qty)
                    in_position = False
                    continue

                log(f"⏳ En posición | Precio {price:.2f} | TP {tp_price:.2f} | SL {sl_price:.2f}")
                time.sleep(SLEEP_SEC)
                continue

            # Si NO estamos dentro, evaluar entrada
            if sig["signal"] == "BUY":
                balance = get_free_usdt(exchange)
                atr_val = float(last["atr14"])
                entry   = price
                qty_raw = position_size_from_risk(balance, entry, atr_val, RISK_PCT)
                qty     = round_amount(exchange, SYMBOL, qty_raw)

                # Validar notional mínimo
                notional = qty * entry
                if min_notional and notional < min_notional:
                    log(f"⚠️ Notional {notional:.2f} < min_notional {min_notional:.2f}. No se compra.")
                elif qty <= 0:
                    log("⚠️ Tamaño calculado <= 0. No se compra.")
                else:
                    log(f"📈 BUY MARKET {SYMBOL} qty={qty} @ {entry:.2f} | razón: {sig['reason']}")
                    order = buy_market(exchange, SYMBOL, qty)
                    filled_price = entry  # aproximamos a precio actual; para más precisión, leer fills si los expone
                    in_position = True
                    entry_price = filled_price
                    atr_entry   = atr_val
                    pos_qty     = qty
                    sl_price    = round_price(exchange, SYMBOL, entry_price - atr_entry * 1.0)  # SL = 1*ATR
                    tp_price    = round_price(exchange, SYMBOL, entry_price + atr_entry * 2.0)  # TP = 2*ATR
                    log(f"🎯 TP={tp_price:.2f} | 🛡 SL={sl_price:.2f}")
            else:
                log(f"Sin señal ({sig['reason']}) | Precio {price:.2f}")

            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            log(f"🌐 NetworkError: {e}. Reintentando…")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            log(f"🏦 ExchangeError: {e}")
            time.sleep(5)
        except Exception as e:
            log("💥 Error inesperado:")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main_loop()

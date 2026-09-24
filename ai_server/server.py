"""
Servidor local de decisiones para NinjaTrader.

NinjaTrader envía las últimas velas al cierre de cada barra (POST /decide),
este servidor calcula indicadores, pregunta a un LLM local (Ollama) y devuelve
{"action": "BUY|SELL|HOLD|EXIT", "confidence": 0-1, "reason": "..."}.

Ante cualquier error (Ollama caído, JSON inválido, datos insuficientes) la
respuesta es siempre HOLD: el bot nunca opera a ciegas.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from indicators import add_indicators

load_dotenv()

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL          = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
TEMPERATURE    = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "15"))
API_TOKEN      = os.getenv("AI_API_TOKEN", "")
MIN_BARS       = int(os.getenv("MIN_BARS", "60"))
BARS_IN_PROMPT = int(os.getenv("BARS_IN_PROMPT", "20"))
LOG_FILE       = Path(os.getenv("DECISION_LOG", "decisions.jsonl"))

VALID_ACTIONS = {"BUY", "SELL", "HOLD", "EXIT"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai_server")

app = FastAPI(title="NinjaTrader AI decision server")


class Bar(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float


class DecideRequest(BaseModel):
    instrument: str
    timeframe: str = ""
    tick_size: float = 0.25
    point_value: float = 1.0
    position: str = "FLAT"          # FLAT | LONG | SHORT
    position_qty: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    trades_today: int = 0
    allow_shorts: bool = True
    bars: List[Bar]


class Decision(BaseModel):
    action: str
    confidence: float
    reason: str


SYSTEM_PROMPT = """You are a disciplined intraday futures trader.
You receive recent OHLCV bars, technical indicators and the current position.
Decide ONE action for the next bar:
- BUY: open a long (or keep/turn long)
- SELL: open a short (or keep/turn short)
- EXIT: close the current position
- HOLD: do nothing
Rules:
- Only trade with a clear edge: trend alignment (EMAs), momentum (RSI/MACD) and volume confirmation.
- Prefer HOLD in choppy, low-volume or unclear conditions. Capital preservation first.
- If already in a position that is still valid, answer HOLD (do not add).
- Stops and targets are handled automatically; do not mention prices for them.
Reply ONLY with JSON: {"action": "BUY|SELL|HOLD|EXIT", "confidence": <0.0-1.0>, "reason": "<short explanation in Spanish>"}"""


def hold(reason: str) -> Decision:
    return Decision(action="HOLD", confidence=0.0, reason=reason)


def to_frame(bars: List[Bar]) -> pd.DataFrame:
    df = pd.DataFrame([{"time": b.t, "open": b.o, "high": b.h, "low": b.l,
                        "close": b.c, "volume": b.v} for b in bars])
    return add_indicators(df)


def fmt(x: float, nd: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def build_prompt(req: DecideRequest, df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    atr_v = last["atr14"] if last["atr14"] > 0 else float("nan")
    rows = ["time | open | high | low | close | volume"]
    for _, r in df.tail(BARS_IN_PROMPT).iterrows():
        rows.append(f"{r['time']} | {r['open']:.2f} | {r['high']:.2f} | {r['low']:.2f} | {r['close']:.2f} | {int(r['volume'])}")

    shorts = "allowed" if req.allow_shorts else "NOT allowed (never answer SELL to open a short)"
    return f"""Instrument: {req.instrument}  Timeframe: {req.timeframe}  Tick size: {req.tick_size}
Shorts: {shorts}

Last {BARS_IN_PROMPT} bars (oldest first):
{chr(10).join(rows)}

Indicators (last closed bar):
- Close: {fmt(last['close'])}
- EMA9: {fmt(last['ema9'])}  EMA21: {fmt(last['ema21'])}  EMA50: {fmt(last['ema50'])}
- Close vs EMA21 in ATRs: {fmt((last['close'] - last['ema21']) / atr_v)}
- RSI14: {fmt(last['rsi14'], 1)}
- MACD histogram: {fmt(last['macd_hist'], 3)} (previous: {fmt(df['macd_hist'].iloc[-2], 3)})
- ATR14: {fmt(last['atr14'])}
- Volume / 20-bar avg: {fmt(last['volume'] / last['vol_ma20'])}
- 20-bar high: {fmt(last['high20'])}  20-bar low: {fmt(last['low20'])}

Account:
- Position: {req.position} qty={req.position_qty} avg_price={fmt(req.avg_price)} unrealized_pnl={fmt(req.unrealized_pnl)}
- Daily PnL: {fmt(req.daily_pnl)}  Trades today: {req.trades_today}

What is your action?"""


def ask_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "format": "json",
            "options": {"temperature": TEMPERATURE},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_decision(raw: str, req: DecideRequest) -> Decision:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return hold("Respuesta de la IA no es JSON válido")

    action = str(data.get("action", "HOLD")).strip().upper()
    if action not in VALID_ACTIONS:
        return hold(f"Acción desconocida: {action}")
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(data.get("reason", ""))[:300]

    position = req.position.upper()
    if action == "EXIT" and position == "FLAT":
        return hold("EXIT sin posición abierta")
    if action == "SELL" and not req.allow_shorts and position != "LONG":
        return hold("Cortos deshabilitados")
    return Decision(action=action, confidence=confidence, reason=reason)


def write_log(req: DecideRequest, decision: Decision) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "instrument": req.instrument,
        "bar_time": req.bars[-1].t if req.bars else None,
        "close": req.bars[-1].c if req.bars else None,
        "position": req.position,
        "daily_pnl": req.daily_pnl,
        **decision.model_dump(),
    }
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("No se pudo escribir el log: %s", e)


@app.get("/")
def root():
    return {"status": "Servidor del bot funcionando", "health": "/health", "decide": "POST /decide"}


@app.get("/health")
def health():
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        return {"ok": MODEL in models, "model": MODEL, "available_models": models}
    except Exception as e:
        return {"ok": False, "model": MODEL, "error": str(e)}


@app.post("/decide", response_model=Decision)
def decide(req: DecideRequest, x_api_token: Optional[str] = Header(default=None)):
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")

    if len(req.bars) < MIN_BARS:
        decision = hold(f"Pocas velas ({len(req.bars)} < {MIN_BARS})")
    else:
        try:
            prompt = build_prompt(req, to_frame(req.bars))
            decision = parse_decision(ask_ollama(prompt), req)
        except requests.RequestException as e:
            decision = hold(f"Error llamando a Ollama: {e}")
        except Exception as e:
            log.exception("Error inesperado")
            decision = hold(f"Error inesperado: {e}")

    log.info("%s %s pos=%s -> %s (%.2f) %s", req.instrument, req.bars[-1].t if req.bars else "-",
             req.position, decision.action, decision.confidence, decision.reason)
    write_log(req, decision)
    return decision


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))

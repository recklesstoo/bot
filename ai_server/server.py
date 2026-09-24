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
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import backtest as bt
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
CHAT_TIMEOUT   = float(os.getenv("OLLAMA_CHAT_TIMEOUT", "90"))
# Hosts aceptados en la cabecera Host (protege contra DNS rebinding desde webs externas)
ALLOWED_HOSTS  = {h.strip().lower() for h in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost,::1").split(",") if h.strip()}
NT_TIMEOUT_SEC = 5      # sin latido de NinjaTrader durante este tiempo => desconectado
COMMAND_TTL    = 15     # un comando del panel no recogido en este tiempo caduca (nunca se ejecuta tarde)
STATIC_DIR     = Path(__file__).resolve().parent / "static"

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


# =====================================================================
# Estado en memoria para el panel (se pierde al reiniciar, salvo el log)
# =====================================================================
_lock = threading.Lock()
STATE = {"paused": False, "nt": None, "nt_seen": 0.0, "last_request": None,
         "ollama": {"ok": None, "models": [], "error": None, "checked": 0.0}}
DECISIONS: deque = deque(maxlen=200)
EVENTS: deque = deque(maxlen=300)
COMMANDS: "OrderedDict[str, dict]" = OrderedDict()
PENDING: deque = deque()
MANUAL_TYPES = {"PING", "BUY", "SELL", "EXIT"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_event(source: str, text: str) -> None:
    with _lock:
        EVENTS.append({"ts": now_iso(), "source": source, "text": text})


def nt_connected() -> bool:
    return STATE["nt"] is not None and time.time() - STATE["nt_seen"] < NT_TIMEOUT_SEC


def load_recent_decisions(n: int = 50) -> None:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return
    for line in lines:
        try:
            DECISIONS.append(json.loads(line))
        except json.JSONDecodeError:
            pass


def write_log(req: DecideRequest, decision: Decision, latency_ms: Optional[int] = None) -> None:
    entry = {
        "ts": now_iso(),
        "instrument": req.instrument,
        "bar_time": req.bars[-1].t if req.bars else None,
        "close": req.bars[-1].c if req.bars else None,
        "position": req.position,
        "daily_pnl": req.daily_pnl,
        "latency_ms": latency_ms,
        **decision.model_dump(),
    }
    with _lock:
        DECISIONS.append(entry)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("No se pudo escribir el log: %s", e)


def run_decision(req: DecideRequest):
    """Pipeline completo: indicadores -> prompt -> Ollama -> validación."""
    t0 = time.monotonic()
    raw = None
    if len(req.bars) < MIN_BARS:
        decision = hold(f"Pocas velas ({len(req.bars)} < {MIN_BARS})")
    else:
        try:
            raw = ask_ollama(build_prompt(req, to_frame(req.bars)))
            decision = parse_decision(raw, req)
        except requests.RequestException as e:
            decision = hold(f"Error llamando a Ollama: {e}")
        except Exception as e:
            log.exception("Error inesperado")
            decision = hold(f"Error inesperado: {e}")
    return decision, int((time.monotonic() - t0) * 1000), raw


def check_ollama(max_age: float = 10.0) -> dict:
    info = STATE["ollama"]
    if time.time() - info["checked"] < max_age:
        return info
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        models = [m["name"] for m in tags.get("models", [])]
        info.update(ok=MODEL in models, models=models,
                    error=None if MODEL in models else f"Modelo {MODEL} no descargado")
    except Exception as e:
        info.update(ok=False, models=[], error=str(e))
    info["checked"] = time.time()
    return info


def check_token(x_api_token: Optional[str]) -> None:
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")


def require_panel(x_panel: Optional[str] = Header(default=None)) -> None:
    # Una cabecera propia obliga al navegador a hacer preflight CORS, que este
    # servidor no acepta: una web externa no puede pulsar los botones por ti.
    if x_panel != "1":
        raise HTTPException(status_code=403, detail="Solo desde el panel")


@app.middleware("http")
async def only_local_hosts(request: Request, call_next):
    host = (request.headers.get("host") or "").lower()
    host = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.split(":")[0]
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "Host no permitido"}, status_code=403)
    return await call_next(request)


# ============
# NinjaTrader
# ============
@app.get("/health")
def health():
    info = check_ollama(max_age=0)
    out = {"ok": bool(info["ok"]), "model": MODEL, "available_models": info["models"]}
    if info["error"]:
        out["error"] = info["error"]
    return out


@app.post("/decide", response_model=Decision)
def decide(req: DecideRequest, x_api_token: Optional[str] = Header(default=None)):
    check_token(x_api_token)
    STATE["last_request"] = req
    latency = None
    if STATE["paused"]:
        decision = hold("IA en pausa desde el panel")
    else:
        decision, latency, _ = run_decision(req)

    log.info("%s %s pos=%s -> %s (%.2f) %s", req.instrument, req.bars[-1].t if req.bars else "-",
             req.position, decision.action, decision.confidence, decision.reason)
    write_log(req, decision, latency)
    return decision


class Ack(BaseModel):
    id: str
    result: str


class Heartbeat(BaseModel):
    account: str = ""
    instrument: str = ""
    state: str = ""
    position: str = "FLAT"
    qty: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_daily_loss: float = 0.0
    trades_today: int = 0
    max_trades: int = 0
    halted: bool = False
    in_hours: bool = False
    last_price: float = 0.0
    bar_time: str = ""
    manual_enabled: bool = True
    events: List[str] = []
    acks: List[Ack] = []


@app.post("/nt/heartbeat")
def nt_heartbeat(hb: Heartbeat, x_api_token: Optional[str] = Header(default=None)):
    check_token(x_api_token)
    was_connected = nt_connected()
    now = time.time()
    with _lock:
        STATE["nt"] = hb.model_dump(exclude={"events", "acks"})
        STATE["nt_seen"] = now
        for text in hb.events:
            EVENTS.append({"ts": now_iso(), "source": "NT", "text": text})
        for ack in hb.acks:
            cmd = COMMANDS.get(ack.id)
            if cmd and cmd["status"] != "done":
                cmd.update(status="done", result=ack.result,
                           latency_ms=int((now - cmd["created"]) * 1000))
                EVENTS.append({"ts": now_iso(), "source": "NT",
                               "text": f"{cmd['type']} → {ack.result} ({cmd['latency_ms']} ms)"})
        out = []
        while PENDING:
            cmd = COMMANDS[PENDING.popleft()]
            if now - cmd["created"] > COMMAND_TTL:
                cmd.update(status="expired", result="Caducado: NinjaTrader no lo recogió a tiempo")
                continue
            cmd["status"] = "sent"
            out.append({"id": cmd["id"], "type": cmd["type"]})
    if not was_connected:
        add_event("Servidor", f"NinjaTrader conectado: {hb.account} {hb.instrument}")
    return {"commands": out}


# ======
# Panel
# ======
@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/state")
def api_state():
    info = check_ollama()
    with _lock:
        return {
            "server_time": now_iso(),
            "paused": STATE["paused"],
            "model": MODEL,
            "ollama": {"ok": info["ok"], "error": info["error"]},
            "nt": {"connected": nt_connected(),
                   "seconds_ago": round(time.time() - STATE["nt_seen"], 1) if STATE["nt"] else None,
                   "status": STATE["nt"]},
            "decisions": list(DECISIONS)[-50:][::-1],
            "events": list(EVENTS)[-100:][::-1],
            "commands": [ {k: v for k, v in c.items()} for c in list(COMMANDS.values())[-20:][::-1] ],
        }


class CommandIn(BaseModel):
    type: str


@app.post("/api/command", dependencies=[Depends(require_panel)])
def api_command(body: CommandIn):
    ctype = body.type.upper()
    if ctype not in MANUAL_TYPES:
        raise HTTPException(status_code=400, detail=f"Comando desconocido: {ctype}")
    if not nt_connected():
        raise HTTPException(status_code=409, detail="NinjaTrader no está conectado (¿estrategia activada?)")
    cmd = {"id": uuid.uuid4().hex[:10], "type": ctype, "created": time.time(),
           "created_iso": now_iso(), "status": "pending", "result": None, "latency_ms": None}
    with _lock:
        COMMANDS[cmd["id"]] = cmd
        PENDING.append(cmd["id"])
        while len(COMMANDS) > 100:
            COMMANDS.popitem(last=False)
    add_event("Panel", f"Enviado a NinjaTrader: {ctype}")
    return cmd


class PauseIn(BaseModel):
    paused: bool


@app.post("/api/pause", dependencies=[Depends(require_panel)])
def api_pause(body: PauseIn):
    STATE["paused"] = body.paused
    add_event("Panel", "IA en PAUSA: no abrirá operaciones" if body.paused else "IA REANUDADA")
    return {"paused": STATE["paused"]}


def demo_request() -> DecideRequest:
    base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    bars = []
    for i in range(100):
        c = 20000 + 25 * math.sin(i / 8) + i * 0.8
        t = (base - timedelta(minutes=5 * (99 - i))).strftime("%Y-%m-%dT%H:%M:%S")
        bars.append(Bar(t=t, o=c - 2, h=c + 6, l=c - 6, c=c, v=1500 + (i % 7) * 120))
    return DecideRequest(instrument="MNQ (DEMO)", timeframe="5 Minute", tick_size=0.25,
                         point_value=2.0, bars=bars)


@app.post("/api/test-ai", dependencies=[Depends(require_panel)])
def api_test_ai():
    req = STATE["last_request"]
    source = "últimas velas reales de NinjaTrader" if req else "velas de ejemplo (NinjaTrader aún no envió datos)"
    decision, latency, raw = run_decision(req or demo_request())
    add_event("Prueba IA", f"{decision.action} conf={decision.confidence:.2f} en {latency} ms — {decision.reason}")
    return {"source": source, "latency_ms": latency, "raw": raw, **decision.model_dump()}


class ChatIn(BaseModel):
    message: str


@app.post("/api/chat", dependencies=[Depends(require_panel)])
def api_chat(body: ChatIn):
    nt = STATE["nt"]
    last = DECISIONS[-1] if DECISIONS else None
    context = (f"Estado de NinjaTrader: {json.dumps(nt, ensure_ascii=False)}\n" if nt else "NinjaTrader no conectado.\n")
    if last:
        context += f"Última decisión del bot: {last.get('action')} ({last.get('confidence')}) — {last.get('reason')}\n"
    t0 = time.monotonic()
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", timeout=CHAT_TIMEOUT, json={
            "model": MODEL, "stream": False, "options": {"temperature": 0.3},
            "messages": [
                {"role": "system", "content": "Eres el asistente del bot de trading de futuros (MNQ) del usuario. "
                                              "Responde en español, breve y claro.\n" + context},
                {"role": "user", "content": body.message[:2000]},
            ]})
        resp.raise_for_status()
        reply = resp.json()["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama no respondió: {e}")
    return {"reply": reply, "latency_ms": int((time.monotonic() - t0) * 1000)}


# =========================================
# Backtest con datos históricos (panel)
# =========================================
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data")))
BT_HISTORY_FILE = DATA_DIR / "bt_history.json"
BT = {"running": False, "done": 0, "total": 0, "result": None, "error": None,
      "params": None, "started": None, "finished": None}
_bt_cancel = threading.Event()
STRATEGY_NAMES = {"qwen": "IA (Ollama)", "ema": "Cruce EMA (referencia)", "random": "Azar (referencia)"}


def _bt_history() -> list:
    try:
        return json.loads(BT_HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _safe_data_path(name: str) -> Path:
    path = DATA_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return path


@app.get("/api/backtest/files")
def bt_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((f for f in DATA_DIR.iterdir() if f.suffix.lower() in (".txt", ".csv")),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)} for f in files]


@app.post("/api/backtest/upload", dependencies=[Depends(require_panel)])
async def bt_upload(request: Request, name: str):
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name)[:120]
    if not safe.lower().endswith((".txt", ".csv")):
        raise HTTPException(status_code=400, detail="Sube el .txt exportado de NinjaTrader")
    body = await request.body()
    if len(body) > 300_000_000:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx. 300 MB)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / safe
    path.write_bytes(body)
    try:
        df = bt.load_nt_export(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Formato no reconocido (se espera 'yyyyMMdd HHmmss;O;H;L;C;V'): {e}")
    add_event("Backtest", f"Datos cargados: {safe} ({len(df)} velas, {df.index[0]:%Y-%m-%d} a {df.index[-1]:%Y-%m-%d})")
    return {"name": safe, "rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1])}


class BacktestIn(BaseModel):
    file: str
    strategy: str = "qwen"
    days: int = 0
    min_confidence: float = 0.65
    tz: str = "UTC"


def _downsample(points: list, n: int = 600) -> list:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def _bt_worker(path: Path, body: BacktestIn):
    t0 = time.time()

    def progress(done, total):
        BT.update(done=done, total=total)
    try:
        df1 = bt.slice_days(bt.load_nt_export(path, body.tz), body.days or None)
        strategy = bt.build_strategy(body.strategy, DATA_DIR)
        params = bt.Params(min_confidence=body.min_confidence)
        res = bt.Backtester(df1, params, strategy, progress, _bt_cancel).run()
        res["equity"] = _downsample(res["equity"])
        res["trades"] = res["trades"][-300:]
        res["meta"] = {"file": path.name, "strategy": body.strategy, "strategy_name": STRATEGY_NAMES.get(body.strategy, body.strategy),
                       "model": MODEL if body.strategy == "qwen" else None, "days": body.days,
                       "min_confidence": body.min_confidence, "seconds": round(time.time() - t0)}
        BT["result"] = res
        s = res["stats"]
        if not res["cancelled"]:
            hist = _bt_history()
            hist.append({"ts": now_iso(), **res["meta"], **{k: s[k] for k in (
                "trades", "net_pnl", "win_rate", "profit_factor", "max_drawdown", "expectancy", "green_days", "red_days")}})
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            BT_HISTORY_FILE.write_text(json.dumps(hist[-50:], ensure_ascii=False, indent=1), encoding="utf-8")
        add_event("Backtest", f"{'Cancelado' if res['cancelled'] else 'Terminado'}: {res['meta']['strategy_name']} → "
                              f"{s['trades']} trades, PnL {s['net_pnl']:+.2f} $, aciertos {s['win_rate']:.0%}")
    except Exception as e:
        log.exception("Backtest falló")
        BT["error"] = str(e)
        add_event("Backtest", f"Error: {e}")
    finally:
        BT.update(running=False, finished=now_iso())


@app.post("/api/backtest/start", dependencies=[Depends(require_panel)])
def bt_start(body: BacktestIn):
    if BT["running"]:
        raise HTTPException(status_code=409, detail="Ya hay un backtest en marcha")
    if body.strategy not in STRATEGY_NAMES:
        raise HTTPException(status_code=400, detail="Estrategia desconocida")
    path = _safe_data_path(body.file)
    if body.strategy == "qwen" and not check_ollama(max_age=0)["ok"]:
        raise HTTPException(status_code=409, detail="Ollama no está disponible: arráncalo antes de probar la IA")
    _bt_cancel.clear()
    BT.update(running=True, done=0, total=0, result=None, error=None, started=now_iso(), finished=None,
              started_ts=time.time(), params=body.model_dump())
    add_event("Backtest", f"Iniciado: {STRATEGY_NAMES[body.strategy]} con {path.name}"
                          + (f" (últimos {body.days} días)" if body.days else ""))
    threading.Thread(target=_bt_worker, args=(path, body), daemon=True).start()
    return {"started": True}


@app.post("/api/backtest/stop", dependencies=[Depends(require_panel)])
def bt_stop():
    _bt_cancel.set()
    return {"stopping": BT["running"]}


@app.get("/api/backtest/status")
def bt_status():
    out = {k: v for k, v in BT.items() if k != "started_ts"}
    if BT["running"] and BT["done"] > 0:
        elapsed = time.time() - BT.get("started_ts", time.time())
        out["eta_sec"] = int(elapsed / BT["done"] * max(BT["total"] - BT["done"], 0))
    out["history"] = _bt_history()[-12:][::-1]
    return out


load_recent_decisions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))

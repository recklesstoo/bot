"""
Motor de decisión por playbook: las reglas (setups.py) detectan un setup con su
stop estructural y, opcionalmente, la IA local actúa como FILTRO decidiendo si
tomarlo según el contexto (context.py) descrito en lenguaje de trader.

Lo usan igual el backtest y el servidor en vivo.
"""
import json
import math
import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from context import add_context, describe
from indicators import add_indicators
from setups import SetupDetector, SetupParams

FILTER_SYSTEM = """Eres un trader profesional de futuros del Nasdaq (NQ/MNQ), disciplinado y selectivo.
Un sistema de reglas ha detectado un SETUP con entrada, stop y objetivo ya definidos.
Tu trabajo es decidir si se TOMA o se SALTA según el contexto. No cambies stop ni objetivo.
Criterios que usan los buenos traders:
- Mejor a favor de la tendencia de fondo y del lado correcto del VWAP; con cuidado si va contra ambos.
- Debe haber espacio: si hay un nivel clave (máx/mín del día anterior, de la noche, rango de apertura, VWAP)
  entre la entrada y el objetivo, muy cerca en contra, la probabilidad baja.
- Volumen relativo bajo y el mediodía suelen dar señales falsas; la apertura y la última hora, más movimiento.
- Volatilidad muy alta (ATR >1.8x lo normal) hace los stops menos fiables.
- Tras pérdidas en el día, sé más exigente.
Responde SOLO con JSON: {"take": true|false, "confidence": <0.0-1.0>, "reason": "<explicación breve en español>"}"""

LEVELS = (("máx. día anterior", "pdh"), ("mín. día anterior", "pdl"), ("cierre día anterior", "pdc"),
          ("máx. noche", "onh"), ("mín. noche", "onl"), ("máx. rango apertura", "orh"),
          ("mín. rango apertura", "orl"), ("VWAP", "vwap"), ("máx. del día", "day_high"), ("mín. del día", "day_low"))


def levels_in_path(row: pd.Series, side: str, target_pts: float):
    """Niveles clave entre la entrada y un poco más allá del objetivo, en la dirección del trade."""
    c, out = row["close"], []
    for name, col in LEVELS:
        lvl = row.get(col)
        if lvl is None or pd.isna(lvl):
            continue
        dist = (lvl - c) if side == "LONG" else (c - lvl)
        if 0.5 < dist <= target_pts * 1.25:
            out.append((dist, name, lvl))
    return sorted(out)


def build_filter_prompt(row: pd.Series, cand: Dict, state: Dict) -> str:
    side_es = "COMPRA (largo)" if cand["side"] == "LONG" else "VENTA (corto)"
    pv = state.get("point_value", 2.0)
    path = levels_in_path(row, cand["side"], cand["target_pts"])
    path_txt = "; ".join(f"{n} {l:.2f} a {d:.1f} pts" for d, n, l in path) or "ninguno: camino libre hasta el objetivo"
    return f"""SETUP DETECTADO: {cand['setup']} — {side_es}
Motivo: {cand['why']}
Entrada ≈ {row['close']:.2f}. Stop a {cand['stop_pts']:.1f} pts ({cand['stop_pts'] * pv:.0f} $ por contrato). Objetivo a {cand['target_pts']:.1f} pts ({cand['target_pts'] * pv:.0f} $).
Niveles clave en el camino hacia el objetivo: {path_txt}.

CONTEXTO DE MERCADO
{describe(row)}

ESTADO DE LA CUENTA HOY
Operaciones: {state.get('trades_today', 0)}. PnL del día: {state.get('daily_pnl', 0):+.2f} $.

¿Tomas este trade?"""


def ask_llm(system: str, user: str) -> str:
    import server
    resp = requests.post(f"{server.OLLAMA_URL}/api/chat", timeout=server.OLLAMA_TIMEOUT, json={
        "model": server.MODEL, "stream": False, "format": "json",
        "options": {"temperature": server.TEMPERATURE},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_verdict(raw: str) -> Dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"take": False, "confidence": 0.0, "reason": "Respuesta de la IA no es JSON válido"}
    take = data.get("take")
    if isinstance(take, str):
        take = take.strip().lower() in ("true", "si", "sí", "yes")
    try:
        conf = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"take": bool(take), "confidence": conf, "reason": str(data.get("reason", ""))[:300]}


def llm_verdict(row: pd.Series, cand: Dict, state: Dict) -> Dict:
    try:
        return parse_verdict(ask_llm(FILTER_SYSTEM, build_filter_prompt(row, cand, state)))
    except Exception as e:
        # Ante cualquier fallo de la IA: NO se opera
        return {"take": False, "confidence": 0.0, "reason": f"Error de la IA: {e}", "error": True}


def make_llm_filter(cache_dir: Optional[Path] = None):
    """Filtro con caché en disco para el backtest (repetir/reanudar sin volver a preguntar)."""
    import server
    cache: Dict[str, Dict] = {}
    cache_file = (cache_dir / f"filter_{server.MODEL.replace(':', '_').replace('/', '_')}.jsonl") if cache_dir else None
    if cache_file and cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                cache[e["key"]] = e["verdict"]
            except (json.JSONDecodeError, KeyError):
                pass
    lock = threading.Lock()

    def filt(ctx: Dict, cand: Dict) -> Dict:
        key = f"{server.MODEL}|{ctx['time']}|{cand['setup']}|{cand['side']}"
        if key in cache:
            return cache[key]
        v = llm_verdict(ctx["row"], cand, {"trades_today": ctx["trades_today"], "daily_pnl": ctx["daily_pnl"],
                                           "point_value": ctx["point_value"]})
        if not v.get("error"):
            with lock:
                cache[key] = v
                if cache_file:
                    with cache_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "verdict": v}, ensure_ascii=False) + "\n")
        return v
    return filt


# ---------------------------- En vivo ----------------------------

def frame_from_bars(bars) -> pd.DataFrame:
    df = pd.DataFrame([{"time": b.t, "open": b.o, "high": b.h, "low": b.l, "close": b.c, "volume": b.v} for b in bars])
    df["time"] = pd.to_datetime(df["time"])
    return add_context(add_indicators(df.set_index("time")))


def live_decision(df: pd.DataFrame, position: str, tick_size: float, point_value: float,
                  start_time: int, end_time: int, trades_today: int, daily_pnl: float,
                  use_llm: bool, params: Optional[SetupParams] = None) -> Dict:
    """Reproduce el detector sobre las velas de hoy (así un reinicio del servidor no
    vuelve a disparar setups ya usados) y decide sobre la última vela."""
    last = df.iloc[-1]
    det = SetupDetector(params)
    in_window = lambda t: start_time <= t.hour * 10000 + t.minute * 100 + t.second <= end_time
    today = df[df["trade_date"] == last["trade_date"]]
    first = df.index.get_loc(today.index[0])
    for i in range(max(first, 1), len(df) - 1):
        if in_window(df.index[i]):
            det.detect(df.iloc[i], df.iloc[i - 1])
    cands = det.detect(last, df.iloc[-2])
    if position != "FLAT":
        return {"action": "HOLD", "confidence": 0.0, "reason": "En posición: gestionan el stop y el objetivo"}
    if not cands:
        return {"action": "HOLD", "confidence": 0.0, "reason": "Sin setup del playbook en esta vela"}
    c = cands[0]
    stop_ticks = int(math.ceil(c["stop_pts"] / tick_size))
    d = {"action": "BUY" if c["side"] == "LONG" else "SELL", "confidence": 1.0, "setup": c["setup"],
         "reason": f"[{c['setup']}] {c['why']}", "stop_ticks": stop_ticks,
         "target_ticks": int(round(c["target_pts"] / tick_size))}
    if use_llm:
        v = llm_verdict(last, c, {"trades_today": trades_today, "daily_pnl": daily_pnl, "point_value": point_value})
        d["confidence"], d["reason"] = v["confidence"], f"[{c['setup']}] {'TOMAR' if v['take'] else 'SALTAR'}: {v['reason']}"
        if not v["take"]:
            d["action"] = "HOLD"
    return d

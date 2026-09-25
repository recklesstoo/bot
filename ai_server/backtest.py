"""
Backtest del bot con datos históricos exportados de NinjaTrader.

Reproduce las reglas de AIFuturesTrader.cs:
  - decisión al cierre de cada vela de 5 min dentro del horario (hora de NY)
  - entrada a mercado en la apertura del minuto siguiente (+ deslizamiento)
  - stop/target por ATR, acotados en ticks; se revisan minuto a minuto
    (si stop y target caen en el mismo minuto, se asume el stop: conservador)
  - confianza mínima, máximo de trades, pérdida diaria máxima y bloqueo de
    entradas cuyo stop superaría el margen restante del día
  - cierre fuera de horario

Uso por línea de comandos:
  python backtest.py datos.txt --strategy ema
  python backtest.py datos.txt --strategy qwen --days 10
"""
import argparse
import json
import math
import random
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from context import add_context
from indicators import add_indicators

NY_TZ = "America/New_York"


# ======
# Datos
# ======
def load_nt_export(path, source_tz: str = "UTC") -> pd.DataFrame:
    """Lee un export de NinjaTrader ('yyyyMMdd HHmmss;O;H;L;C;V', hora de CIERRE de la vela).
    Devuelve velas con índice en hora de Nueva York (sin zona)."""
    df = pd.read_csv(path, sep=";", header=None, names=["time", "open", "high", "low", "close", "volume"],
                     dtype={"time": str})
    df["time"] = pd.to_datetime(df["time"].str.strip(), format="%Y%m%d %H%M%S")
    df = df.dropna().drop_duplicates("time").sort_values("time")
    idx = pd.DatetimeIndex(df["time"]).tz_localize(source_tz).tz_convert(NY_TZ).tz_localize(None)
    df = df.drop(columns="time").set_index(idx)
    df.index.name = "time"
    return df.astype(float)


def merge_contracts(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Une varios contratos (p. ej. 06-26 y 09-26) en una serie continua.
    Cambia de contrato el primer día de sesión en que el nuevo tiene más volumen
    y ajusta los precios anteriores por la diferencia entre contratos (back-adjust)."""
    frames = sorted((f for f in frames if len(f)), key=lambda f: f.index[0])
    out = frames[0]
    for nxt in frames[1:]:
        sess = lambda d: (d.index + pd.Timedelta(hours=6)).normalize()
        v_old = out["volume"].groupby(sess(out)).sum()
        v_new = nxt["volume"].groupby(sess(nxt)).sum()
        both = v_old.index.intersection(v_new.index)
        roll_days = [d for d in both if v_new[d] > v_old[d]]
        roll = roll_days[0] if roll_days else (both[-1] if len(both) else sess(nxt)[0])
        roll_ts = roll - pd.Timedelta(hours=6)                 # 18:00 NY del día anterior
        common = out.index.intersection(nxt.index)
        before = common[common < roll_ts]
        # diferencia entre contratos justo antes del cambio (o, si no se solapan antes, al empezar a solaparse)
        ref = before[-60:] if len(before) else common[:60]
        offset = float((nxt.loc[ref, "close"] - out.loc[ref, "close"]).median()) if len(ref) else 0.0
        old = out[out.index < roll_ts].copy()
        old[["open", "high", "low", "close"]] += offset
        out = pd.concat([old, nxt[nxt.index >= roll_ts]])
        out.attrs["rolls"] = out.attrs.get("rolls", []) + [{"date": str(roll.date()), "offset": round(offset, 2)}]
    return out


def resample(df1: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    # Etiqueta = hora de cierre, igual que NinjaTrader
    r = df1.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return r.dropna(subset=["open"])


# ==========
# Parámetros
# ==========
@dataclass
class Params:
    tick_size: float = 0.25
    point_value: float = 2.0          # MNQ
    quantity: int = 1
    bars_to_send: int = 100
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 3.0
    min_stop_ticks: int = 40
    max_stop_ticks: int = 160
    max_daily_loss: float = 200.0
    max_trades_per_day: int = 4
    min_confidence: float = 0.65
    allow_shorts: bool = True
    allow_reversal: bool = False
    start_time: int = 93500           # HHmmss, hora de NY (cierre de vela)
    end_time: int = 154500
    commission_per_side: float = 0.62  # por contrato (comisión + tasas aprox.)
    slippage_ticks: int = 1           # en entradas/salidas a mercado y stops
    timeframe_min: int = 5


@dataclass
class Trade:
    side: str
    entry_time: str
    entry_price: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    reason: str = ""
    setup: str = ""


# ===========
# Estrategias
# ===========
# Una estrategia recibe el contexto de la vela y devuelve {"action","confidence","reason"}.
Strategy = Callable[[dict], dict]


def ema_strategy(ctx: dict) -> dict:
    """Referencia simple: cruce EMA9/EMA21 a favor de la EMA50."""
    cur, prev = ctx["row"], ctx["prev"]
    if prev["ema9"] <= prev["ema21"] and cur["ema9"] > cur["ema21"] and cur["close"] > cur["ema50"]:
        return {"action": "BUY", "confidence": 1.0, "reason": "Cruce EMA9>EMA21 sobre EMA50"}
    if prev["ema9"] >= prev["ema21"] and cur["ema9"] < cur["ema21"] and cur["close"] < cur["ema50"]:
        return {"action": "SELL", "confidence": 1.0, "reason": "Cruce EMA9<EMA21 bajo EMA50"}
    return {"action": "HOLD", "confidence": 0.0, "reason": ""}


def make_random_strategy(seed: int = 42, p_entry: float = 0.06) -> Strategy:
    """Referencia de azar: entra al azar ~6% de las velas. Si la IA no supera esto, no aporta."""
    rng = random.Random(seed)

    def strat(ctx: dict) -> dict:
        x = rng.random()
        if x < p_entry / 2:
            return {"action": "BUY", "confidence": 1.0, "reason": "azar"}
        if x < p_entry:
            return {"action": "SELL", "confidence": 1.0, "reason": "azar"}
        return {"action": "HOLD", "confidence": 0.0, "reason": ""}
    return strat


def make_llm_strategy(cache_file: Optional[Path] = None) -> Strategy:
    """Usa exactamente el mismo prompt y validación que el servidor en vivo.
    Las respuestas se guardan en caché para poder repetir/reanudar sin volver a preguntar."""
    import server  # import diferido: evita dependencia circular y carga de FastAPI en CLI

    cache: Dict[str, dict] = {}
    if cache_file and cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                cache[e["key"]] = e["decision"]
            except (json.JSONDecodeError, KeyError):
                pass
    lock = threading.Lock()

    def strat(ctx: dict) -> dict:
        key = f"{server.MODEL}|{ctx['time']}|{ctx['position']}"
        if key in cache:
            return cache[key]
        bars = [server.Bar(t=t.strftime("%Y-%m-%dT%H:%M:%S"), o=r.open, h=r.high, l=r.low, c=r.close, v=r.volume)
                for t, r in ctx["window"].iterrows()]
        req = server.DecideRequest(instrument="MNQ", timeframe=f"{ctx['timeframe']} Minute",
                                   tick_size=ctx["tick_size"], point_value=ctx["point_value"],
                                   position=ctx["position"], position_qty=ctx["qty"], avg_price=ctx["avg_price"],
                                   unrealized_pnl=ctx["unrealized"], daily_pnl=ctx["daily_pnl"],
                                   trades_today=ctx["trades_today"], allow_shorts=ctx["allow_shorts"], bars=bars)
        decision, _, raw = server.run_decision(req)
        d = decision.model_dump()
        # Errores de conexión no se cachean: se reintentan en la próxima ejecución
        if not d["reason"].startswith(("Error llamando a Ollama", "Error inesperado")):
            with lock:
                cache[key] = d
                if cache_file:
                    with cache_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "decision": d}, ensure_ascii=False) + "\n")
        return d
    return strat


# =====
# Motor
# =====
class Backtester:
    def __init__(self, df1: pd.DataFrame, params: Params, strategy: Strategy,
                 progress: Optional[Callable[[int, int], None]] = None,
                 cancel: Optional[threading.Event] = None):
        self.p = params
        self.df1 = df1
        self.df5 = add_context(add_indicators(resample(df1, params.timeframe_min)))
        self.strategy = strategy
        self.trade_from = df1.attrs.get("trade_from")
        self.progress = progress
        self.cancel = cancel or threading.Event()
        self._reset_day(None)
        self.trades: List[Trade] = []
        self.pos: Optional[Trade] = None
        self.realized = 0.0
        self.equity: List[tuple] = []
        self.decisions: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0, "EXIT": 0}
        self.blocked: Dict[str, int] = {"confianza": 0, "max_trades": 0, "riesgo_diario": 0, "dia_detenido": 0, "stop_grande": 0}
        self._pending: dict = {}
        self.day_stats: Dict[str, dict] = {}

    # --- utilidades ---
    def _reset_day(self, day):
        self.day = day
        self.day_start_realized = getattr(self, "realized", 0.0)
        self.trades_today = 0
        self.halted = False

    def _hhmmss(self, t: pd.Timestamp) -> int:
        return t.hour * 10000 + t.minute * 100 + t.second

    def _in_hours(self, t) -> bool:
        return self.p.start_time <= self._hhmmss(t) <= self.p.end_time

    def _unrealized(self, price: float) -> float:
        if not self.pos:
            return 0.0
        d = price - self.pos.entry_price if self.pos.side == "LONG" else self.pos.entry_price - price
        return d * self.p.point_value * self.p.quantity

    def _daily_pnl(self, price: float) -> float:
        return self.realized - self.day_start_realized + self._unrealized(price)

    def _stop_ticks(self, atr: float, stop_pts: Optional[float] = None) -> int:
        if stop_pts:   # stop estructural del setup: nunca más ajustado que el mínimo
            return max(self.p.min_stop_ticks, int(math.ceil(stop_pts / self.p.tick_size)))
        ticks = int(round(atr / self.p.tick_size * self.p.stop_atr_mult)) if atr > 0 else self.p.min_stop_ticks
        return min(self.p.max_stop_ticks, max(self.p.min_stop_ticks, ticks))

    def _slip(self, price: float, side_buy: bool) -> float:
        s = self.p.slippage_ticks * self.p.tick_size
        return price + s if side_buy else price - s

    # --- órdenes ---
    def _open(self, side: str, t, open_price: float, atr: float, reason: str, pending: Optional[dict] = None):
        pending = pending or {}
        stop_pts, target_pts = pending.get("stop_pts"), pending.get("target_pts")
        stop_ticks = self._stop_ticks(atr, stop_pts)
        if stop_pts and target_pts:   # mantiene la relación riesgo/beneficio del setup
            target_ticks = int(round(target_pts / stop_pts * stop_ticks))
        else:
            target_ticks = max(stop_ticks, int(round(atr / self.p.tick_size * self.p.target_atr_mult)))
        fill = self._slip(open_price, side == "LONG")
        sd, td = stop_ticks * self.p.tick_size, target_ticks * self.p.tick_size
        self.pos = Trade(side=side, entry_time=str(t), entry_price=fill, reason=reason, setup=pending.get("setup", ""),
                         stop=fill - sd if side == "LONG" else fill + sd,
                         target=fill + td if side == "LONG" else fill - td)
        self.trades_today += 1

    def _close(self, t, price: float, why: str):
        tr = self.pos
        tr.exit_time, tr.exit_price, tr.exit_reason = str(t), price, why
        d = price - tr.entry_price if tr.side == "LONG" else tr.entry_price - price
        tr.pnl = round(d * self.p.point_value * self.p.quantity - 2 * self.p.commission_per_side * self.p.quantity, 2)
        self.realized += tr.pnl
        self.trades.append(tr)
        self.pos = None
        self.equity.append((str(t), round(self.realized, 2)))

    def _check_bracket(self, t, bar) -> bool:
        """Revisa stop/target en un minuto. True si cerró."""
        tr = self.pos
        if tr.side == "LONG":
            if bar.low <= tr.stop:
                self._close(t, self._slip(min(bar.open, tr.stop), False), "Stop")
                return True
            if bar.high >= tr.target:
                self._close(t, max(bar.open, tr.target), "Target")
                return True
        else:
            if bar.high >= tr.stop:
                self._close(t, self._slip(max(bar.open, tr.stop), True), "Stop")
                return True
            if bar.low <= tr.target:
                self._close(t, min(bar.open, tr.target), "Target")
                return True
        return False

    def _can_open(self, price: float, atr: float, stop_pts: Optional[float] = None) -> Optional[str]:
        if self.trades_today >= self.p.max_trades_per_day:
            return "max_trades"
        ticks = self._stop_ticks(atr, stop_pts)
        if stop_pts and ticks > self.p.max_stop_ticks:
            return "stop_grande"
        risk = ticks * self.p.tick_size * self.p.point_value * self.p.quantity
        if risk > self.p.max_daily_loss + self._daily_pnl(price):
            return "riesgo_diario"
        return None

    def _plan(self, action: str, price: float, atr: float, stop_pts: Optional[float] = None) -> Optional[str]:
        """Traduce la decisión a una orden, igual que Execute() en NinjaTrader."""
        side = self.pos.side if self.pos else "FLAT"
        if action == "BUY":
            if side == "LONG":
                return None
            if side == "SHORT" and not self.p.allow_reversal:
                return "EXIT"
        elif action == "SELL":
            if side == "SHORT":
                return None
            if side == "LONG" and (not self.p.allow_reversal or not self.p.allow_shorts):
                return "EXIT"
            if not self.p.allow_shorts:
                return None
        elif action == "EXIT":
            return "EXIT" if self.pos else None
        else:
            return None
        why = self._can_open(price, atr, stop_pts)
        if why:
            self.blocked[why] += 1
            return None
        return "REVERSE_" + action if self.pos else action

    # --- bucle principal ---
    def run(self) -> dict:
        p, df5 = self.p, self.df5
        m1_times = self.df1.index.values
        warm = max(p.bars_to_send, 60)
        idx = df5.index
        tradable = lambda t: self._in_hours(t) and (self.trade_from is None or t >= self.trade_from)
        total = int(sum(1 for t in idx[warm:] if tradable(t)))
        done = 0

        for i in range(warm, len(df5) - 1):
            if self.cancel.is_set():
                break
            t, row = idx[i], df5.iloc[i]
            day = t.date()
            if day != self.day:
                self._reset_day(day)

            price, atr = float(row["close"]), float(row["atr14"]) if not math.isnan(row["atr14"]) else 0.0
            order = None

            if not self.halted and self._daily_pnl(price) <= -p.max_daily_loss:
                self.halted = True
                if self.pos:
                    order = "EXIT"
            if order is None and not tradable(t):
                if self.pos:
                    order = "EXIT"
            elif order is None and not self.halted:
                ctx = {"time": str(t), "row": row, "prev": df5.iloc[i - 1], "window": df5.iloc[i - p.bars_to_send + 1:i + 1],
                       "position": self.pos.side if self.pos else "FLAT", "qty": p.quantity if self.pos else 0,
                       "avg_price": self.pos.entry_price if self.pos else 0.0, "unrealized": self._unrealized(price),
                       "daily_pnl": self._daily_pnl(price), "trades_today": self.trades_today,
                       "allow_shorts": p.allow_shorts, "tick_size": p.tick_size, "point_value": p.point_value,
                       "timeframe": p.timeframe_min}
                d = self.strategy(ctx)
                action = str(d.get("action", "HOLD")).upper()
                self.decisions[action] = self.decisions.get(action, 0) + 1
                if action != "EXIT" and float(d.get("confidence", 0)) < p.min_confidence:
                    if action in ("BUY", "SELL"):
                        self.blocked["confianza"] += 1
                else:
                    order = self._plan(action, price, atr, d.get("stop_pts"))
                    self._reason = d.get("reason", "")
                    self._pending = d
                done += 1
                if self.progress and done % 5 == 0:
                    self.progress(done, total)
            elif order is None and self.halted and tradable(t):
                self.blocked["dia_detenido"] += 1

            # Fin de sesión (cierre diario, fin de semana o festivo con cierre anticipado):
            # NinjaTrader cierra la posición antes del cierre (IsExitOnSessionCloseStrategy).
            t_next = idx[i + 1]
            if t_next - t > pd.Timedelta(minutes=30):
                if self.pos:
                    self._close(t, self._slip(price, self.pos.side == "SHORT"), "Cierre de sesión")
                order = None

            # Minutos de la vela siguiente: ejecución en la apertura del primero y revisión del bracket
            lo, hi = np.searchsorted(m1_times, t.to_datetime64(), side="right"), np.searchsorted(m1_times, t_next.to_datetime64(), side="right")
            minutes = self.df1.iloc[lo:hi]
            for j, (mt, mb) in enumerate(minutes.iterrows()):
                if j == 0 and order:
                    if order == "EXIT" or order.startswith("REVERSE_"):
                        if self.pos:
                            self._close(mt, self._slip(mb.open, self.pos.side == "SHORT"), "Señal/regla")
                    if order in ("BUY", "REVERSE_BUY"):
                        self._open("LONG", mt, mb.open, atr, getattr(self, "_reason", ""), self._pending)
                    elif order in ("SELL", "REVERSE_SELL"):
                        self._open("SHORT", mt, mb.open, atr, getattr(self, "_reason", ""), self._pending)
                if self.pos:
                    self._check_bracket(mt, mb)

        if self.pos:
            last_t, last = self.df1.index[-1], self.df1.iloc[-1]
            self._close(last_t, float(last.close), "Fin de datos")
        if self.progress:
            self.progress(total, total)
        return self.report()

    # --- resultados ---
    def _stats(self, trades: List[Trade]) -> dict:
        pnls = [t.pnl for t in trades]
        wins, losses = [x for x in pnls if x > 0], [x for x in pnls if x <= 0]
        eq = np.concatenate([[0.0], np.cumsum(pnls)]) if pnls else np.array([0.0])
        max_dd = float((eq - np.maximum.accumulate(eq)).min())
        by_day: Dict[str, float] = {}
        for t in trades:
            by_day[t.exit_time[:10]] = by_day.get(t.exit_time[:10], 0.0) + t.pnl
        days = list(by_day.values())
        gp, gl = sum(wins), -sum(losses)
        pf = round(gp / gl, 2) if gl > 0 else None
        return {
            "trades": len(pnls),
            "net_pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(pnls), 3) if pnls else 0.0,
            "profit_factor": pf,
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
            "expectancy": round(float(np.mean(pnls)), 2) if pnls else 0.0,
            "max_drawdown": round(max_dd, 2),
            "days_traded": len(days),
            "green_days": sum(1 for d in days if d > 0),
            "red_days": sum(1 for d in days if d <= 0),
            "best_day": round(max(days), 2) if days else 0.0,
            "worst_day": round(min(days), 2) if days else 0.0,
            "long_trades": sum(1 for t in trades if t.side == "LONG"),
            "short_trades": sum(1 for t in trades if t.side == "SHORT"),
            "stops": sum(1 for t in trades if t.exit_reason == "Stop"),
            "targets": sum(1 for t in trades if t.exit_reason == "Target"),
            "commissions": round(len(pnls) * 2 * self.p.commission_per_side * self.p.quantity, 2),
        }

    def report(self, design_frac: float = 0.6) -> dict:
        stats = self._stats(self.trades)
        start = self.trade_from if self.trade_from is not None else self.df5.index[0]
        stats.update(decisions=self.decisions, blocked=self.blocked, period=[str(start), str(self.df5.index[-1])])

        # Diseño vs validación: los primeros X% de días sirven para diseñar/ajustar,
        # el resto es "fuera de muestra". Solo el tramo de validación dice si hay edge.
        days = sorted({d for d in self.df5.index[self.df5.index >= start].normalize()})
        split = str(days[int(len(days) * design_frac)].date()) if len(days) > 4 else None
        segments = {}
        if split:
            segments = {"split_date": split,
                        "design": self._stats([t for t in self.trades if t.entry_time[:10] < split]),
                        "validation": self._stats([t for t in self.trades if t.entry_time[:10] >= split])}
        by_setup = {}
        for name in sorted({t.setup for t in self.trades if t.setup}):
            tr = [t for t in self.trades if t.setup == name]
            by_setup[name] = {"all": self._stats(tr)}
            if split:
                by_setup[name]["design"] = self._stats([t for t in tr if t.entry_time[:10] < split])
                by_setup[name]["validation"] = self._stats([t for t in tr if t.entry_time[:10] >= split])
        by_day: Dict[str, float] = {}
        for t in self.trades:
            by_day[t.exit_time[:10]] = round(by_day.get(t.exit_time[:10], 0.0) + t.pnl, 2)
        return {"stats": stats, "segments": segments, "by_setup": by_setup, "equity": self.equity, "daily": by_day,
                "trades": [asdict(t) for t in self.trades], "params": asdict(self.p),
                "cancelled": self.cancel.is_set()}


def slice_days(df1: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    """Últimos N días de sesión, más colchón previo para calentar indicadores."""
    if not days:
        return df1
    dates = sorted(set(df1.index.date))
    if days >= len(dates):
        return df1
    first = pd.Timestamp(dates[-days])
    out = df1[df1.index >= first - pd.Timedelta(days=4)].copy()
    out.attrs["trade_from"] = first      # antes de esta fecha solo se calientan indicadores
    return out


def make_setups_strategy(setup_params=None, llm_filter: Optional[Callable[[dict, dict], dict]] = None) -> Strategy:
    """Playbook: las reglas detectan el setup (con su stop estructural); opcionalmente
    la IA decide si tomarlo o saltarlo según el contexto."""
    from setups import SetupDetector
    det = SetupDetector(setup_params)

    def strat(ctx: dict) -> dict:
        cands = det.detect(ctx["row"], ctx["prev"])      # se llama siempre: lleva la cuenta del día
        if ctx["position"] != "FLAT" or not cands:
            return {"action": "HOLD", "confidence": 0.0, "reason": ""}
        c = cands[0]
        d = {"action": "BUY" if c["side"] == "LONG" else "SELL", "confidence": 1.0,
             "reason": f"[{c['setup']}] {c['why']}", "setup": c["setup"],
             "stop_pts": c["stop_pts"], "target_pts": c["target_pts"]}
        if llm_filter:
            v = llm_filter(ctx, c)
            d["confidence"] = float(v.get("confidence", 0.0))
            d["reason"] = f"[{c['setup']}] {v.get('reason', '')}"
            if not v.get("take", False):
                d["action"] = "HOLD"
        return d
    return strat


def build_strategy(name: str, cache_dir: Optional[Path] = None) -> Strategy:
    name = name.lower()
    if name == "setups":
        return make_setups_strategy()
    if name == "setups_ia":
        import playbook
        return make_setups_strategy(llm_filter=playbook.make_llm_filter(cache_dir))
    if name == "ema":
        return ema_strategy
    if name == "random":
        return make_random_strategy()
    if name in ("qwen", "ia", "llm"):
        import server
        cache = (cache_dir / f"cache_{server.MODEL.replace(':', '_').replace('/', '_')}.jsonl") if cache_dir else None
        return make_llm_strategy(cache)
    raise ValueError(f"Estrategia desconocida: {name}")


def main():
    ap = argparse.ArgumentParser(description="Backtest del bot con datos de NinjaTrader")
    ap.add_argument("data")
    ap.add_argument("--strategy", default="ema", choices=["ema", "random", "qwen", "setups", "setups_ia"])
    ap.add_argument("--days", type=int, default=0, help="Solo los últimos N días")
    ap.add_argument("--tz", default="UTC", help="Zona horaria del archivo (NinjaTrader exporta en UTC)")
    ap.add_argument("--min-confidence", type=float, default=Params.min_confidence)
    ap.add_argument("--out", help="Guardar resultado completo en JSON")
    a = ap.parse_args()

    df1 = slice_days(load_nt_export(a.data, a.tz), a.days)
    params = Params(min_confidence=a.min_confidence)
    last = [0]

    def prog(d, t):
        pct = int(d * 100 / max(t, 1))
        if pct >= last[0] + 5:
            last[0] = pct
            print(f"  {pct}% ({d}/{t})", flush=True)
    res = Backtester(df1, params, build_strategy(a.strategy, Path("data")), prog).run()
    print(json.dumps(res["stats"], indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

"""
Playbook de setups intradía para NQ/MNQ (hora de NY, velas de 5 min con contexto).

Cada setup devuelve un candidato con stop ESTRUCTURAL (donde la idea deja de
ser válida), no una distancia fija de ATR:
  {"setup", "side": "LONG"/"SHORT", "stop_pts", "target_pts", "why"}

  ORB    Ruptura del rango de apertura (9:30–9:45) con cierre fuera del rango.
         Stop: extremo de la vela de ruptura. Objetivo: 2R.
  VWAP   Retroceso al VWAP a favor de la tendencia del día (≥75% de la última
         hora del lado correcto) con vela de rechazo. Stop: tras la mecha. 2R.
  SWEEP  Barrido fallido de un nivel clave (máx/mín del día anterior o de la
         noche): la vela lo supera y cierra de vuelta dentro. Stop: tras el
         extremo del barrido. Objetivo: 2R.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

TICK = 0.25


@dataclass
class SetupParams:
    rr: float = 2.0                 # objetivo en múltiplos del riesgo
    buffer_ticks: int = 8           # margen más allá del extremo para el stop (elegido en el tramo de diseño)
    orb_until: int = 1130           # ORB solo hasta esta hora (HHMM)
    vwap_from: int = 1000
    vwap_until: int = 1500
    vwap_trend_frac: float = 0.75
    sweep_from: int = 945
    sweep_until: int = 1500
    # VWAP queda desactivado: fue negativo en el tramo de diseño (may–jul 2026)
    enabled: tuple = ("ORB", "SWEEP")


class SetupDetector:
    """Guarda qué setups ya se dispararon hoy (uno por setup/dirección/nivel y día)."""

    def __init__(self, params: Optional[SetupParams] = None):
        self.p = params or SetupParams()
        self.day = None
        self.fired: set = set()

    def _once(self, key: str) -> bool:
        if key in self.fired:
            return False
        self.fired.add(key)
        return True

    def detect(self, row: pd.Series, prev: pd.Series) -> List[Dict]:
        if not row.get("rth", False):
            return []
        if row["trade_date"] != self.day:
            self.day, self.fired = row["trade_date"], set()
        hm = row.name.hour * 100 + row.name.minute
        p, buf, out = self.p, self.p.buffer_ticks * TICK, []
        c, h, l, o = row["close"], row["high"], row["low"], row["open"]

        def cand(setup, side, stop_price, why):
            stop_pts = (c - stop_price) if side == "LONG" else (stop_price - c)
            if stop_pts <= 0:
                return
            out.append({"setup": setup, "side": side, "stop_pts": round(stop_pts, 2),
                        "target_pts": round(stop_pts * p.rr, 2), "why": why})

        # --- ORB ---
        if "ORB" in p.enabled and not pd.isna(row["orh"]) and 945 < hm <= p.orb_until:
            if c > row["orh"] >= prev["close"] and self._once("ORB_L"):
                cand("ORB", "LONG", l - buf, f"Cierre sobre el máximo del rango de apertura ({row['orh']:.2f})")
            elif c < row["orl"] <= prev["close"] and self._once("ORB_S"):
                cand("ORB", "SHORT", h + buf, f"Cierre bajo el mínimo del rango de apertura ({row['orl']:.2f})")

        # --- VWAP pullback ---
        if "VWAP" in p.enabled and not pd.isna(row["vwap"]) and p.vwap_from <= hm <= p.vwap_until:
            vw, frac = row["vwap"], row["above_vwap_frac"]
            if frac >= p.vwap_trend_frac and l <= vw < c and c > o:
                cand("VWAP", "LONG", min(l, vw) - buf, f"Retroceso al VWAP ({vw:.2f}) en día alcista, vela de rechazo")
            elif frac <= 1 - p.vwap_trend_frac and h >= vw > c and c < o:
                cand("VWAP", "SHORT", max(h, vw) + buf, f"Retroceso al VWAP ({vw:.2f}) en día bajista, vela de rechazo")

        # --- Sweep de niveles clave ---
        if "SWEEP" in p.enabled and p.sweep_from <= hm <= p.sweep_until:
            for name, lvl, side in (("máx. día anterior", row["pdh"], "SHORT"), ("máx. de la noche", row["onh"], "SHORT"),
                                    ("mín. día anterior", row["pdl"], "LONG"), ("mín. de la noche", row["onl"], "LONG")):
                if pd.isna(lvl):
                    continue
                if side == "SHORT" and h > lvl > c and self._once(f"SW_{name}"):
                    cand("SWEEP", "SHORT", h + buf, f"Barrido fallido del {name} ({lvl:.2f}): superó y cerró debajo")
                elif side == "LONG" and l < lvl < c and self._once(f"SW_{name}"):
                    cand("SWEEP", "LONG", l - buf, f"Barrido fallido del {name} ({lvl:.2f}): perforó y cerró encima")
        return out

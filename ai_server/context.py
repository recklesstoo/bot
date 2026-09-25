"""
Contexto de mercado "como lo ve un trader", calculado sobre velas de 5 min con
índice en hora de Nueva York (hora de CIERRE de la vela, como NinjaTrader).

Se usa igual en el backtest y en vivo, para que la IA vea exactamente lo mismo.

Columnas que añade (por vela):
  trade_date         fecha de sesión (la sesión de CME empieza a las 18:00 del día anterior)
  rth                True si la vela es de la sesión regular (9:30–16:00)
  pdh, pdl, pdc      máximo / mínimo / cierre de la sesión regular anterior
  onh, onl           máximo / mínimo de la noche (18:00–9:30)
  orh, orl           rango de apertura (9:30–9:45); NaN antes de las 9:45
  day_high, day_low  máximo / mínimo de la sesión regular hasta esta vela
  vwap, vwap_sd      VWAP de la sesión regular y su desviación
  trend              +1 alcista / -1 bajista / 0 lateral (pendiente de la EMA de ~5 h en ATRs)
  above_vwap_frac    fracción de las últimas 12 velas que cerraron sobre el VWAP
  atr_ratio          ATR actual / ATR típico (mediana de ~5 días)
  rvol               volumen / volumen medio a esta misma hora (5 sesiones)
  tod                franja horaria (apertura, media mañana, mediodía, tarde, última hora)
"""
import numpy as np
import pandas as pd

from indicators import add_indicators, ema

RTH_OPEN, RTH_CLOSE = 930, 1600


def _hhmm(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour * 100 + idx.minute


def time_bucket(hhmm: int) -> str:
    if hhmm <= 1030:
        return "apertura"
    if hhmm <= 1130:
        return "media mañana"
    if hhmm <= 1330:
        return "mediodía"
    if hhmm <= 1500:
        return "tarde"
    return "última hora"


def add_context(df5: pd.DataFrame) -> pd.DataFrame:
    df = df5 if "atr14" in df5.columns else add_indicators(df5)
    df = df.copy()
    idx = df.index
    hm = _hhmm(idx)
    df["trade_date"] = (idx + pd.Timedelta(hours=6)).normalize()      # 18:00 -> día siguiente
    df["rth"] = (hm > RTH_OPEN) & (hm <= RTH_CLOSE)
    df["overnight"] = (hm > 1800) | (hm <= RTH_OPEN)

    # --- niveles del día anterior (sesión regular) ---
    rth = df[df["rth"]]
    daily = rth.groupby("trade_date").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    prev = daily.shift(1)
    df["pdh"] = df["trade_date"].map(prev["h"])
    df["pdl"] = df["trade_date"].map(prev["l"])
    df["pdc"] = df["trade_date"].map(prev["c"])

    # --- noche: 18:00 -> 9:30 (valor final visible durante la sesión regular) ---
    on = df[df["overnight"]]
    on_agg = on.groupby("trade_date").agg(h=("high", "max"), l=("low", "min"))
    df["onh"] = np.where(df["rth"], df["trade_date"].map(on_agg["h"]), np.nan)
    df["onl"] = np.where(df["rth"], df["trade_date"].map(on_agg["l"]), np.nan)

    # --- rango de apertura (velas que cierran 9:35, 9:40, 9:45) ---
    orb = df[(hm > RTH_OPEN) & (hm <= 945)]
    or_agg = orb.groupby("trade_date").agg(h=("high", "max"), l=("low", "min"), n=("close", "size"))
    or_agg = or_agg[or_agg["n"] >= 3]
    after_or = df["rth"] & (hm >= 945)
    df["orh"] = np.where(after_or, df["trade_date"].map(or_agg["h"]), np.nan)
    df["orl"] = np.where(after_or, df["trade_date"].map(or_agg["l"]), np.nan)

    # --- máximos/mínimos del día y VWAP (solo sesión regular) ---
    g = df[df["rth"]].groupby("trade_date")
    r = df[df["rth"]]
    tp = (r["high"] + r["low"] + r["close"]) / 3
    cum_v = g["volume"].cumsum()
    cum_pv = (tp * r["volume"]).groupby(r["trade_date"]).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    cum_pv2 = (tp * tp * r["volume"]).groupby(r["trade_date"]).cumsum()
    var = (cum_pv2 / cum_v.replace(0, np.nan) - vwap * vwap).clip(lower=0)
    df["vwap"] = vwap
    df["vwap_sd"] = np.sqrt(var)
    df["day_high"] = g["high"].cummax()
    df["day_low"] = g["low"].cummin()
    above = (r["close"] > vwap).astype(float)
    df["above_vwap_frac"] = above.groupby(r["trade_date"]).transform(lambda s: s.rolling(12, min_periods=3).mean())

    # --- régimen: pendiente de EMA(60 velas ≈ 5 h) en la última hora, medida en ATRs ---
    slow = ema(df["close"], 60)
    slope = (slow - slow.shift(12)) / df["atr14"].replace(0, np.nan)
    df["trend_slope"] = slope
    df["trend"] = np.select([slope > 0.5, slope < -0.5], [1, -1], 0)

    # --- volatilidad y volumen relativos ---
    # Ventanas de ~5 días: en vivo NinjaTrader envía ~1500 velas, así backtest y vivo calculan igual
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(1380, min_periods=500).median()
    tod_key = hm
    df["rvol"] = df["volume"] / df.groupby(tod_key)["volume"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df["tod"] = [time_bucket(x) for x in hm]
    return df


def _pts(x: float) -> str:
    return "n/d" if pd.isna(x) else f"{x:+.1f}"


def describe(row: pd.Series) -> str:
    """Resumen en lenguaje de trader de la vela actual (para el prompt de la IA)."""
    c = row["close"]
    trend = {1: "alcista", -1: "bajista", 0: "lateral"}[int(row["trend"])]
    lines = [
        f"Hora NY: {row.name:%H:%M} ({row['tod']}). Precio: {c:.2f}.",
        f"Tendencia de fondo (EMA ~5 h): {trend} (pendiente {row['trend_slope']:+.2f} ATR/h).",
        f"VWAP: {row['vwap']:.2f} (precio {_pts(c - row['vwap'])} pts; {row['above_vwap_frac']:.0%} de la última hora sobre VWAP)."
        if not pd.isna(row.get("vwap")) else "VWAP: aún no disponible.",
        f"Día anterior: máx {row['pdh']:.2f} ({_pts(c - row['pdh'])}), mín {row['pdl']:.2f} ({_pts(c - row['pdl'])}), cierre {row['pdc']:.2f} ({_pts(c - row['pdc'])})."
        if not pd.isna(row.get("pdh")) else "Día anterior: n/d.",
        f"Noche: máx {row['onh']:.2f} ({_pts(c - row['onh'])}), mín {row['onl']:.2f} ({_pts(c - row['onl'])})."
        if not pd.isna(row.get("onh")) else "Noche: n/d.",
        f"Rango de apertura (15 min): {row['orl']:.2f}–{row['orh']:.2f} ({row['orh'] - row['orl']:.1f} pts)."
        if not pd.isna(row.get("orh")) else "Rango de apertura: aún formándose.",
        f"Rango de hoy: {row['day_low']:.2f}–{row['day_high']:.2f}." if not pd.isna(row.get("day_high")) else "",
        f"Volatilidad: ATR(5m) {row['atr14']:.1f} pts = {row['atr_ratio']:.2f}x lo normal. Volumen relativo a esta hora: {row['rvol']:.2f}x."
        if not pd.isna(row.get("atr_ratio")) and not pd.isna(row.get("rvol")) else f"ATR(5m) {row['atr14']:.1f} pts.",
        f"RSI14 {row['rsi14']:.0f}, histograma MACD {row['macd_hist']:+.2f}.",
    ]
    return "\n".join(l for l in lines if l)

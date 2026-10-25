"""Servidor MCP de solo lectura para TradingView Desktop vía CDP (localhost).

Herramientas fijas: no hay evaluación arbitraria de JS ni clics, así que no
puede operar el broker. Requiere TradingView lanzado con --remote-debugging-port.
"""
import base64, json, os, subprocess, threading, time, urllib.request

from mcp.server.fastmcp import FastMCP, Image
from websockets.sync.client import connect

CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
mcp = FastMCP("tradingview")

# Expresiones JS fijas (solo lectura)
# Los selectores del DOM de TradingView pueden cambiar entre versiones (best effort).
_STATE_JS = """JSON.stringify({
  title: document.title,
  url: location.href,
  symbol: (document.querySelector('#header-toolbar-symbol-search') || {}).innerText || null,
  timeframe: (document.querySelector('[id*="header-toolbar-intervals"] button') || {}).innerText || null
})"""


LAUNCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_tradingview.sh")


def _cdp_up():
    try:
        urllib.request.urlopen(f"{CDP_URL}/json/version", timeout=2).close()
        return True
    except OSError:
        return False


def _ensure_running():
    """Si TradingView no responde en el puerto CDP, lo abre con launch_tradingview.sh."""
    if _cdp_up():
        return "TradingView ya estaba abierto"
    subprocess.Popen(["bash", LAUNCH_SCRIPT], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(45):
        time.sleep(1)
        if _cdp_up():
            time.sleep(8)  # deja que cargue el gráfico
            return "TradingView abierto"
    raise RuntimeError("No pude abrir TradingView; revisa launch_tradingview.sh")


def _chart_target():
    _ensure_running()
    with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=5) as r:
        targets = json.load(r)
    for t in targets:
        if t.get("type") == "page" and "tradingview.com/chart" in t.get("url", ""):
            return t
    raise RuntimeError("No hay una pestaña de gráfico de TradingView abierta")


def _cdp(method, params=None):
    t = _chart_target()
    with connect(t["webSocketDebuggerUrl"], max_size=32 * 1024 * 1024) as ws:
        ws.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(ws.recv(timeout=15))
            if msg.get("id") == 1:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]


def _try(fn):
    try:
        fn()
    except Exception:
        pass


@mcp.tool()
def open_tradingview() -> str:
    """Abre TradingView Desktop con el puerto de depuración si no está abierto."""
    return _ensure_running()


@mcp.tool()
def list_tabs() -> list[dict]:
    """Lista las pestañas/páginas expuestas por TradingView Desktop."""
    with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=5) as r:
        return [{"title": t.get("title"), "url": t.get("url")}
                for t in json.load(r) if t.get("type") == "page"]


@mcp.tool()
def get_chart_state() -> dict:
    """Símbolo y timeframe del gráfico activo. Para ver el precio usa screenshot()."""
    res = _cdp("Runtime.evaluate", {"expression": _STATE_JS, "returnByValue": True})
    state = json.loads(res["result"]["value"])
    return state


@mcp.tool()
def screenshot() -> Image:
    """Captura PNG del gráfico actual."""
    res = _cdp("Page.captureScreenshot", {"format": "png"})
    return Image(data=base64.b64decode(res["data"]), format="png")


if __name__ == "__main__":
    # Abre TradingView al arrancar Hermes, sin bloquear el handshake MCP.
    if os.getenv("TV_AUTOLAUNCH", "1") == "1":
        threading.Thread(target=lambda: _try(_ensure_running), daemon=True).start()
    mcp.run()

"""Servidor MCP de solo lectura para TradingView Desktop vía CDP (localhost).

Herramientas fijas: no hay evaluación arbitraria de JS ni clics, así que no
puede operar el broker. Requiere TradingView lanzado con --remote-debugging-port.
"""
import base64, json, os, re, urllib.request

from mcp.server.fastmcp import FastMCP, Image
from websockets.sync.client import connect

CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9222")
mcp = FastMCP("tradingview")

# Expresiones JS fijas (solo lectura)
# Los selectores del DOM de TradingView pueden cambiar entre versiones (best effort).
_STATE_JS = """JSON.stringify({
  title: document.title,
  url: location.href,
  legend: (document.querySelector('[data-name="legend-source-title"]') || {}).innerText || null,
  legend_values: (document.querySelector('[data-name="legend-series-item"]') || {}).innerText || null,
  timeframe: (document.querySelector('#header-toolbar-intervals button[aria-checked="true"]') || {}).innerText || null
})"""


def _chart_target():
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


@mcp.tool()
def list_tabs() -> list[dict]:
    """Lista las pestañas/páginas expuestas por TradingView Desktop."""
    with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=5) as r:
        return [{"title": t.get("title"), "url": t.get("url")}
                for t in json.load(r) if t.get("type") == "page"]


@mcp.tool()
def get_chart_state() -> dict:
    """Título y URL del gráfico activo (el título suele incluir símbolo y precio)."""
    res = _cdp("Runtime.evaluate", {"expression": _STATE_JS, "returnByValue": True})
    state = json.loads(res["result"]["value"])
    m = re.match(r"\s*([^\s]+)\s+([\d.,]+)", state["title"])
    if m:
        state["symbol"], state["price"] = m.group(1), m.group(2)
    return state  # si legend es null, usa screenshot()


@mcp.tool()
def screenshot() -> Image:
    """Captura PNG del gráfico actual."""
    res = _cdp("Page.captureScreenshot", {"format": "png"})
    return Image(data=base64.b64decode(res["data"]), format="png")


if __name__ == "__main__":
    mcp.run()

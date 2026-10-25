#!/usr/bin/env bash
# Instala todo para conectar Hermes con TradingView Desktop (idempotente).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$HOME/.hermes/config.yaml"

[[ -f "$CFG" ]] || { echo "No encuentro $CFG (¿Hermes instalado?)" >&2; exit 1; }

echo "1/4 Entorno Python"
[[ -x "$DIR/.venv/bin/python" ]] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
"$DIR/.venv/bin/python" -c "import mcp, websockets"

echo "2/4 Config de Hermes"
if grep -q '^  tradingview:' "$CFG"; then
  echo "   ya existe la entrada tradingview, no la toco"
else
  cp "$CFG" "$CFG.bak.$(date +%s)"
  if grep -q '^mcp_servers:' "$CFG"; then
    echo "   mcp_servers ya existe: añade a mano la entrada de config.example.yaml" >&2
    exit 1
  fi
  cat >> "$CFG" <<YAML

mcp_servers:
  tradingview:
    command: $DIR/.venv/bin/python
    args:
      - $DIR/tv_mcp.py
    env:
      CDP_URL: http://127.0.0.1:9222
YAML
fi

echo "3/4 Prueba del servidor MCP"
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
  | TV_AUTOLAUNCH=0 timeout 5 "$DIR/.venv/bin/python" "$DIR/tv_mcp.py" | grep -q serverInfo && echo "   MCP responde"

echo "4/4 Abriendo TradingView"
"$DIR/launch_tradingview.sh" >/dev/null 2>&1 &
sleep 20
curl -fs http://127.0.0.1:9222/json/version >/dev/null && echo "   TradingView con CDP activo" \
  || echo "   TradingView aún no responde en 9222 (revisa launch_tradingview.sh)"
"$DIR/.venv/bin/python" -c "import tv_mcp; print(tv_mcp.get_chart_state())" 2>&1 | tail -1 || true

echo "Listo. Reinicia Hermes."

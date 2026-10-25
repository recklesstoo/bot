#!/usr/bin/env bash
# Lanza TradingView Desktop (Linux) con el puerto de depuración CDP, solo en localhost.
set -euo pipefail

PORT="${CDP_PORT:-9222}"
APPIMAGE="${TRADINGVIEW_APPIMAGE:-}"
ARGS=(--remote-debugging-port="$PORT" --remote-debugging-address=127.0.0.1)

if [[ -n "$APPIMAGE" ]]; then
  "$APPIMAGE" "${ARGS[@]}" &
elif command -v tradingview >/dev/null 2>&1; then
  tradingview "${ARGS[@]}" &
elif command -v snap >/dev/null 2>&1 && snap list tradingview >/dev/null 2>&1; then
  snap run tradingview "${ARGS[@]}" &
else
  echo "No encontré TradingView. Define TRADINGVIEW_APPIMAGE=/ruta/TradingView.AppImage" >&2
  exit 1
fi

for _ in $(seq 1 30); do
  if curl -fs "http://127.0.0.1:$PORT/json/version" >/dev/null; then
    echo "CDP activo en http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
echo "El puerto $PORT no respondió. Si usas snap, prueba la AppImage o el .deb." >&2
exit 1

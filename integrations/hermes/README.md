# TradingView Desktop (Linux) -> Hermes Agent

Flujo: Hermes --MCP stdio--> servidor MCP de TradingView --CDP :9222--> TradingView Desktop.

1. `./launch_tradingview.sh` (usa `TRADINGVIEW_APPIMAGE=/ruta/app.AppImage` si aplica).
2. Elige un servidor MCP que hable CDP con TradingView y audítalo.
3. Copia `config.example.yaml` a `~/.hermes/config.yaml` con el comando real y reinicia Hermes.
4. Verifica que Hermes liste las herramientas `tradingview`.

Seguridad: el puerto CDP da control total de la app; el script lo ata a 127.0.0.1.

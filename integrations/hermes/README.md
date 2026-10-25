# TradingView Desktop (Linux) -> Hermes Agent

Hermes --MCP stdio--> tv_mcp.py --CDP :9222--> TradingView Desktop (solo lectura).

1. `./launch_tradingview.sh`
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Añade `config.example.yaml` a `~/.hermes/config.yaml` y reinicia Hermes.
4. Pregunta a Hermes por el símbolo/precio del gráfico o pídele una captura.

Seguridad: CDP solo en 127.0.0.1; el servidor no ejecuta JS arbitrario ni hace clics.
Aun así, usa cuenta demo o desconecta el broker mientras pruebas.

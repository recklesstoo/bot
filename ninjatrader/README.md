# Bot de futuros para NinjaTrader 8 con IA local (Ollama)

```
NinjaTrader 8  ── cierre de vela: POST /decide (velas + posición + PnL) ──▶  ai_server (Python, FastAPI)
(AIFuturesTrader.cs)                                                         │ calcula EMA/RSI/MACD/ATR
   ▲  ejecuta la orden con stop + target                                     │ pregunta a Ollama (LLM local)
   └──────────────  {"action":"BUY|SELL|HOLD|EXIT","confidence":0.8}  ◀──────┘
```

- **La IA decide la dirección.** El riesgo lo gestiona siempre NinjaTrader: cada entrada lleva stop y target (en múltiplos de ATR), hay pérdida máxima diaria, límite de trades por día, horario de operación y cierre antes del fin de sesión.
- **Si algo falla, el bot no opera:** si Ollama está caído, la respuesta no es válida, llega tarde (ya cerró otra vela) o la confianza es baja, la acción es HOLD.
- La IA solo se consulta en tiempo real, nunca con datos históricos.

## 1. Instalar la IA local (gratis)

1. Instala Ollama: https://ollama.com/download (Windows)
2. Descarga el modelo:
   ```
   ollama pull qwen2.5:7b-instruct
   ```
   - GPU de 8 GB o más → `qwen2.5:7b-instruct` o `llama3.1:8b`
   - GPU de 12–16 GB → `qwen2.5:14b-instruct` (mejor razonamiento)
   - Sin GPU → `qwen2.5:3b-instruct` (más rápido, menos preciso)

## 2. Arrancar el servidor de decisiones

En la misma PC donde corre NinjaTrader:

```
cd ai_server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          (ajusta OLLAMA_MODEL si usas otro)
python server.py
```

Comprueba en el navegador que todo está bien: http://127.0.0.1:8000/health tiene que devolver `"ok": true`.
Cada decisión se guarda en `ai_server/decisions.jsonl`.

## 3. Instalar la estrategia en NinjaTrader 8

1. Copia `AIFuturesTrader.cs` en `Documentos\NinjaTrader 8\bin\Custom\Strategies\`
2. En NinjaTrader: **New → NinjaScript Editor**, abre la estrategia y pulsa **F5** para compilarla.
3. Abre un gráfico del contrato, por ejemplo **MNQ 12-26** en velas de **5 minutos**, con al menos 5 días de datos cargados.
4. Clic derecho → **Strategies** → añade **AIFuturesTrader**.
5. En **Account**, elige tu cuenta real (o Sim101 para probar), revisa los parámetros y marca **Enabled**.

> Para operar con dinero real, NinjaTrader necesita una cuenta de broker de futuros conectada (NinjaTrader Brokerage, o un broker vía Rithmic/CQG) y licencia de live trading, o usar su propio broker.

## Parámetros

| Grupo | Parámetro | Por defecto | Qué hace |
|---|---|---|---|
| IA | Server URL | `http://127.0.0.1:8000/decide` | Dónde escucha `ai_server` |
| IA | Api token | vacío | Si lo pones, debe ser igual que `AI_API_TOKEN` en `.env` |
| IA | Confianza mínima | 0.65 | Por debajo de este valor, no se abre posición |
| Órdenes | Contratos | 1 | Contratos por operación |
| Órdenes | Permitir cortos | Sí | Permite abrir posiciones en corto |
| Órdenes | Permitir reversión directa | No | Si está en No, una señal contraria solo cierra la posición y no la da vuelta |
| Riesgo | Stop / Target (x ATR) | 1.5 / 3.0 | Distancia del stop y del target |
| Riesgo | Stop mín/máx (ticks) | 8 / 200 | Límites del stop |
| Riesgo | Pérdida diaria máx ($) | 300 | Si se alcanza, cierra todo y no opera más hasta la próxima sesión |
| Riesgo | Máx trades por día | 6 | |
| Horario | Inicio / Fin (HHmmss) | 093500 / 154500 | **En la zona horaria de tu PC/NinjaTrader** (los valores por defecto asumen hora de Nueva York) |
| Horario | Cerrar fuera de horario | Sí | Cierra la posición al salir de la ventana |

La pérdida diaria se revisa al cierre de cada vela. Entre velas, lo que protege es el stop de cada operación, que está en el broker.

## Antes de poner dinero real

- Empieza con **micros** (MNQ, MES, MCL) y 1 contrato.
- Déjalo unos días en **Sim101** con datos en vivo, o en **Playback** (Market Replay), y revisa `decisions.jsonl` y el Strategy Performance. Un LLM no garantiza ganancias: compara sus resultados con los de no operar.
- Ajusta **Pérdida diaria máx** a una cantidad que puedas perder sin problema.
- La PC y el servidor deben seguir encendidos mientras el bot esté activo. Si `ai_server` se cae, el bot deja de abrir operaciones, pero los stops ya colocados siguen en el broker.

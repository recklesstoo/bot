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

## Instalación rápida (Windows)

1. Descarga el repo desde GitHub: rama `claude/laughing-ptolemy-95sg9p` → **Code → Download ZIP**, y descomprímelo (por ejemplo en `C:\bot`).
2. Doble clic en **`INSTALAR.bat`**. Copia la estrategia a `Documentos\NinjaTrader 8\bin\Custom\Strategies\` (también si tus Documentos están en OneDrive), instala las dependencias de Python e instala Ollama con el modelo.
3. Doble clic en **`INICIAR_SERVIDOR.bat`** y deja la ventana abierta.
4. En NinjaTrader: **New → NinjaScript Editor → Strategies → AIFuturesTrader → F5**.

Los pasos 1 a 3 de abajo explican lo mismo a mano.

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

## Configuración por defecto: MNQ, 5 minutos, pérdida diaria de 200 $

En MNQ, 1 tick = 0.25 puntos = **0.50 $** y 1 punto = **2 $** por contrato.

| | Ticks | Puntos | $ con 1 contrato |
|---|---|---|---|
| Stop mínimo | 40 | 10 | 20 $ |
| Stop máximo | 160 | 40 | 80 $ |
| Stop típico (1.5 × ATR de 5 min, con ATR de ~15–25 pts) | 90–150 | 22–38 | 45–75 $ |
| Target (3 × ATR, 2:1 respecto al stop) | 180–300 | 45–75 | 90–150 $ |

Con 200 $ de límite diario caben entre 2 y 4 stops seguidos. Por ejemplo, si llevas -150 $ en el día, el bot solo abre una operación si su stop cuesta 50 $ o menos; si no, la bloquea. La pérdida del día solo puede pasar de 200 $ por deslizamiento (slippage) en el stop o por un hueco de precio (gap).

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
| Riesgo | Stop mín/máx (ticks) | 40 / 160 | Límites del stop (MNQ: 20 $ / 80 $ por contrato) |
| Riesgo | Pérdida diaria máx ($) | 200 | Si se alcanza, cierra todo y no opera más hasta la próxima sesión. Además, **no se abre ninguna entrada cuyo stop pueda superar lo que queda de ese margen** |
| Riesgo | Máx trades por día | 4 | |
| Horario | Inicio / Fin NY (HHmmss) | 093500 / 154500 | **Siempre en hora de Nueva York**, la zona horaria de tu PC da igual. La estrategia convierte la hora y aplica sola el cambio de horario de EE. UU. En Panamá (UTC-5) equivale a 08:35–14:45 de marzo a noviembre y a 09:35–15:45 de noviembre a marzo |
| Horario | Cerrar fuera de horario | Sí | Cierra la posición al salir de la ventana |

La pérdida diaria se revisa al cierre de cada vela. Entre velas, lo que protege es el stop de cada operación, que está en el broker.

## Antes de poner dinero real

- Empieza con **micros** (MNQ, MES, MCL) y 1 contrato.
- Déjalo unos días en **Sim101** con datos en vivo, o en **Playback** (Market Replay), y revisa `decisions.jsonl` y el Strategy Performance. Un LLM no garantiza ganancias: compara sus resultados con los de no operar.
- Ajusta **Pérdida diaria máx** a una cantidad que puedas perder sin problema.
- La PC y el servidor deben seguir encendidos mientras el bot esté activo. Si `ai_server` se cae, el bot deja de abrir operaciones, pero los stops ya colocados siguen en el broker.

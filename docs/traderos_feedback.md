# TraderOS: fallos y mejoras encontrados al usarlo por MCP

Contexto general: sesión de investigación con las herramientas MCP de TraderOS (`query_data`,
`market_ohlcv`, `market_orb_study`, `market_coverage`, `list_strategies`). Objetivo: medir si ideas
intradía sobre el rango de apertura (ES y NQ, velas de 5 minutos, 09:30-11:00 Nueva York, 2024-2026)
tenían ventaja, con un año reservado para validar. Cada punto dice qué se estaba haciendo, qué falló
y qué se propone. Lo marcado "no probado" se deduce de la descripción de la herramienta.

## A. Fallos

### A1. `query_data` repite las filas en la respuesta
- **Qué hacía:** bajar las velas de 5 minutos de ES y NQ, mes por mes (unas 380 filas por consulta,
  sin LIMIT), para analizarlas fuera. En total, 16 consultas por año de datos.
- **Qué falló:** una consulta de 396 filas devolvió 95.163 caracteres. Las 396 filas aparecen en
  `artifacts[0].payload.rows` y otra vez en `frames[].payload.rows`, además de 50 en `data.rows`. Solo
  las filas son unos 47.000 caracteres. La respuesta superó el límite del cliente y quedó guardada en
  un archivo. Con LIMIT 50 y paginación (17 llamadas), cada página también traía las filas repetidas,
  así que el contexto se llenaba sin necesidad. Un cliente sin copia a archivo perdería los datos.
- **Propuesta:** devolver las filas una sola vez (o una referencia al artefacto) y no copiarlas en
  `frames`. Opción `return=inline|file`.

### A2. `market_ohlcv` no devuelve la ventana que describe
- **Qué hacía:** pedir 120 días de velas de 5 minutos de ES para estudiar el volumen.
- **Qué falló:** el resumen dice `bars = 23568` pero `recent_bars` trae 10. La descripción promete
  "raw bars" y no hay forma de paginar el resto, así que la herramienta no sirve para obtener datos.
- **Propuesta:** devolver todas las filas (paginadas o en archivo) o renombrar la herramienta y
  aclarar que solo da un resumen.

### A3. El volumen del futuro continuo se distorsiona en las semanas de vencimiento
- **Qué hacía:** calcular el volumen relativo de la vela de ruptura (su volumen frente a la media de
  la misma hora en los 20 días anteriores) en ES.
- **Qué falló:** el volumen de 09:30 a 11:00 cae a menos de la mitad de la mediana (362.602) justo
  antes de cada vencimiento trimestral: 17 a 19 de marzo, 15 a 19 de junio y 14 y 15 de septiembre de
  2026. La serie parece seguir el contrato que vence cuando el volumen ya pasó al siguiente. Hubo que
  excluir esos días a mano con un filtro de "volumen del día < 50% de la mediana". También aparecen
  los festivos (25 de mayo, 3 de julio, 7 de septiembre).
- **Propuesta:** cambiar de contrato por volumen o por interés abierto, y devolver una columna
  `contract` o `roll_week`. Documentar la regla junto a `contract_stitching`.

### A4. `market_coverage`: `history_start` contradice la profundidad real
- **Qué hacía:** comprobar cuánto historial tenía MES antes de usarlo.
- **Qué falló:** `readiness` dice `history_start: 2026-10-17` (dos semanas) y `bar_depth.MES.1m.earliest`
  dice `2019-05-05`. La nota lo explica, pero el nombre del campo lleva a una conclusión equivocada.
  La llamada tardó unos 29 segundos y la comprobación de cadenas de opciones salió "unverified".
- **Propuesta:** llamar `history_start` a lo que es la primera barra guardada, o quitarlo del bloque de
  readiness.

### A5. Texto de relleno que no aplica a futuros
- **Qué hacía:** las mismas consultas de ES.
- **Qué falló:** cada respuesta incluye unos 1.900 caracteres (`volume_scope`, `ranking_basis`,
  `intraday_history`, `paging`), con un aviso sobre "Databento US equities feed, EQUS.MINI" aunque su
  propio campo `applies_to` dice que es para acciones. En 17 páginas son más de 30.000 caracteres sin
  información.
- **Propuesta:** omitir esos campos cuando no aplican o enviarlos una sola vez.

### A6. `market_orb_study` no define sus campos y no responde "cuándo"
- **Qué hacía:** el usuario preguntó "cuándo ocurre el breakout". Corrí el estudio en ES y NQ, rango de
  5 minutos, 250 días.
- **Qué falló:** devolvió 84.417 y 86.051 caracteres por el registro día a día y quedó guardado en un
  archivo. No trae la hora de la ruptura. `win_rate` (0,497) no está definido. No se explica qué es
  `held` ni qué dirección tiene un día con ruptura doble (en ES, 135 de 179 días). Hubo que calcular a
  mano el resumen a partir de `by_day`.
- **Propuesta:** añadir la hora de la primera ruptura, un modo `summary_only` y un apartado en la
  descripción que defina `held`, `double_break`, `win_rate`, el criterio por defecto (`by-wick`) y la
  zona horaria.

### A7. Desconexiones intermitentes del servidor
- **Qué hacía:** toda la sesión.
- **Qué falló:** la lista de herramientas de `traderos` se marcó varias veces como no disponible y
  disponible de nuevo, y hubo que volver a cargar los esquemas. Puede ser del cliente o del proxy; no
  tengo forma de saber si fue el servidor.
- **Propuesta:** registrar en el servidor las caídas de sesión MCP, para poder cruzarlas con este reporte.

## B. Mejoras

### B1. Que `run_code` pueda leer los datos de TraderOS
- **Qué hacía:** calcular fallos de ruptura y resultados por tramos de volumen sobre 3.000 velas por año.
- **Qué falló:** el entorno de `run_code` no tiene acceso a los datos y `input_data` admite 2 MB. Para
  usarlo habría tenido que retranscribir miles de filas. Trabajé fuera de TraderOS, y solo fue posible
  porque mi cliente guarda las respuestas grandes en archivos.
- **Propuesta:** un argumento `data` (ticker, timeframe, rango, sesión) que monte las velas como archivo
  dentro de `run_code`, o una herramienta de exportación a CSV o parquet.

### B2. Consultas con sesión y zona horaria
- **Qué hacía:** seleccionar 09:30-11:00 de Nueva York con `query_data`.
- **Qué falló:** la tabla está en UTC, así que filtré 13:30-14:55 UTC. Eso solo vale con horario de
  verano. Tuve que limitar todo a marzo-octubre y descartar unos cuatro meses de cada año. En invierno
  el mismo filtro habría seleccionado en silencio 08:30-09:55.
- **Propuesta:** un parámetro `session` (`RTH`, `ETH` o rango horario en la zona del mercado) que
  gestione el cambio de hora.

### B3. SQL más completo
- **Qué hacía:** querer detectar "primera vela que cierra fuera del rango y siguiente que vuelve dentro"
  directamente en SQL para no mover las velas.
- **Qué falló:** no hay funciones de ventana ni uniones consigo misma, y el tope de 500 filas con 5 s de
  tiempo obligó a bajar los datos.
- **Propuesta:** `lag`, `lead`, `row_number` y más filas con salida a archivo.

### B4. Crear y seleccionar estrategias desde MCP
- **Qué hacía:** llevar a TraderOS las estrategias de la sesión. Con `list_strategies` vi las 10 que ya
  tiene el usuario y quise probar la nueva con `backtest_run`.
- **Qué falló (no probado, por la descripción):** `backtest_run` corre "la estrategia activa de esta
  conversación" y pide un `session_id`, pero ninguna herramienta MCP crea o selecciona una sesión.
  `list_strategies` devuelve `strategy_id`, `saved_strategy_id` y `lineage_id`, y no dice cuál espera
  `backtest_run`.
- **Propuesta:** exponer `strategy_create` y una herramienta de selección de sesión, y documentar qué id
  espera cada herramienta.

### B5. Más historial en `run_pine_script`
- **Qué hacía:** preparar una versión en Pine de las estrategias para comprobarla contra Python.
- **Qué falló (no probado, por la descripción):** el tope de 10.000 velas recientes son unas 36 sesiones
  en 5 minutos si incluye horario extendido, poco para evaluar una estrategia.
- **Propuesta:** cargar solo el horario regular, o ejecutar por tramos, y devolver la lista de operaciones
  para poder compararla con otro motor.

### B6. Corrección por pruebas múltiples en los barridos
- **Qué hacía:** probar 14.580 combinaciones de reglas y elegir las 10 mejores con 2025 y 2026.
- **Qué falló:** las 10 daban t de 1,7 a 3,1 en esos dos años y promediaron -0,033 R en 2024, que no se
  usó para elegir. Con tantas combinaciones, la mejor sale buena por azar.
- **Propuesta (no probé `quant_sweep_*`; si ya lo hacen, ignorar):** que los barridos informen el número
  efectivo de pruebas independientes, calculen el Sharpe corregido y reserven un periodo de validación
  por defecto.

## Lo que funcionó bien
`query_data` responde en 15-50 ms, hay datos de ES y NQ desde 2016, los avisos de honestidad de datos
son claros y los artefactos permiten recuperar resultados grandes.

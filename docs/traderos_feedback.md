# TraderOS: qué debería hacer la plataforma sola, y fallos encontrados al usarla

Contexto: sesión de investigación por MCP. Objetivo: encontrar estrategias intradía con ventaja en ES y
NQ (rango de apertura, VWAP, FVG), validarlas con un año reservado y llevarlas a TraderOS. Dos partes:
(1) lo que tuve que hacer FUERA de la plataforma y debería hacer ella; (2) fallos e inconsistencias
encontrados al usar sus herramientas de punta a punta, con evidencia. Lo marcado "no probado" se
deduce de la descripción de la herramienta.

## Parte 1. Lo que la plataforma debería hacer sola

Para cada punto: qué necesitaba, qué tuve que hacer por fuera y qué debería ofrecer TraderOS.

### G1. Generar y probar muchas ideas, no solo filtros sobre un backtest terminado
- **Necesitaba:** probar miles de combinaciones de zona, disparador, stop, objetivo y filtros.
- **Hice por fuera:** un generador de 14.580 combinaciones y un motor de señales en Python. Midió todas
  en 20 segundos sobre 3 años de ES y NQ. El lazo de mejora de TraderOS (`strategy_filter_catalog`, 61
  filtros) solo prueba filtros sobre una estrategia ya backtesteada.
- **Debería ofrecer:** una búsqueda de ideas (disparadores x zonas x salidas) con las mismas garantías
  de validación que el resto de la plataforma.

### G2. Validación con periodo reservado y corrección por pruebas múltiples, por defecto
- **Necesitaba:** saber si el "mejor" de miles de candidatos era real.
- **Hice por fuera:** elegí las 10 mejores con 2025 y 2026 y las medí en 2024. Promediaron -0,033 R y
  solo 3 de 10 fueron positivas. Sin el año reservado, parecían buenas (t de 1,7 a 3,1 en los años de
  selección).
- **Debería ofrecer:** que todo barrido reserve un periodo que no se use para elegir y reporte el
  número de pruebas independientes. Mis 2.000 ideas equivalían a unas 47 independientes (matriz de
  correlación de resultados diarios); ese número alimenta `trials` en `quant_validate_deflated_sharpe`.

### G3. Cobertura de condiciones que mis mejores ideas necesitaban
- **Necesitaba:** banda de VWAP ±1 sigma, volumen relativo de la vela de la señal, ventanas de 30
  minutos (10:00-10:30), retesteo de FVG.
- **Qué hay:** `strategy_filter_catalog` ofrece 11 ventanas horarias fijas de 90 minutos, `volume_pace`
  mide el volumen relativo de la sesión (`rvol_prior`), y `price_vs_vwap` solo dice a favor o en contra
  de VWAP. No hay bandas de VWAP, FVG ni volumen por vela.
- **Debería ofrecer:** ventanas horarias libres, bandas de VWAP en sigmas y volumen relativo por vela.

### G4. Ejecutar una idea nueva sin salir de la plataforma
- **Necesitaba:** correr en TraderOS la estrategia que construimos.
- **Qué pasó:** la conexión MCP es de solo lectura. `strategy_find(resume=true)` devolvió
  `this connection is read-only; use the TraderOS app for that`, y no hay herramienta para crear una
  estrategia. `backtest_run` y `run_pine_script` piden aprobación en la app. Terminé usando un motor
  propio en Python y un exportador a NinjaScript y Pine.
- **Debería ofrecer:** crear y backtestear una estrategia desde el cliente (con aprobación del
  usuario), o al menos importar un Pine o una especificación y devolver el `run_id`.

### G5. Exportar datos y trabajar con ellos dentro de la plataforma
- **Necesitaba:** analizar unas 3.000 velas de 5 minutos por año (fallos de ruptura, volumen).
- **Qué pasó:** `run_code` no tiene acceso a los datos de TraderOS y `input_data` admite 2 MB. `query_data`
  no tiene funciones de ventana y su tope es de 500 filas. Tuve que bajar todo y analizarlo fuera.
- **Debería ofrecer:** que `run_code` monte las velas pedidas (ticker, rango, sesión), o exportación a
  CSV de cualquier consulta.

## Parte 2. Fallos e inconsistencias, con evidencia

### Probando las herramientas de investigación y validación

**F1. `quant_discovery_scan` falla siempre y no dice por qué**
- Lo llamé dos veces (por defecto y con `tickers=[ES,NQ], limit=5`). Ambas devolvieron:
  `Could not reach TraderOS API at https://api.traderos.ai/api/v1: .` (la causa va vacía).
- Propuesta: mensaje con la causa real (tiempo de espera, 5xx, límite) y reintento. Todo el resto de
  llamadas del día funcionó.

**F2. Errores de rechazo contradictorios**
- `strategy_find(resume=true)` devolvió `retry_same_tool: false`, `available_tools: []`,
  `what_exists_instead: []` y a la vez `next: "Retry once with a narrower request (one ticker, a
  shorter date range)"`. Ese consejo no tiene sentido para un rechazo de solo lectura.
- Propuesta: un mensaje específico ("esta conexión no puede cambiar la estrategia activa") y que la
  descripción de `strategy_find` avise de que `resume` necesita escritura.

**F3. `quant_scan_inefficiency` saca conclusiones de unos 9 días**
- ES y NQ en 5 minutos usaron `bars_used: 2500`, es decir, unas 9 sesiones (con horario extendido). La
  vida media sale en 444 y 934 velas (1,6 y 3,4 días), más larga que un tercio de la muestra, y no hay
  intervalo de confianza. NQ queda como "mean-reverting" (varianza 0,939) y la sugerencia siguiente es
  "probar una entrada de reversión dimensionada a la vida media", 934 velas en un gráfico de 5 minutos.
- El `inefficiency_score` es 0,8088 para ES, que sale "noisy / mixed", y 0,7886 para NQ, que sale
  "mean-reverting". El score no está definido y parece al revés.
- Los metadatos del artefacto marcan `gaps: no question or hypothesis recorded, limitations missing`.
- Propuesta: usar una ventana mínima configurable (semanas), dar intervalo de confianza o p-valor y
  documentar el score.

**F4. Walk-forward que excluye la mitad de la historia y esconde el último pliegue**
- Estrategia "NY open VWAP momentum continuation", 394 operaciones, backtest 2021-01-03 a 2026-10-23.
  `quant_validate_walk_forward` empieza el primer pliegue de entrenamiento el 2024-02-01: las
  operaciones de 2021 a 2023 (203 de 394, incluido 2022 con -549,7) no entran en ningún pliegue.
- Con `train_frac: 0.7` esperaba 276 operaciones de entrenamiento; cada pliegue tiene 68 a 72.
- El resumen es "3 robust / 1 overfit" y `strategy_results_read` muestra `wfo_verdict: "robust"`. El
  pliegue "overfit" es el más reciente (mayo a octubre de 2026, Sharpe fuera de muestra -0,0811).
- Propuesta: usar todo el historial, explicar la ventana de cada pliegue y que el veredicto global
  tenga en cuenta el pliegue más reciente.

**F5. Tres números de "confianza" distintos para la misma estrategia**
- Misma ejecución (394 operaciones): `strategy_results_read` da `overall_confidence_pct: 46` y
  `overall_verdict: MARGINAL`; `quant_validate_deflated_sharpe` da `confidence_pct: 76.58` y
  "needs more evidence"; el walk-forward da "robust".
- El Sharpe también difiere: 0,2983 en la ficha de resultados (diario, equity realizada) y 0,0359 en el
  Sharpe deflactado, sin unidad ni base indicada.
- `trials: 1` aunque la estrategia está en la versión 3 de su especificación (se cuentan solo las
  especificaciones backtesteadas).
- Propuesta: un único veredicto con la base de cada número, y que `trials` explique qué cuenta.

**F6. "NY open VWAP reversal MNQ": 9 operaciones y una confianza del 63%**
- `overall_verdict: "INSUFFICIENT SAMPLE"` (9 operaciones, 30 necesarias) junto a
  `overall_confidence_pct: 63`, win rate 66,67% y expectativa +0,67 R.
- `report_key_takeaways` dice que 140 señales caracterizadas no se reconciliaron, frente a 9 entradas.
  No hay motivo por señal. Podría ser un fallo del motor o una consecuencia de los filtros.
- `exit_reason` viene `null` en las 5 peores operaciones. En la operación 6 (long, 2025-06-05, stop de
  30,75 puntos) la salida fue 5 minutos después con -58,5 puntos (-118,6 USD, 1,9 R): comprobar el
  modelo de relleno del stop.
- Propuesta: no mostrar confianza cuando la muestra es insuficiente, dar el motivo de cada señal no
  reconciliada y rellenar `exit_reason`.

**F7. `strategy_dna` clasifica mal una reversión y pone nombre provisional**
- "NY open VWAP reversal MNQ": `factor_loadings: {trend: 0.3}`, sin carga de reversión a la media; el
  artefacto se titula "DNA: MNQ discovery strategy" (el nombre provisional).
- `list_strategies` tiene 3 de 10 estrategias con `name_is_placeholder: true`.
- Propuesta: calcular la carga de reversión y nombrar automáticamente desde la especificación.

**F8. `evaluate_edge` es correcto, pero repite sus datos unas diez veces**
- Verificación positiva: con las 178 operaciones de 2024 de una idea mía devolvió n 178, expectativa
  -0,0449, t -0,498, win rate 38,76% y PF 0,92, idéntico a mi cálculo en Python.
- Pero las 178 cifras aparecen unas diez veces en la respuesta (en `data`, en `meta.args`, en
  `categoryMeta.contract.settings` y de nuevo dentro de `frames`). Una respuesta de cuatro números ocupa
  más de 12.000 caracteres.
- Propuesta: devolver los argumentos una vez, y aceptar un `run_id` en lugar de pegar las operaciones.

### Datos y respuestas (observados antes)

**F9. `query_data` repite las filas.** 396 filas dieron 95.163 caracteres: las filas van en el artefacto
y otra vez en `frames`, además de 50 en `data.rows`. Superó el límite del cliente y quedó en archivo.
Propuesta: devolverlas una sola vez.

**F10. `market_ohlcv` no devuelve la ventana.** ES, 5m, 120 días: `summary.bars = 23568` y
`recent_bars` trae 10. Propuesta: devolver o paginar las filas, o aclarar que solo da un resumen.

**F11. Volumen de futuros continuos distorsionado antes de cada vencimiento.** En ES, el volumen de 09:30
a 11:00 cae por debajo del 50% de la mediana (362.602) el 17 a 19 de marzo, 15 a 19 de junio y 14 y 15
de septiembre de 2026, además de festivos. Propuesta: cambiar de contrato por volumen e indicar el
contrato o la semana de vencimiento.

**F12. `market_coverage`.** MES muestra `history_start: 2026-10-17` mientras la profundidad real es
2019-05-05, y la llamada tardó unos 29 s. Propuesta: usar la primera barra guardada.

**F13. `strategy_results_read` con ruido.** Para 394 operaciones devuelve un `futures_sizing.ledger` de
200 filas con riesgo por operación (más de 30.000 caracteres) cuando lo útil son unos 20 números.
Propuesta: omitir el libro de riesgo salvo petición.

**F14. Consultas en UTC.** Para seleccionar 09:30-11:00 de Nueva York filtré 13:30-14:55 UTC, válido
solo con horario de verano; tuve que descartar unos cuatro meses por año. Propuesta: parámetro `session`.

**F15. Texto de relleno.** Cada respuesta de `query_data` suma unos 1.900 caracteres de avisos sobre la
fuente de acciones, incluso en futuros.

**F16. Desconexiones intermitentes del servidor.** La lista de herramientas se marcó varias veces como
no disponible y de nuevo disponible durante la sesión. Puede ser del cliente o del proxy.

## Lo que funcionó bien
`evaluate_edge` coincidió con mi cálculo, `strategy_results_read` y `quant_validate_*` respondieron en
100 a 350 ms, hay datos de ES y NQ desde 2016, y los avisos de honestidad de datos son claros.

## Parte 3. Por qué importa cada mejora (la razón de trading)

Criterio para ordenarlas: **qué decisión con dinero cambia** si no se arregla. Primero lo que lleva a
operar algo sin ventaja, después lo que frena el trabajo, al final el ruido.

### Prioridad 1: llevan a operar algo que no tiene ventaja

**G2 y F5: validación con periodo reservado y un solo veredicto honesto.**
Con N ideas probadas, el mejor resultado esperado por puro azar crece con N. Con unas 47 ideas
independientes, un t entre 2 y 3 es lo normal por suerte, y mis 10 "mejores" tenían t de 1,7 a 3,1 en
los años de selección y -0,033 R en el año reservado. El costo de un falso positivo es capital real. Solo
la plataforma ve todas las pruebas de un usuario (cada edición de especificación y cada ronda del lazo
de mejora), así que solo ella puede corregirlo bien. Hoy `trials` vale 1 aunque la estrategia va por la
versión 3. Subestimar las pruebas infla la confianza. Por eso hace falta un único veredicto con la base de
cada número: en un caso hay 46%, 76,58% y "robust" a la vez, y el Sharpe aparece como 0,2983 y 0,0359.
Quien dimensiona capital lee un solo número; si hay tres, se queda con el más cómodo.

**F4: el walk-forward debe incluir los años malos y pesar el pliegue más reciente.**
La pregunta del walk-forward es "¿habría funcionado esta regla a medida que cambiaba el mercado?". La
respuesta más informativa está en los periodos hostiles. Aquí el año 2022 (-549,7 en 70 operaciones)
queda fuera, y el pliegue de mayo a octubre de 2026 (el más cercano a operar hoy) es el único "overfit".
Además, con unas 30 operaciones por pliegue el error estándar de un Sharpe por operación ronda 0,18, mayor
que las diferencias entre pliegues (0,05 frente a 0,19). Clasificar "robust" u "overfit" sin intervalo de
confianza es leer ruido.

**F6: muestra pequeña con confianza alta, y señales que desaparecen.**
Con 9 operaciones, ganar 6 ocurre por azar el 25% de las veces con una moneda justa, y "63% de confianza"
junto a "INSUFFICIENT SAMPLE" ancla al usuario en lo equivocado. Más grave: el estudio de eventos tenía
unas 149 señales y el motor operó 9. El usuario creerá que las estadísticas de las 149 describen un sistema
que ejecuta el 6%. Hay que decir por qué se descartaron (una posición a la vez, límite de sesión, filtro de
riesgo). La operación con -1,9 R y un stop de 0,95 R afecta a la expectativa: o hubo un hueco real que el
usuario debe ver, o el modelo de relleno del stop está mal.

**F11: el contrato continuo contamina el volumen y crea huecos falsos.**
El volumen de ES migra al siguiente trimestre unos ocho días antes del vencimiento. Una serie que sigue
al contrato que vence muestra "días de poco volumen" que no lo son, y contamina todo lo basado en volumen
relativo (`volume_pace`, `rvol_bucket`). Además, sin ajuste de contrato, la mañana del cambio muestra un
hueco del orden del diferencial entre contratos (alrededor del 0,5% del precio) que no ocurrió en el
mercado. Eso falsea el filtro `gap_size` y cualquier ruptura de rango. Un trader que filtra por hueco o
volumen estaría operando artefactos de datos.

**F3: regímenes "mean-reverting" a partir de 9 días.**
Varianza, autocorrelación y vida media necesitan cientos de observaciones independientes. Con 2.500 velas
solapadas de unas 9 sesiones, la estimación la domina el patrón horario (apertura y cierre) y el salto
nocturno. Una vida media de 934 velas son 3,4 días, no algo que se opere en intradía. Aun así la sugerencia
es "probar una entrada de reversión". Una etiqueta que invita a un trade sin respaldo estadístico es
peor que no tener la herramienta.

### Prioridad 2: frenan la investigación y hacen que se trabaje fuera

**G1: buscar disparadores, no solo filtros.** El edge de un trader está en la pareja evento más lugar
("fallo de una ruptura de la banda de VWAP", "retroceso al FVG"), no en apilar filtros sobre una
entrada mediocre. Un filtro solo quita operaciones: no descubre que otro disparador era mejor. Además,
cada filtro reduce la muestra y recorta las pocas operaciones grandes que concentran el beneficio, como
discutimos con el 33% de aciertos. Un lazo de mejora que solo filtra puede empeorar una estrategia.

**G3: las condiciones que usa un trader intradía.** VWAP y sus desviaciones son la referencia institucional
y actúan como imán de reversión. El volumen que importa es el de la vela del disparador frente a lo
normal de esa hora, porque una ruptura sin participación suele fallar. El volumen del día completo no se
conoce a las 09:35, así que usarlo sería mirar al futuro, y el nombre `rvol_prior` no aclara si es del día
anterior. Las ventanas de 90 minutos mezclan la subasta de apertura (primeros 30 minutos) con lo que
ocurre después; los edges intradía suelen vivir en ventanas de 15 a 30 minutos.

**G4: ejecutar la idea dentro de la plataforma.** Si la investigación se hace fuera, los costos, los
rellenos y las sesiones son distintos a los del motor real (mi motor entra a la apertura siguiente sin
deslizamiento; el motor de TraderOS aplica su propia regla de recorrido de la vela). Los resultados no se
pueden comparar. Y las pruebas hechas fuera no entran en el recuento de `trials`, con lo que se rompe la
propia protección contra el sobreajuste.

**G5: cómputo junto a los datos.** Ninguna plataforma puede anticipar cada estudio ("tasa de fallo de
ruptura por volumen relativo"). Si el usuario se lleva los datos, pierde las reglas de contrato, de sesión
y de fuente de volumen, y el resultado deja de ser reproducible dentro del producto.

**F14: sesión y zona horaria.** La lógica de futuros de EE. UU. es en hora local (apertura 09:30 ET). Un
filtro en UTC cambia de hora dos veces al año, y durante unas tres semanas Europa y EE. UU. cambian en
fechas distintas. Un filtro equivocado en silencio estropea justo los estudios de la primera hora, que son
los más comunes.

### Prioridad 3: confianza, costo y ruido

**F1 y F2: errores sin causa.** Un escaneo de mercados falla siempre y dice solo "api/v1: .". Quien depura
no puede separar un fallo de datos de uno de red, y el usuario no sabe si reintentar.

**F7: clasificación de la estrategia.** El "DNA" alimenta la diversificación y el ajuste al régimen. Llamar
"tendencia" a una reversión engaña sobre su correlación con otras estrategias y sobre cuándo apagarla:
las reversiones pierden en días de tendencia.

**F13: riesgo por operación muy desigual.** Con 1 contrato fijo y un stop distinto en cada operación, el
riesgo por trade va de 6,89 a 260 USD. El resultado en dólares lo mandan pocas operaciones de stop ancho,
y la expectativa en R (0,07) y en dólares (3,43 por operación) cuentan historias distintas. Conviene
mostrar un modo de riesgo fijo por operación como comparación por defecto.

**F8, F9, F10, F12, F15: tamaño y claridad de las respuestas.** Las respuestas repetidas cuestan tokens y
cabida de contexto, y un resultado truncado o con la historia mal etiquetada puede llevar a descartar un
estudio válido o a creer en uno corto (MES aparece con dos semanas de historia y tiene siete años). Pegar
178 cifras a mano en `evaluate_edge` también arriesga un error de transcripción que cambia el veredicto.

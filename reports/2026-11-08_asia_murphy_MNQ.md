# Murphy - Informe previo ASIA - MNQ
Autor: Murphy | Generado: sáb 7-nov-2026 ~00:38 ET (04:38 UTC) | Sesión: Asia, domingo 8-nov-2026
Apertura Globex: dom 8-nov 18:00 ET (23:00 UTC). Apertura Tokio / referencia del informe: 19:00 Bogotá = 00:00 UTC lun 9-nov = dom 8-nov 19:00 ET (EE. UU. ya en horario estándar desde el 1-nov).

**MERCADO CERRADO.** Todo precio es "último precio" del viernes 6-nov 16:59 ET (21:59 UTC), con 6h 39m de antigüedad al consultar. No es precio actual. Alcance: solo MNQ (MES y MGC no incluidos por encargo).

## 1. Resumen de una línea
MNQ cerró el vie 6-nov en 27107.25 (último precio, 16:59 ET), +1.91% sobre el cierre Globex del jueves (26600.5), tras caer 9.05% de cierre a cierre entre el 27-oct y el 6-nov (29804 a 27107.25). Rebota desde el mínimo de 5 sesiones (26367.25). Relación con MES y oro: no evaluada (fuera de alcance).

## 2. Niveles clave (MNQ)
Fuente: TraderOS market_quote, market_prev_day_range, market_key_levels, market_ohlcv (datos a 6-nov 16:59 ET / 21:59 UTC). Ventana indicada en cada nivel.

| Nivel | Precio | Ventana / nota |
|---|---|---|
| Último precio | 27107.25 | 6-nov 16:59 ET, barra 1m, no es en vivo |
| Máx día anterior (PDH) | 27425 | Globex 5-nov 18:00 a 6-nov 17:00 ET; igual al máx RTH |
| Mín día anterior (PDL) | 26367.25 | Globex; mín RTH 26988.5 |
| Cierre Globex | 27107.25 | Cierre RTH (16:00 ET): 27075.25 |
| Apertura Globex vie | 26635 | 5-nov 18:00 ET. Apertura RTH 9:30: no devuelta, SIN CONFIRMAR |
| VWAP de sesión | 27131.22 | Del viernes; base exacta de la ventana no indicada por la herramienta |
| POC / VAH / VAL RTH previo | 27258.80 / 27294.05 / 27104.90 | Perfil RTH del 6-nov |
| Rango Globex del viernes | 26367.25 - 27425 (1057.75 pts) | Volumen 4,253,960 |
| Rango overnight de Asia (dom) | SIN DATOS | La sesión no ha empezado |
| Huecos sin rellenar | SIN CONFIRMAR | La herramienta no devolvió huecos; no equivale a "no hay huecos" |
| AVWAP semanal / mensual | 27888.03 / 27904.20 | Anclas 2-nov / 1-nov; ambas por encima del último precio |
| Semana previa: PWH / PWL / PWC | 30180 / 28232.25 / 28545.75 | Semana W44, Globex |
| Mes previo: PMH / PML / PMC | 31616.5 / 28232.25 / 28545.75 | Octubre |
| SMA200 / EMA200 diaria | 28186.30 / 28342.14 | Por encima del último precio |

Aviso de datos:
- La serie continua no está ajustada por rolls. El Dec vence el 18-dic (SIN CONFIRMAR con herramienta), así que no se espera salto de roll esta semana. No se verificó con el historial de contratos.
- key_levels etiqueta las filas PDH/PDL/PDC con `session_date` 2026-11-05 y fecha 2026-11-06. Sus precios coinciden con market_prev_day_range del 6-nov, que es el que uso. Su campo `bucket_level_price` (UTC-day, p. ej. PDH 28578.25) lo ignoro porque está distorsionado.
- key_levels reportó que no pudo leer los límites de swing de 5 días. Parcial, no ausente.
- Última semana (cierres diarios Globex): 29804, 29755, 29217.75, 28545.75, 28226, 28416.75, 28646.25, 26600.5, 27107.25. El 5-nov hizo un rango de 26571.75 a 28668.75 (~2097 pts) con volumen 5,458,714, el mayor de la tabla.

## 3. GEX (MNQ)
**Es la instantánea del cierre del viernes 6-nov 16:45 ET (21:45 UTC), antigüedad 6h 53m al consultar. No es en vivo.** Fuente: TraderOS market_gex_levels, calculado de la cadena de opciones (OI x gamma por strike). La siguiente instantánea se construye cuando reabra el mercado.
- Régimen: gamma negativa. GEX neto -1,747,893 (unidades no especificadas por la herramienta).
- Put wall: 27000 (-0.40% frente a 27107.25). Soporte GEX 1.
- Soporte GEX 2: 26000.
- Resistencia GEX 2: 27400.
- Call wall: 28400 (+4.77%). Resistencia GEX 1.
- Gamma flip: 28793.99 (+6.22%), fuera del rango put/call wall. La herramienta lo señala como posible, por ser medidas distintas.
- Distancias medidas desde 27107.25 (cierre diario del 6-nov). La herramienta indicó que no hay precio de la última hora, por lo que no saco conclusión de "por encima/debajo" en vivo.
- Interpretación: la gamma negativa suele asociarse a mayor amplitud de movimiento. Es contexto, no señal.

## 4. Volatilidad
- ATR(14) diario: **638.13 pts (2.36% del precio)**, percentil 98 de su ventana de 120 días; máximo de la ventana 663.58 (4-ago). Termina en la barra del 6-nov. Base: sesión RTH 9:30-16:00 ET (market_atr_study). Hace una semana estaba en 468.44 (29-oct) y hace dos meses en ~372 (mín. 11-sep a 15-sep).
- Discrepancia a vigilar: market_key_levels usa ATR diario 765.84 (base distinta, no documentada). Citar siempre la base. El rango Globex completo del 5-nov (~2097 pts) y del 6-nov (1057.75 pts) supera ampliamente el ATR RTH de 638.
- Rango esperado (IV): 1.6772% (1 desviación típica de la IV ATM a 30 días / raíz de 252; IV tomada el 7-nov 00:01 UTC). Derivado por mí sobre 27107.25: unos +/-455 pts, es decir 26652 - 27562. **Cálculo propio, no devuelto directamente.**
- Banda predicha para lunes 9-nov (cierre previo +/- vol realizada 20d, 2.1123%): **26534.66 - 27679.84**. Registro reciente poco fiable: contenida el día completo en 1 de 5 y en 3 de 19 sesiones (rompió por abajo 11 veces de 19). Úsese como referencia y no como límite.
- Banda IV de la sesión 6-nov (26188.29 - 27081.71) fue rota por arriba (máx 27425).
- Canales: 5 sesiones 26367.25 - 28970.75; 20 sesiones 26367.25 - 31290. El mínimo de ambos canales es 26367.25.
- Nota: la banda de ATR/IV es para el día. Para la sesión de Asia (≈ 18:00 ET dom a 09:30 ET lun) no hay herramienta con rango específico de Asia, SIN CONFIRMAR.

## 5. Calendario programado (semana entrante)
Fuente: TraderOS market_economic_calendar, consulta 7-nov 04:38 UTC. Horas ET desde `event_time_et`. **El feed no trae consenso (`forecast` vacío) en estos eventos y no trae impacto del proveedor** (el "High/Medium" es clasificación de TraderOS por tipo de evento). Consenso: SIN CONFIRMAR. Valores "previous" abajo, tal como los devuelve el feed.

| Fecha ET | Hora ET | Evento | Previo | Consenso |
|---|---|---|---|---|
| Lun 9-nov | 10:00 | Monthly Wholesale Trade (sep) | 0.5% | SIN CONFIRMAR |
| Lun 9-nov | 10:00 | Goolsbee (Fed Chicago) en Leadership Greater Chicago | - | - |
| Mar 10-nov | 06:00 | NFIB Small Business Optimism (oct) | 97.9 | SIN CONFIRMAR |
| Mar 10-nov | **08:30** | **CPI (oct)**: CPI m/m, CPI a/a, Core CPI m/m, Core CPI a/a | 0.6% / 3.7% / 0.3% / 2.6% | SIN CONFIRMAR |
| Jue 12-nov | 08:30 | Peticiones semanales de desempleo | - | SIN CONFIRMAR |
| Jue 12-nov | 09:00 | Hammack (Fed Cleveland), conversación con estudiantes | - | - |
| Jue 12-nov | 10:00 | Existing Home Sales (oct) | 4,000,000 | SIN CONFIRMAR |
| Jue 12-nov | 14:00 | Monthly Treasury Balance (oct) | -27 mil millones | SIN CONFIRMAR |
| Jue 12-nov | 18:30 | Williams (Fed NY), panel en UC Berkeley | - | - |
| Vie 13-nov | **08:30** | **PPI (oct)**: PPI m/m, PPI a/a, PPI core m/m | 1.3% / 6.3% / 0.7% | SIN CONFIRMAR |

Notas:
- El CPI del martes es el evento de mayor peso de la semana, y el mercado llega con ATR en percentil 98.
- El feed muestra el NFP del 6-nov como fecha pasada en el calendario de lanzamientos; el feed nunca trae el dato publicado. El resultado del NFP no está en mis herramientas: SIN CONFIRMAR.
- Subastas del Tesoro: el feed no las lista. SIN CONFIRMAR (no equivale a que no haya).
- Festivos: el miércoles 11-nov es Veterans Day. Por conocimiento general, las bolsas de acciones operan y el mercado de bonos cierra; el feed no lo indica y no lo verifiqué con una fuente. **SIN CONFIRMAR**: comprobar horario CME antes de operar ese día. No hay festivo en domingo 8 ni lunes 9.
- Fuera de la ventana: FOMC 9-dic (calendario TraderOS).
- Vencimiento semanal de opciones: SIN CONFIRMAR.

## 6. Ruido a vigilar (máx. 3)
No consulté fuentes de noticias en esta ejecución (solo TraderOS). **Ningún titular verificado: SIN CONFIRMAR.** Para evitar relleno, no listo titulares. Dato (no titular) de contexto, por TraderOS: caída del 9.05% en 8 sesiones y el 5-nov con el mayor volumen de la tabla. La causa no está documentada en mis herramientas, así que no la atribuyo.
Pendiente para el trader: contrastar titulares del fin de semana en una fuente con hora, separando dato de opinión.

## 7. Revisión del informe anterior
**No existe informe anterior de Murphy para MNQ en `reports/`.** No hay aciertos ni fallos que revisar. Observación para el próximo informe, sin pretender que fuera una predicción previa: la banda IV del 6-nov (26188.29 - 27081.71) quedó rota por arriba (máx 27425), y el registro de la banda realizada (contenida 3 de 19) sugiere que subestima este régimen.
Nota: no verifiqué el directorio con una búsqueda para este informe, y me baso en la instrucción del encargo.

## 8. Fuentes y horas
Todas TraderOS (MCP), consultadas el 7-nov-2026 ~04:38 UTC (~00:38 ET):
- market_quote MNQ: barra 1m al 6-nov 21:59 UTC (16:59 ET).
- market_prev_day_range MNQ: sesión Globex 6-nov (datos 21:00 UTC; 1m hasta 21:59 UTC).
- market_key_levels MNQ: niveles a 6-nov; GEX 21:45 UTC.
- market_gex_levels MNQ: 6-nov 21:45 UTC (16:45 ET), antigüedad 6h 53m.
- market_atr_study MNQ (14, 120 d): termina 6-nov, base RTH.
- market_day_bands MNQ: IV a 7-nov 00:01 UTC; banda realizada basada en sesión 6-nov.
- market_economic_calendar: consulta 7-nov 04:38 UTC; horas del proveedor en UTC, convertidas a ET por la herramienta.
- market_ohlcv MNQ 1d: hasta 6-nov, sin ajuste por rolls.

*Contexto, no señales. No es recomendación de comprar ni vender.*

# Murphy - Informe previo a la apertura ASIA - MGC (oro micro)

- Sesión: Asia, apertura Tokio domingo 8-nov-2026 00:00 UTC = 19:00 ET = 19:00 Bogotá (desde el 1-nov EE. UU. está en horario estándar, UTC-5, así que ET y Bogotá coinciden). La reapertura de Globex es el domingo 18:00 ET = 23:00 UTC.
- Elaborado: sábado 7-nov-2026, ~04:39 UTC (consulta a TraderOS). Alcance: SOLO MGC. MNQ y MES no se han consultado.
- MERCADO CERRADO. Todo precio es "último precio" del viernes 6-nov, 16:59 ET (21:59 UTC). Edad del dato en la consulta: ~6h 39m. No es precio actual.

## 1. Resumen de una línea
MGC cerró la semana en 4079.8 (último precio, vie 6-nov 16:59 ET), +4.42% sobre el primer cierre horario de la semana (3907.2, dom 1-nov 18:00 ET), pero -2.0% bajo el máximo semanal de 4156.0 (jue 5-nov 22:00 ET). Relación con renta variable (MNQ/MES): SIN CONFIRMAR, no se consultó.

## 2. Niveles clave (MGC)
Fuente: TraderOS (market_quote, market_key_levels, market_prev_day_range, market_ohlcv 1h). Datos a 6-nov 21:59 UTC (16:59 ET). Ventana indicada en cada fila.

| Nivel | Precio | Ventana / nota |
|---|---|---|
| Último precio | 4079.8 | 1m bar, vie 6-nov 16:59 ET |
| Máx. día previo (PDH) | 4116.4 | Globex 18:00 ET jue 5-nov a 17:00 ET vie 6-nov; el máximo se hizo jue 19:00-20:00 ET |
| Mín. día previo (PDL) | 4032.0 | Misma sesión Globex; mínimo en la barra de 08:00 ET del viernes |
| Cierre día previo (PDC, Globex) | 4079.8 | Última barra 1m de la sesión Globex |
| Cierre RTH 13:30 ET | 4075.9 | Horario RTH del contrato 08:20-13:30 ET; máx. 4077.7, mín. 4032.0 |
| Apertura de la sesión Globex | 4098.0 | Jue 5-nov 18:00 ET (open_price del quote) |
| Rango cash-hours (09:30-16:59 ET vie) | 4039.3 - 4088.2 (mid 4063.75) | NO es el rango Globex completo |
| Rango overnight del viernes (aprox.) | 4032.0 - 4116.4 | Jue 18:00 ET a vie ~09:00 ET; granularidad de barras 1h, aproximado |
| VWAP de sesión | 4069.10 | Valor de la sesión del viernes. No se reinicia hasta la reapertura; no lo uses como VWAP de la sesión Asia |
| Área de valor RTH previa | VAL 4053.33 / POC 4058.55 / VAH 4076.18 | Perfil RTH previo, as_of 6-nov 18:29 UTC |
| Máx / mín semana en curso | 4156.0 / 3888.7 | Barras 1h, dom 1-nov 18:00 ET a vie 6-nov 16:00 ET |
| Semana previa (W44, Globex) | PWH 4097.3 / PWL 3879.3 / PWC 3884.4 | Cierre 30-oct |
| Mes previo (oct) | PMH 4259.0 / PML 3879.3 / PMC 3884.4 | |
| AVWAP semanal (ancla 2-nov) | 4033.35 | |
| AVWAP mensual (ancla 1-nov) | 4027.80 | |
| AVWAP swing low (ancla 7-oct) | 4089.53 | |
| AVWAP swing high (ancla 25-ago) | 4267.07 | |
| SMA200 4h / EMA200 4h | 4132.44 / 4127.98 | Por encima del último precio |
| EMA200 diaria / SMA200 diaria | 4284.18 / 4483.76 | Por encima |
| EMA200 semanal / SMA200 semanal | 3198.47 / 3037.91 | Muy por debajo |

Notas de datos:
- Posición del precio respecto a niveles: la herramienta marca `price_unavailable` (sin precio en la última hora), así que no saco conclusiones de "por encima/debajo" en vivo. Las distancias son las del último cierre.
- El quote da `prev_close` 4109.1 y `prev_session_close` 4098.1. Son cierres del jueves 5-nov (16:00 ET y cierre Globex anterior). El cierre del viernes es 4079.8.
- Huecos sin rellenar: SIN CONFIRMAR. Ninguna herramienta consultada devuelve huecos de MGC. Lo único observable es que el cierre del viernes (4079.8) y la reapertura del domingo definirán el hueco de fin de semana, si lo hay.
- Rolls: serie continua sin ajustar (roll_adjusted: false). Si hubo salto de contrato a finales de octubre, el mes previo y la semana previa pueden contenerlo. SIN CONFIRMAR si lo hubo. No hay vencimiento de MGC previsto para la semana entrante según lo que se sabe del calendario de contratos, pero tampoco lo verifiqué con una fuente: SIN CONFIRMAR.

## 3. GEX
No disponible para MGC (GEX solo MNQ y MES). `market_gex_levels` devolvió `data_availability: partial`: "no tenemos niveles gamma para MGC". Gamma flip, call wall y put wall: null.
Solo hay niveles técnicos de barras diarias, que NO son GEX: soporte 4071.0 (PDL/SMA20) y resistencia 4156.0 (PDH/BB). La herramienta no da timestamp (`as_of` null).

## 4. Volatilidad
- ATR(14) diario: 73.06 pts = 1.79% del precio. Percentil 32 en 120 días. Máximo de la ventana 88.06 (3-sep). Base: sesión cash 09:30-16:00 ET, barras de 5m, termina en la vela del 6-nov. Fuente: market_atr_study.
- Contraste: market_key_levels usa un ATR diario de 86.29 (base distinta, 14 periodos) y ATR 4h de 35.41, semanal 222.26. Dos bases dan cifras distintas; úsalas con esa salvedad.
- El rango Globex del viernes fue 84.4 pts (4032.0 a 4116.4), por encima del ATR cash de 73.06. En 32 de 85 sesiones (37.6%) el rango superó al ATR.
- ATR a la baja desde el 3-sep (88.06), mínimo reciente 61.29 (21-oct) y repunte a 73.
- Rango esperado de la sesión: `market_day_bands` FALLÓ (error de conexión con la API de TraderOS). Banda predicha por herramienta: NO DISPONIBLE. Referencia mía, no de la herramienta: último precio 4079.8 +/- ATR 73.06 da 4006.7 a 4152.9. Es solo escala de volatilidad, no pronóstico.

## 5. Calendario programado (semana entrante, relevante para oro)
Fuente: market_economic_calendar, consulta 7-nov 04:38 UTC. El feed trae previo pero NO consenso: todos los "forecast" vienen vacíos, así que consenso = SIN CONFIRMAR. El impacto lo asigna TraderOS por tipo de evento, no el proveedor. El feed no trae el dato publicado (actual), tampoco el NFP del 6-nov.

| Fecha (ET) | Hora ET / UTC | Evento | Previo | Consenso |
|---|---|---|---|---|
| Lun 9-nov | 10:00 / 15:00 | Goolsbee (Fed Chicago) habla | n/a | n/a |
| Mar 10-nov | 08:30 / 13:30 | CPI (IPC) oct: general Y/Y, M/M y subyacente. Impacto alto | Y/Y 3.7%; M/M 0.6%; subyacente Y/Y 2.6%; subyacente M/M 0.3% | SIN CONFIRMAR |
| Jue 12-nov | 08:30 / 13:30 | Peticiones de subsidio por desempleo | n/a | SIN CONFIRMAR |
| Jue 12-nov | 09:00 / 14:00 | Hammack (Fed Cleveland) | n/a | n/a |
| Jue 12-nov | 18:30 / 23:30 | Williams (Fed Nueva York) | n/a | n/a |
| Vie 13-nov | 08:30 / 13:30 | PPI oct. Impacto alto | Y/Y 6.3%; M/M 1.3%; subyacente M/M 0.7% | SIN CONFIRMAR |
| Mar 17-nov | 08:30 / 13:30 | Ventas minoristas. Impacto alto | n/a | SIN CONFIRMAR |

- Los "previo" son del proveedor; algunos valores (p. ej. PPI Y/Y 6.3%) llaman la atención y no los he contrastado con una segunda fuente. SIN CONFIRMAR.
- Festivos: el miércoles 11-nov es Veterans Day. Si afecta al horario de CME metales o a otra sesión: SIN CONFIRMAR (no verificado).
- Fuera de la ventana: FOMC 9-dic (calendario propio de TraderOS). La ventana consultada llega solo hasta el 16-nov; que no aparezca algo el 11-nov o el 16-nov no significa que no haya.
- Subastas del Tesoro: el feed no las lista. SIN CONFIRMAR.
- La Asia de la noche del lunes al martes cierra con el CPI del martes 08:30 ET (13:30 UTC), en el cierre de la sesión Asia/apertura de Londres.

## 6. Ruido a vigilar
No se consultó ninguna fuente de noticias (TraderOS no tiene herramienta de titulares y no se hizo búsqueda web). Titulares verificados: 0. SIN CONFIRMAR cualquier titular. Dato de contexto del propio precio, no una noticia: el oro subió unos 270 pts (3888.7 a 4156.0) del 30-oct al 5-nov y devolvió ~76 pts el viernes.

## 7. Revisión del informe anterior
No existe informe previo de MGC en el que basar una revisión (confirmado por el solicitante). Nada que revisar. Este informe sirve de línea base.

## 8. Fuentes y horas
- TraderOS market_quote MGC: barra 1m al 6-nov 21:59 UTC (16:59 ET). Consulta 7-nov ~04:38 UTC.
- TraderOS market_key_levels MGC (26 niveles; falta "swing bounds" 5 días): consulta 7-nov 04:38 UTC.
- TraderOS market_prev_day_range MGC: sesión Globex 6-nov, completa.
- TraderOS market_ohlcv MGC 1h, 5 días: última barra vie 6-nov 16:00 ET; mercado cerrado de vie 17:00 a dom 18:00 ET.
- TraderOS market_atr_study MGC: ATR(14) hasta el 6-nov.
- TraderOS market_gex_levels MGC: parcial, solo niveles técnicos.
- TraderOS market_economic_calendar: consulta 7-nov 04:38 UTC.
- FALLÓ: market_day_bands MGC (error de conexión con api.traderos.ai).
- Contexto, no señales: este informe no recomienda comprar ni vender.

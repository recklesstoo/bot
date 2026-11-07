# Murphy - Informe previo a la apertura ASIA - MES
Apertura Tokio: dom 8-nov-2026 19:00 Bogota = 19:00 ET = 00:00 UTC (lun 9-nov). Globex reabre dom 18:00 ET = 23:00 UTC.
Horario: EE. UU. cambio a horario estandar el 1-nov-2026, por lo que ET = UTC-5 = Bogota (sin desfase esta semana).
Alcance: solo MES (a peticion). MNQ y MGC no cubiertos en este informe.
Generado: sab 7-nov-2026 ~23:38 ET (04:38 UTC del 7-nov segun marcas de las herramientas).

AVISO DE MERCADO CERRADO: CME cerrado (vie 17:00 ET a dom 18:00 ET). Todos los precios son el ULTIMO PRECIO del vie 6-nov 16:59 ET (21:59 UTC), con 6h 39m de antiguedad al capturar. No es precio actual.

## 1. Resumen de una linea
MES cerro la semana en 7404 (ultimo precio, vie 6-nov 16:59 ET), -3.27% desde el cierre de 1h del lun 2-nov (7654.5) y -4.4% bajo el maximo semanal 7746.5 (mie 4-nov); ATR en su percentil 99 de 120 d y GEX en regimen gamma negativo. Relacion MNQ/oro: no cubierta (solo MES).

## 2. Niveles clave (MES, continuo sin ajustar por rolls)
Fuente: TraderOS market_quote, market_key_levels, market_prev_day_range, market_ohlcv (datos a vie 6-nov 21:59 UTC).

| Nivel | Precio | Nota |
|---|---|---|
| Ultimo precio | 7404 | vie 6-nov 16:59 ET (barra 1m) |
| Cierre RTH (16:00 ET) | 7400 | ventana 09:30-16:00 ET |
| Cierre Globex vie | 7404 | sesion 18:00-17:00 ET |
| Maximo vie (Globex y RTH) | 7469 | 9:00-10:00 ET |
| Minimo vie Globex | 7322.25 | formado jue 5-nov 18:00 ET (inicio de la sesion Globex de trade date 6-nov) |
| Minimo vie RTH | 7392 | |
| Apertura Globex vie (dato day_bands) | 7359 | |
| VWAP sesion | 7417.08 | de la sesion del vie; base horaria exacta SIN CONFIRMAR |
| POC / VAH / VAL RTH previo | 7452.40 / 7453.60 / 7417.67 | |
| EMA(200) diaria | 7402.23 | practicamente en el precio |
| SMA(200) diaria | 7365.54 | |
| AVWAP semanal (ancla 2-nov) / mensual (1-nov) | 7552.06 / 7553.11 | |
| Max semana 2-6 nov | 7746.5 | mie 4-nov 13:00 ET |
| Semana previa (PWH / PWL / PWC) | 7819.5 / 7557.75 / 7614.5 | Globex |
| Mes previo (PMH / PML / PMC) | 7897.5 / 7557.75 / 7614.5 | oct-2026 |

- Rango overnight de la apertura Asia (dom 18:00 ET): NO EXISTE todavia (mercado cerrado). SIN CONFIRMAR hasta que abra.
- Hueco de fin de semana: se medira contra 7404 (Globex) / 7400 (RTH) cuando abra; hoy no hay dato.
- Huecos sin rellenar previos: la herramienta de niveles no devolvio filas de gap fill. SIN CONFIRMAR (no equivale a "no hay huecos"). Dato parcial: `swing_bounds` de 5 d tampoco se pudo leer.
- Contexto de la semana (1h): jue 5-nov RTH, apertura 7724.75 a 7377-7380 en la tarde (barra 10:00 ET con minimo 7420.75, volumen 627k); vie 6-nov rango RTH 7392-7469, cierre 7400. Caida de ~3.3% en la semana.
- Roll: sin vencimiento esta semana (trimestral dic). SIN CONFIRMAR la fecha exacta de roll; la serie no esta ajustada, vigilar saltos cerca del roll de diciembre.

## 3. GEX (MES)
Fuente: TraderOS market_gex_levels, calculo propio sobre cadena de opciones (OI x gamma por strike).
**Snapshot del CIERRE del vie 6-nov 16:45 ET (21:45 UTC), 6h 53m de antiguedad. NO es en vivo; se reconstruye al reabrir.**
- Call wall / Resistencia 1: 7775 (+5.01% vs 7404)
- Resistencia 2: 7725
- Put wall: 7400 (-0.05% vs 7404), coincide con el cierre RTH y la EMA(200) diaria
- Gamma flip: 8096.65 (+9.36%), fuera de los muros (el propio sistema indica que son medidas distintas)
- Regimen: gamma negativo, net GEX -9,802,951 (metodo ladder_sum)
- Aviso de la herramienta: el soporte almacenado (7500) estaba del lado equivocado respecto al precio, por lo que no se sirve como soporte.
- Posicion precio vs niveles: la herramienta marca `price_unavailable`; no se saca conclusion de lado del nivel mas alla de las distancias vs 7404.

## 4. Volatilidad
- ATR(14) diario (estudio, sesion cash 09:30-16:00 ET, barra del 6-nov): 101.89 pts = 1.38% del precio, percentil 99 de la ventana de 120 d (max 102.98 el 5-nov). Era 69.5 el 29-oct: salto en una semana.
- Discrepancia: market_key_levels da ATR diario 121.75 (base distinta, probablemente Globex). Se usa 101.89 (cash) como referencia y se indica la diferencia; metodo de key_levels SIN CONFIRMAR.
- Rango esperado, banda predicha para lun 9-nov (cierre previo 7404 +/- VIX/16): 7285.35 a 7522.65 (+/-1.60%). VIX cierre vie = 25.64 (fuente: index_daily).
- Historial de esa banda: ultimas 5 sesiones contenida 3 de 5 (60%); ultimas 20 puntuables 10 de 14 (71%); roturas por abajo 4 de 14. La banda por IV de opciones (1.09%, +/-80 pts) cubre la sesion del 6-nov, no la del lunes.
- Canal 5 sesiones: 7322.25 a 7746.5 (tasa de acierto historica 40% sobre 10 sesiones). Canal 20 sesiones: no disponible.
- El rango del 6-nov en Globex fue 146.75 pts (volumen 1,744,290 contratos).

## 5. Calendario programado (semana entrante, horas ET)
Fuente: TraderOS market_economic_calendar, consultado 7-nov 04:38 UTC. El feed NO trae consenso (forecast vacio en todos los eventos) ni valor publicado; el consenso queda SIN CONFIRMAR. Impacto: calificacion de TraderOS por tipo de evento, no del proveedor.
- Lun 9-nov 10:00 ET: Goolsbee (Fed Chicago) habla; Wholesale Trade sept (previo 0.5%). Consenso: SIN CONFIRMAR.
- Mar 10-nov 08:30 ET: CPI oct (alto): CPI a/a previo 3.7%, CPI m/m previo 0.6%, Core m/m previo 0.3%, Core a/a previo 2.6%. Consenso SIN CONFIRMAR. Las cifras "previo" son las del proveedor y no se contrastaron con una segunda fuente (la combinacion Core 2.6% vs headline 3.7% es llamativa). NFIB 06:00 ET (previo 97.9).
- Mie 11-nov: Dia de los Veteranos; la bolsa de acciones abre normalmente y el mercado de bonos suele cerrar, pero este calendario no lo lista. SIN CONFIRMAR festivo/horario CME.
- Jue 12-nov: Claims 08:30 ET; Hammack 09:00 ET; Existing Home Sales 10:00 ET (previo 4.0M); Tesoro mensual 14:00 ET; Williams 18:30 ET.
- Vie 13-nov 08:30 ET: PPI oct (alto): PPI a/a previo 6.3%, Core m/m previo 0.7%, PPI m/m previo 1.3%. Consenso SIN CONFIRMAR; mismo aviso sobre cifras previas.
- Mar 17-nov 08:30 ET: Retail Sales; 09:15 ET Produccion Industrial.
- Nomina (NFP): el calendario interno la situaba el vie 6-nov, pero no aparece fila ni resultado en el feed (que nunca trae el dato publicado). Si se publico y su efecto en el cierre: SIN CONFIRMAR.
- FOMC: proxima decision 9-dic-2026 (calendario interno). Subastas de Tesoro: no aparecen en el feed; SIN CONFIRMAR.

## 6. Ruido a vigilar
Busqueda web (WebSearch, 7-nov) no devolvio titulares verificables de la sesion del 6-nov ni del resultado del informe de empleo; solo un adelanto semanal de ThriveInMarkets con consenso antes del dato (desempleo 4.2%, salarios +0.3%), sin hora y sin verificar el cierre. Por la regla de no inventar:
1. SIN CONFIRMAR: resultado del informe de empleo de oct (vie 6-nov 08:30 ET). Dato por comprobar en BLS.
2. SIN CONFIRMAR: causa de la caida del jue 5-nov (barra 9:00-10:00 ET, rango ~145 pts, volumen 627k a las 10:00 ET). Los datos muestran el movimiento, no el motivo.
3. Opinion/previa (no dato): ThriveInMarkets "week ahead" (https://thriveinmarkets.com/market-insights/week-ahead-2026-11-02/), sin hora verificada; solo como contexto de consenso.

## 7. Revision del informe anterior
No existe informe anterior de Murphy para MES en `reports/` en esta fecha (primer informe). No hay aciertos/fallos que evaluar. Lo que si cambio: el ATR cash paso de 69.5 (29-oct) a 101.89 (6-nov) y el precio perdio ~340 pts desde el maximo del 4-nov.

## 8. Fuentes y horas
- TraderOS market_quote MES: as_of 2026-11-06 21:59 UTC (16:59 ET).
- TraderOS market_key_levels MES: niveles a 6-nov; GEX 21:45 UTC; consultado 7-nov 04:38 UTC.
- TraderOS market_gex_levels MES: snapshot 2026-11-06 21:45 UTC (16:45 ET).
- TraderOS market_atr_study MES: ATR hasta barra 2026-11-06.
- TraderOS market_prev_day_range MES: sesion 6-nov, Globex.
- TraderOS market_day_bands MES: VIX 25.64; IV de opciones a 2026-11-07 00:01 UTC.
- TraderOS market_ohlcv MES 1h: ultima barra vie 6-nov 16:00 ET.
- TraderOS market_economic_calendar: consultado 2026-11-07 04:38 UTC.
- WebSearch 7-nov: ThriveInMarkets week ahead 2-nov (https://thriveinmarkets.com/market-insights/week-ahead-2026-11-02/).

Este informe es contexto, no una senal. No recomienda comprar ni vender.

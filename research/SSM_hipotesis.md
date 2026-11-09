# Short Session Method (SSM) — ficha de hipótesis

Estado: **pendiente de probar** · Autor de la ficha: Claude (a partir de la descripción del indicador) · Validación: Prado (independiente)

## Reglas tal como las describe el indicador
- **Instrumento:** NQ (lo probaremos también en MNQ y MES; MGC solo como control, otro tipo de activo).
- **Rango de apertura (OR):** máximo y mínimo de los primeros 5 min tras las 9:30 ET. Fijo toda la sesión.
- **Señal larga:** una vela de 5 min completada cierra por encima del OR high **y** del VWAP de sesión regular (se reinicia a las 9:30 ET). **Corta:** cierra por debajo del OR low y del VWAP. La mecha no cuenta.
- **Entrada:** modelada en la apertura de la siguiente vela de 1 min. Deslizamiento por defecto: 1 tick.
- **Stop:** extremo opuesto de la vela de confirmación.
- **Objetivo:** fijo, 0,5R (la mitad de la distancia entrada-stop).
- **Límites:** máx. 2 operaciones por sesión (una larga y una corta; la segunda exige la primera cerrada y una señal nueva en dirección contraria). Última entrada 14:55 ET. Lo abierto se cierra al final de la sesión regular.
- **Convención pesimista:** si una vela de 1 min toca stop y objetivo, se registra el stop primero.

## Por qué la hipótesis es exigente
Con objetivo de 0,5R el sistema necesita ganar **más del 66,7 %** de las veces solo para empatar antes de costes (1 / (1 + 0,5)). Con comisiones y deslizamiento el umbral sube. Si el stop de la vela de confirmación es ancho, el coste fijo pesa menos; si es estrecho, pesa más. La pregunta no es "gana en el histórico", sino si la entrada aporta algo frente al azar.

## Hipótesis falsable (a escribir ANTES de medir)
- **H1:** la tasa de acierto de SSM en MNQ/MES supera el umbral de equilibrio neto de costes de forma estadísticamente distinguible del azar.
- **H0:** el resultado es igual al de entrar al azar con la misma gestión.
- **Refutación:** SSM no supera al rival aleatorio (p. ej. percentil < 95 de la distribución aleatoria), o el resultado neto de costes es ≤ 0 en datos fuera de muestra.
- **Apuesta del trader:** _(escribir)_ · **Apuesta de Graham:** _(escribir)_

## Plan de prueba (Aarons)
1. Reproducir las reglas en velas de 1 min con VWAP de sesión regular. Registrar cuántas señales hay por año y por instrumento.
2. **Rival aleatorio:** mismos días, misma cantidad de largos y cortos, mismo stop/objetivo (0,5R) y mismas reglas de salida; entrada al azar (≥ 1.000 simulaciones). Comparar la distribución del resultado con el de SSM.
3. **Costes realistas:** comisión por contrato + 1 tick de deslizamiento por lado como base, y sensibilidad a 0 y 2 ticks.
4. **Tasa base:** ¿qué % de días tiene una señal? ¿La condición "cierra sobre OR y VWAP" aparece también en días perdedores?
5. **Datos nuevos:** reservar un periodo (p. ej. los años más recientes, o MES si se diseña con MNQ) sin mirarlo hasta el final.
6. **Variantes:** contar cuántas se prueban (objetivo 0,5R fijo vs otras) y avisar del riesgo de elegir la mejor.
7. **Estrés:** peores días (eventos macro, CPI/FOMC) y sensibilidad a la hora de entrada.
8. Segmentar por día de evento macro, volatilidad (percentil ATR) y régimen de gamma donde haya GEX (MNQ, MES).

## Puntos ambiguos que hay que fijar antes de codificar
- ¿La primera vela de confirmación puede ser la vela 9:35-9:40 o ya la vela del OR (9:30-9:35)?
- ¿"Señal fresca en la dirección contraria" exige cruzar el OR opuesto, o basta cerrar en el lado opuesto del VWAP?
- ¿El tamaño del stop tiene tope? (Una vela de confirmación enorme da un riesgo desproporcionado.)
- Roll de contrato: usar continua sin ajustar; excluir los días de roll o marcarlos.
- Horario de verano: el OR se define en hora de Nueva York (9:30 ET), no en hora de Bogotá.

## Reglas del equipo que aplican
- Aarons investiga, **no valida su propio resultado**; Prado audita. Nadie decide operar: decide el trader.
- Todo dato con fuente y hora; lo no verificable, SIN CONFIRMAR.

---
## Anexo 2026-11-09: estudio descriptivo de rupturas del rango de apertura (TraderOS `market_orb_study`)
Fuente: TraderOS · ventana de 250 días naturales (175 sesiones, aprox. mar-nov 2026) · cierre de vela de 5 min · "held" = el precio cierra la sesión más allá del nivel roto. **Es un estudio de eventos: no hay stop, objetivo, VWAP, costes ni P&L.**

| Rango (min) | NQ held | ES held |
|---|---|---|
| 5 | 52,0 % (n=175) | 56,0 % (n=175) |
| 15 | 49,1 % (n=175) | 54,6 % (n=174) |
| 30 | 55,0 % (n=169) | 57,0 % (n=172) |

Lectura: con n≈175 el error típico de cada cifra es ≈ ±3,8 puntos; las diferencias entre longitudes de rango y entre NQ y ES no son distinguibles del azar. Un día (5-nov, ruptura bajista de ~4 %) aporta gran parte del recorrido medio; muestra de un solo régimen. Este dato NO equivale a "acertar el objetivo de 0,5R": no se puede comparar con el 66,7 % de equilibrio.

## Combinaciones a probar (pre-registrar antes de medir; máx. ~12 variantes)
Ejes (SSM base = rango 5 min, cierre de 5 min sobre OR y VWAP, stop = extremo de la vela de confirmación, objetivo 0,5R):
1. Longitud del OR: 5 / 15 / 30 min.
2. Filtro VWAP: con / sin.
3. Confirmación: cierre / mecha.
4. Objetivo: 0,5R / 1R / 2R / sin objetivo (cierre de sesión).
5. Stop: extremo de la vela de confirmación / lado opuesto del OR.
6. Dirección: ambas / solo a favor de la tendencia diaria.
Instrumentos: MNQ y MES (MGC solo como control). Siempre contra el rival aleatorio con la misma gestión.

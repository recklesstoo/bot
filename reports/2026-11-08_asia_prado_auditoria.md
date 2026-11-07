# Prado - Auditoría de los informes previos ASIA (sesión dom 8-nov-2026): MNQ, MES, MGC
Autor: Prado (auditor independiente) | Fecha: vie 6-nov-2026 (hora ET) / sáb 7-nov UTC, sobre archivos con mtime 7-nov 04:39 UTC

**Alcance y límites.** Audito solo los documentos: CLAUDE.md, `.claude/agents/murphy.md` y los tres informes. No tengo herramientas TraderOS en esta ejecución, así que NO he verificado ningún dato contra el feed; todo lo que sigue es consistencia interna, recálculo aritmético y cumplimiento de reglas. No creo estrategias ni recomiendo operar. Verifiqué con `ls` que en `reports/` no hay informes previos de Murphy (lo que Murphy-MNQ dejó sin comprobar).

## Veredictos
| Informe | Veredicto |
|---|---|
| MNQ | **NO CONCLUYENTE** |
| MES | **NO CONCLUYENTE** |
| MGC | **RECHAZADO** (hay contradicciones internas en niveles clave y errores horarios; rehacer) |

Ninguno supera la auditoría. Criterio: RECHAZADO si hay un error verificable en un nivel o en una cifra que el trader usaría; NO CONCLUYENTE si la aritmética cuadra pero las cifras clave siguen sin contraste.

## 1. Cumplimiento de formato y reglas
Cumple en los tres: cada tabla cita herramienta y hora; "último precio" con 16:59 ET y antigüedad (6h39m y 6h53m del GEX, recalculadas y correctas); GEX marcado como snapshot de cierre; consenso, huecos y festivo como SIN CONFIRMAR; ninguna señal de compra/venta; ningún "máximo histórico" (grep).

Incumplimientos:
- **Hora de generación errónea (MNQ y MES).** 04:38 UTC del 7-nov son las **23:38 ET del vie 6-nov** (ET = UTC-5 desde el 1-nov). MNQ dice "sáb 00:38 ET" (usa UTC-4, contradice su propia conversión 16:59 ET = 21:59 UTC). MES dice "sáb 7-nov 23:38 ET" (día equivocado).
- **MGC: "domingo 8-nov 00:00 UTC = 19:00 ET".** Incorrecto: 00:00 UTC del 8-nov es el sáb 7 a las 19:00 ET. La apertura de Tokio es **lun 9-nov 00:00 UTC = dom 8-nov 19:00 ET** (MNQ y MES lo tienen bien).
- **Ruido (máx. 3 titulares con fuente y hora):** MNQ y MGC no consultaron noticias (cero titulares) aunque Murphy tiene WebSearch/WebFetch; MNQ delega en el trader ("contrastar titulares"). MES hizo una búsqueda pero aportó una previa de consenso del 2-nov (anterior al dato) sin hora; no es un titular verificado.
- **Resumen y relación entre instrumentos** (MNQ vs MES; oro vs renta variable): ninguno la entrega ("fuera de alcance"). Es consecuencia del encargo por separado, pero los tres se generaron a la vez y los datos permiten una comparación mínima (ver sección 5).
- **Consenso en calendario:** vacío en todos (el feed no lo trae). Murphy podía intentar WebSearch para consenso y no lo hizo.
- **Apertura:** MNQ deja la apertura RTH como SIN CONFIRMAR y solo da la Globex; MES toma la apertura de `day_bands`, no de la misma fuente que los demás niveles.
- Nombre de archivo `YYYY-MM-DD_asia_murphy_<TICKER>.md` difiere del patrón `..._murphy.md`; irrelevante si fue por encargo.

## 2. Consistencia interna (recalculado con los números de los informes)
Cuadran: MNQ +1.91% (506.75/26600.5), caída 9.05% (29804 a 27107.25), distancias GEX (-0.40%, +4.77%, +6.22%), rango Globex 1057.75, banda IV +/-455 y banda realizada 2.1123% (26534.7-27679.8), ATR% (2.36%); MES -3.27%, -4.4%, ~340 pts, GEX (+5.01%, -0.05%, +9.36%), VIX/16 (7285.35-7522.65), rango 146.75, ATR 1.38%; MGC +4.42%, rango 84.4, 32/85 = 37.6%, ATR 1.79%, referencia 4006.7-4152.9. Ratios de precio MNQ/MES (3.65-3.66 en muros) coherentes. Calendarios: conversiones ET/UTC de MGC correctas; días de la semana y Veterans Day (mié 11) correctos.

Contradicciones y errores:
1. **MGC, PDH vs máximo semanal.** PDH 4116.4 (sesión Globex jue 18:00 ET a vie 17:00 ET, "máximo jue 19:00-20:00") pero el máximo semanal 4156.0 se marca "jue 5-nov 22:00 ET", dentro de esa misma sesión. Ambos no pueden ser ciertos. Además la sección 3 llama a 4156.0 "PDH/BB". Afecta a PDH, rango overnight (4032.0-4116.4), rango Globex (84.4) y la comparación con el ATR. Posible error de huso horario en una de las fuentes; sin confirmar.
2. **MGC, "-2.0% bajo el máximo semanal":** (4079.8-4156)/4156 = **-1.83%**. Y "devolvió ~76 pts el viernes" mezcla jueves noche y viernes (el máximo es del jueves).
3. **MGC, "subió ~270 pts del 30-oct al 5-nov (3888.7 a 4156.0)":** el mínimo 3888.7 pertenece a la semana en curso (barras desde dom 1-nov 18:00), no al 30-oct (cierre 3884.4, mín. previo 3879.3).
4. **MGC, calendario:** dice que la ventana consultada llega "solo hasta el 16-nov" y lista Ventas minoristas del 17-nov. Dice que el CPI (13:30 UTC) cae en "el cierre de la sesión Asia/apertura de Londres": Tokio cierra ~06-07 UTC y Londres abre 08:00 UTC; 13:30 UTC es media sesión de Londres. (Según la definición de Asia de MNQ, 18:00-09:30 ET, el CPI sí cae dentro de la ventana del martes, pero la etiqueta es incorrecta.)
5. **MGC, base del ATR:** el estudio usa "cash 09:30-16:00 ET", pero el propio informe da el RTH del contrato como 08:20-13:30 ET. La base no coincide con el horario del oro; comparar "rango Globex 84.4 > ATR 73.06" mezcla bases.
6. **MES, IV frente a VIX.** La IV de opciones da 1.09% diario (~17.3% anualizado); el mismo informe cita VIX 25.64 (1.6% diario). Diferencia ~47% sin explicar. MNQ da 1.6772% (~26.6% anualizado). O la IV de MES es de otro vencimiento/fuente, o está mal. No se puede usar la banda IV de MES sin aclararlo.
7. **MES omite que su banda IV del 6-nov (7359 +/- 80 = ~7279-7439) quedó rota por arriba** (máx 7469); MNQ sí lo dice. Frase ambigua "cubre la sesión del 6-nov".
8. **Bandas no comparables entre informes:** MNQ usa vol realizada 20d (contenida 3/19), MES VIX/16 (10/14), MGC ninguna. Una banda de 1 sigma debería contener ~68%: MES (71%) cuadra, MNQ (16%) no; el régimen de MNQ está fuera de su propio modelo.
9. **Semana medida con bases distintas:** MNQ desde 27-oct (9 sesiones, -9.05%), MES desde cierre 1h del 2-nov (-3.27%), MGC desde el primer cierre horario del dom 1-nov (+4.42%). MES "caída semanal ~3.3%" frente al cierre semanal previo (PWC 7614.5) es -2.76%.
10. **NFP:** MNQ afirma que el feed muestra el NFP como "fecha pasada"; MES y MGC dicen que no aparece fila. Mismo feed y misma consulta, relatos distintos.
11. **MES, jueves 5-nov:** "barra 10:00 ET con mínimo 7420.75" frente a "barra 9:00-10:00 ET" y "7377-7380 en la tarde"; el mínimo del jueves y su hora no están claros.
12. Etiquetado GEX: en MNQ y MES "Resistencia GEX 1" es el muro más lejano y "2" el más cercano (27400 vs 28400; 7725 vs 7775); riesgo de lectura invertida. Unidades de GEX neto no especificadas (-1.75M vs -9.80M no comparables).
13. Perfil de volumen: POC/VAH/VAL con decimales fuera de la rejilla de 0.25 (p. ej. 27258.80, 7452.40): están agrupados/interpolados y el informe no lo dice. El POC queda pegado a VAH en ambos.
14. Menores: MNQ 2.36% (cálculo da 2.354% con 27107.25; cuadra con 27075.25); MNQ describe una IV (1.68% diario) inferior a la vol realizada (2.11%) sin comentario.

## 3. Cifras llamativas sin segunda fuente
| Cifra | Qué falta para contrastarla |
|---|---|
| **PPI a/a 6.3%, m/m 1.3%, core m/m 0.7%** (los tres). Un m/m de 1.3% es extremo, y 6.3% frente a CPI 3.7% / core 2.6% es una brecha de 2.6 pts. Solo MES y MGC lo señalan; MNQ lo copia sin aviso. | Nota del BLS con el dato de septiembre y su fecha de publicación, y que sea el "previo" correcto de la liberación del 13-nov. Podría ser un campo mal mapeado del proveedor (m/m vs a/a). |
| **CPI previo 3.7% / core 2.6% / m/m 0.6%.** | Mismo contraste con BLS. Solo MES lo marca. |
| **MNQ -9.05% en 8 sesiones y -7.14% en un día** (28646.25 a 26600.5, rango 5-nov de 2097 pts = 7.3%). MES en el mismo día cae ~4.5% (7724.75 a ~7377). La proporción 1.6x es coherente con la beta, pero un día así no tiene causa documentada. | Segunda fuente de precio (barras del contrato Dec en otro proveedor/NDX), causa del movimiento con titular y hora, y comprobar que no hay error de barra (rango Globex 5-nov 2097 pts frente a ATR 638). |
| **Rebote del vie** MNQ +1.9% frente a MES +0.6% desde la apertura Globex (+1.77% vs +0.61%), y rango Globex MNQ 3.97% vs MES 1.99%. | Mismas fuentes; sin ellas, no se sabe si es dinámica real o desalineación de datos. |
| **Gamma flip fuera de los muros** (MNQ +6.22%, MES +9.36%), y put wall a -0.40% / -0.05% del último precio. El sistema avisa de "medidas distintas", pero ambos flips quedan además fuera del canal de 5 sesiones. | Subyacente de la cadena (NDX/SPX/ETF) y basis frente al futuro: los muros de opciones están en puntos de índice y el futuro cotiza con prima de carry; sin ese ajuste, distancias del orden de 0.05-0.4% no son interpretables. Unidades del GEX y fecha de la cadena. |
| **Banda predicha MNQ con historial pobre** (3/19, rompe por abajo 11/19). | Documentar el modelo; no es un rango fiable en este régimen. |
| **Tesoro mensual previo -27 mil millones; Existing Home Sales 4.0M.** | Contraste con la fuente oficial; no lo señala ningún informe. (Sin verificar; solo extrañeza por conocimiento general.) |
| **VIX 25.64** (MES). | Segunda fuente de cierre del VIX del 6-nov. |

## 4. Discrepancia de ATR (key_levels vs estudio ATR)
| | Estudio (RTH) | key_levels | Ratio |
|---|---|---|---|
| MNQ | 638.13 | 765.84 | 1.200 |
| MES | 101.89 | 121.75 | 1.195 |
| MGC | 73.06 | 86.29 | 1.181 |

- El ratio es casi constante (1.18-1.20) en tres activos y regímenes distintos (MNQ/MES en percentil 98-99; MGC en percentil 32). Eso apunta a una diferencia sistemática de base (sesión completa frente a RTH, o definición del true range), no a un efecto de suavizado tras un pico.
- **Más defendible como cifra citable: el estudio ATR**, porque su base (ventana, barras, percentil, máximo) está documentada y es reproducible. Salvedad: en MGC su base (09:30-16:00 ET) no coincide con el horario del oro (08:20-13:30 ET).
- **Más relevante para una sesión overnight: probablemente key_levels** (rangos Globex de Murphy: MNQ 1057.75 = 1.66x el ATR RTH; MES 146.75 = 1.44x; MGC 84.4 = 1.16x), pero su base no está documentada (MES lo supone "probablemente Globex": conjetura).
- **Conclusión: no se puede decidir** qué base es la correcta para Asia. Hace falta la definición de key_levels o recalcular el ATR(14) con `market_ohlcv` en ambas bases. Mientras tanto, citar siempre las dos con su base (como hace Murphy) y no usar ninguna como límite.

## 5. Huecos de datos para usar los informes en la apertura
Quién es responsable (M = Murphy, F = feed/herramienta, E = encargo):
- **Antigüedad.** Datos del vie 21:59 UTC; en la apertura (dom 23:00 UTC reapertura, lun 00:00 UTC Tokio) tendrán ~49-50 h. Cualquier noticia o dato del fin de semana queda fuera (F/E). Hace falta regenerar tras la reapertura de Globex.
- **Rango overnight y hueco de fin de semana:** no existen aún; correcto marcarlos SIN CONFIRMAR (no es fallo de nadie). Hay que medirlos tras la apertura.
- **Huecos sin rellenar:** ninguna herramienta los devolvió en los tres, y `swing_bounds` 5 d no se leyó en MNQ, MES y MGC (F). Murphy lo marcó bien; no cabe concluir "sin huecos". Se podría calcular desde `market_ohlcv` (M, no lo hizo).
- **Consenso de calendario:** vacío en todos (F). Murphy no intentó WebSearch para consenso (M). Faltan consensos de CPI, PPI y NFP, y el dato publicado del NFP del 6-nov (clave para entender el cierre del viernes).
- **Calendario específico de Asia:** ningún informe tiene eventos de Japón/China ni de la ventana 18:00 ET dom a 09:30 ET lun; el primer evento listado (Goolsbee, 10:00 ET) cae después. No se sabe si el feed no cubre Asia o si Murphy no lo pidió (F/M sin determinar).
- **Noticias:** 0 titulares verificados en MNQ y MGC (M, herramientas disponibles); MES, ninguno aprovechable (F/M).
- **Opciones/IV de MGC y `market_day_bands` de MGC:** falló por error de conexión con api.traderos.ai (F). Murphy lo declaró y no inventó banda (bien); la referencia +/-ATR es solo escala.
- **Correlación entre los tres:** no está en ningún informe (E/M). Con los propios datos y la misma base (cierre semanal 30-oct a último precio 6-nov): MNQ -5.04%, MES -2.76%, MGC +5.03%; es decir, renta variable abajo, MNQ con aproximadamente 1.8x la caída de MES, y oro arriba. Es descripción de datos de los informes (no contrastada con el feed), no una relación estable ni una señal.
- **Roll:** Dec vence el 18-dic (viernes, correcto); no se espera salto esta semana, pero Murphy no lo verificó con el historial de contratos (MNQ lo dice; MES y MGC también sin verificar).
- **Fallos de Murphy atribuibles:** errores horarios, PDH/máximo de MGC sin reconciliar, IV de MES sin reconciliar con VIX, bases de semana distintas, no usar WebSearch en MNQ/MGC, no calcular huecos desde OHLCV.

## 6. Preguntas de decisión para el trader (con plazo sugerido: antes de la reapertura Globex, dom 8-nov 23:00 UTC = 18:00 ET)
1. MGC: ¿se rehace el informe tras reconciliar PDH 4116.4 frente a máximo 4156.0 y el huso horario de las barras, o se descarta MGC para la sesión? (RECHAZADO)
2. MNQ/MES: ¿se piden a Murphy los contrastes pendientes (PPI/CPI en BLS, causa del día -7.1% en MNQ, VIX/IV de MES, subyacente y basis del GEX) o se acepta que esas cifras se lean como SIN CONFIRMAR durante la sesión?
3. ¿Se ordena regenerar los tres informes tras la reapertura de Globex, con rango overnight, hueco de fin de semana, titulares del fin de semana y NFP publicado, o se asume la lectura con datos de ~50 h de antigüedad?
4. ATR: ¿se pide a TraderOS la definición de la base de key_levels o un recálculo con OHLCV para fijar una sola base de volatilidad para Asia?
5. ¿Se unifica el método de la banda predicha (hoy distinto en MNQ y MES, ausente en MGC) antes de usarla como referencia?

*Auditoría de documentos; no verifica datos contra TraderOS. No es señal ni aprobación para operar; decide el trader.*

# ConfluenciaMTF — Indicador para NinjaTrader 8

Combina **RSI + MACD + Volumen + ATR** en **1, 5 y 15 minutos**, marca **COMPRA / VENTA** en el gráfico y muestra el **ADR** (rango diario promedio).

## Instalación

1. Copia `ConfluenciaMTF.cs` a
   `Documentos\NinjaTrader 8\bin\Custom\Indicators\`
2. En NinjaTrader: **New → NinjaScript Editor**, abre el archivo y pulsa **F5** (compilar).
3. En un gráfico (recomendado **1 minuto**): clic derecho → **Indicators** → `ConfluenciaMTF`.
4. En *Data Series* del gráfico carga **al menos 20 días**, así hay historial suficiente para calcular el ADR.
5. Deja **Calculate = On bar close** para que las señales no repinten.

## Cómo leerlo

| En el gráfico | Significado |
|---|---|
| ⬆ Flecha verde + "COMPRA" | Las 3 temporalidades están alcistas y el volumen confirma |
| ⬇ Flecha roja + "VENTA" | Las 3 temporalidades están bajistas y el volumen confirma |
| Fondo verde / rojo suave | Las 3 temporalidades están alineadas (aunque todavía no haya señal) |
| Línea roja discontinua | Stop sugerido (ATR × 1.5) |
| Línea azul discontinua | Objetivo sugerido (2R) |
| Línea gris punteada | Precio de entrada |

### Panel (esquina superior derecha)

```
CONFLUENCIA MTF
TF     RSI  MACD     VOL  SESGO
1m    58.3  + alc   1.6x  ALCISTA
5m    55.1  + alc   1.1x  ALCISTA
15m   52.4  + alc   0.9x  ALCISTA
--------------------------------
ATR(14)  1m 2.75 (11 ticks)
         5m 6.25  15m 11.50
ADR(14d) 62.50 (250 ticks)
Hoy      41.25  = 66% del ADR
Restante 21.25
--------------------------------
Consenso: +3/3
Estado:   >>> COMPRA <<<
```

- **SESGO por temporalidad**: *ALCISTA* = histograma MACD > 0 y RSI > 50; *BAJISTA* = histograma < 0 y RSI < 50; en cualquier otro caso, *neutral*.
- **VOL**: volumen de la vela dividido entre su media de 20 velas (1.6x = 60 % por encima de lo normal).
- **Consenso**: suma de los tres sesgos, de -3 a +3. Cuanto más cerca de ±3, más fuerte.
- **ADR**: cuánto se mueve el instrumento en un día de media. Si hoy ya consumió cerca del 100 %, queda poco recorrido y el indicador bloquea las señales (se puede desactivar).

## Reglas de la señal

**COMPRA** si se cumplen todas estas condiciones:
1. 1m, 5m y 15m están en ALCISTA
2. El volumen de 1m es ≥ media × 1.2
3. El RSI de 1m está por debajo de 70 (sin sobrecompra)
4. El ATR de 1m es ≥ el mínimo configurado (0 = sin filtro)
5. Hoy se ha consumido menos del 100 % del ADR

**VENTA**: las mismas condiciones en sentido contrario (RSI de 1m > 30).

Tras una señal no se repite otra del mismo lado hasta que se pierde la alineación, así se evita llenar el gráfico de flechas.

## Parámetros

Todos se pueden ajustar desde la ventana del indicador: periodos de RSI, MACD, volumen, ATR y ADR, multiplicador de volumen, stop en ATR, relación R:R, filtro de ADR, colores, tamaño del panel y alertas con sonido.

## Uso en estrategias

La serie `Senal` devuelve `1` (compra), `-1` (venta) o `0` (sin señal):

```csharp
if (ConfluenciaMTF().Senal[0] == 1) EnterLong();
```

> ⚠️ Es una herramienta de análisis, no una garantía. Pruébalo en simulación / Market Replay antes de operar con dinero real.

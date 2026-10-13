# Audio 16D

Separa una canción por instrumentos y mueve cada uno en 3D alrededor de quien escucha:
izquierda/derecha y delante/detrás como el **8D**, y además **arriba y abajo**.
La intensidad se mantiene constante en cualquier dirección. Se escucha con **auriculares**.

## Instalación

```bash
cd audio16d
pip install -r requirements.txt
```

## Uso

```bash
python espacial16d.py cancion.mp3                  # separa con Demucs y genera cancion_16d.wav
python espacial16d.py --demo                       # prueba de 20 s: ruido que da vueltas por encima y por debajo
python espacial16d.py cancion.mp3 --sin-separar    # sin Demucs: mueve la mezcla completa
python espacial16d.py --stems carpeta_con_wavs     # usa stems que ya tengas separados
python espacial16d.py cancion.mp3 --ambisonics     # además, 16 canales Ambisonics (AmbiX, 3er orden)
```

| Opción | Qué hace |
|---|---|
| `--modelo htdemucs` | 4 instrumentos (voz, batería, bajo, otros). Por defecto `htdemucs_6s`, que añade guitarra y piano |
| `--velocidad 1.5` | movimiento más rápido |
| `--altura 1.3` | más subida/bajada (`0` = 8D plano) |
| `--realce-elevacion 1.5` | exagera las pistas de arriba/abajo (modelo interno) |
| `--reverb 0.2` | más reverb (ayuda a oír el sonido "fuera de la cabeza") |
| `--sofa hrtf.sofa` | usa una HRTF medida de una cabeza real (más realista). Descargas en sofaconventions.org |
| `--config mov.json` | trayectorias personalizadas |

### Movimiento por defecto

| Instrumento | Movimiento |
|---|---|
| Voz | espiral suave alrededor de la cabeza, sube y baja poco |
| Batería | vuelta vertical: pasa por delante, por encima, por detrás y por debajo |
| Bajo | órbita lenta por debajo del oyente |
| Otros / guitarra / piano | espirales con mucha subida y bajada, desfasadas entre sí |

### Trayectorias personalizadas (`--config`)

```json
{
  "vocals": {"tipo": "fijo", "az": 0, "elev_base": 20},
  "drums":  {"tipo": "vertical", "altura": 70, "periodo_altura": 6},
  "other":  {"tipo": "espiral", "periodo_giro": 8, "sentido": -1, "altura": 60, "periodo_altura": 5}
}
```

Tipos: `fijo`, `orbita`, `vertical`, `espiral`, `vuelta_vertical`.
Ángulos en grados: `az` 0 = delante, 90 = izquierda, 180 = detrás; elevación +90 = encima, -90 = debajo.
Periodos en segundos.

## Cómo funciona

1. **Separación**: Demucs divide la canción en stems.
2. **Trayectorias**: cada stem recibe una posición (azimut, elevación) que cambia con el tiempo.
3. **Binaural**: cada fragmento de 23 ms se filtra con la HRTF de su dirección:
   - diferencia de tiempo y nivel entre oídos (izquierda/derecha),
   - ecos de la oreja (modelo de Brown & Duda), que crean las muescas espectrales con las que el
     cerebro distingue la altura,
   - bandas direccionales de Blauert: más brillo cerca de 8 kHz suena arriba y menos agudos suena abajo.
4. **Intensidad constante**: cada dirección se normaliza a la misma energía, como en el 8D.

La percepción de arriba/abajo con auriculares varía mucho entre personas, porque depende de la forma
de cada oreja. Si con el modelo interno no se nota lo suficiente, prueba `--realce-elevacion 1.5`
o una HRTF `.sofa`.

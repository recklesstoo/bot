"""
espacial16d.py — Convierte una canción en audio "16D".

1. Separa la canción por instrumentos (voz, batería, bajo, otros...) con Demucs.
2. Mueve cada instrumento por una trayectoria en 3D alrededor del oyente:
   izquierda/derecha, delante/detrás y también ARRIBA/ABAJO.
3. Renderiza todo en estéreo binaural (para auriculares) manteniendo la
   intensidad constante, como el efecto 8D, pero con elevación.
4. Opcional: exporta además Ambisonics de 3er orden = 16 canales (AmbiX).

Uso rápido:
    python espacial16d.py cancion.mp3
    python espacial16d.py --demo                    # prueba sin canción
    python espacial16d.py cancion.mp3 --sin-separar # mueve la mezcla completa
    python espacial16d.py cancion.mp3 --sofa hrtf.sofa --ambisonics
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

# =======================
# Constantes
# =======================
RADIO_CABEZA = 0.0875   # metros
VEL_SONIDO   = 343.0    # m/s
FRAME        = 1024     # muestras por bloque de render
HOP          = FRAME // 2

# Ecos de la oreja (pinna) del modelo estructural de Brown & Duda (1998).
# Son los que dan la sensación de arriba/abajo.
PINNA_RHO = np.array([0.5, -1.0, 0.5, -0.25, 0.25])
PINNA_A   = np.array([1.0, 5.0, 5.0, 5.0, 5.0])
PINNA_B   = np.array([2.0, 4.0, 7.0, 11.0, 13.0])
PINNA_D   = np.array([1.0, 0.5, 0.5, 0.5, 0.5])

# Movimiento por defecto de cada instrumento (nombres de stems de Demucs).
# az = azimut (0 = delante, 90 = izquierda), el = elevación (+90 = encima).
TRAYECTORIAS = {
    "vocals": {"tipo": "espiral", "periodo_giro": 20, "altura": 25, "periodo_altura": 10},
    "drums":  {"tipo": "vuelta_vertical", "periodo_giro": 12, "plano": 0},
    "bass":   {"tipo": "orbita", "periodo_giro": 24, "sentido": -1, "elev_base": -35},
    "other":  {"tipo": "espiral", "periodo_giro": 14, "sentido": -1, "altura": 60, "periodo_altura": 7},
    "guitar": {"tipo": "espiral", "periodo_giro": 10, "altura": 45, "periodo_altura": 9, "fase": 120},
    "piano":  {"tipo": "espiral", "periodo_giro": 16, "sentido": -1, "altura": 50, "periodo_altura": 11, "fase": 240},
    "mezcla": {"tipo": "espiral", "periodo_giro": 12, "altura": 60, "periodo_altura": 8},
    "demo":   {"tipo": "vuelta_vertical", "periodo_giro": 10, "plano": 0},
}


def log(msg):
    print(f"[16D] {msg}", flush=True)


# =======================
# Trayectorias
# =======================
def posicion(cfg, t, velocidad=1.0, escala_altura=1.0):
    """Devuelve (azimut, elevación) en grados para los tiempos t (segundos)."""
    t = np.asarray(t, dtype=np.float64) * velocidad
    tipo     = cfg.get("tipo", "espiral")
    periodo  = cfg.get("periodo_giro", 12)
    sentido  = cfg.get("sentido", 1)
    fase     = cfg.get("fase", 0)
    altura   = cfg.get("altura", 45) * escala_altura
    p_altura = cfg.get("periodo_altura", 8)
    base     = cfg.get("elev_base", 0) * escala_altura

    if tipo == "fijo":
        az = np.full_like(t, cfg.get("az", 0.0))
        el = np.full_like(t, base)
    elif tipo == "orbita":          # el 8D clásico: gira alrededor de la cabeza
        az = fase + sentido * 360.0 * t / periodo
        el = np.full_like(t, base)
    elif tipo == "vertical":        # sube y baja sin girar
        az = np.full_like(t, cfg.get("az", 0.0))
        el = base + altura * np.sin(2 * np.pi * t / p_altura + np.radians(fase))
    elif tipo == "espiral":         # gira y a la vez sube y baja
        az = fase + sentido * 360.0 * t / periodo
        el = base + altura * np.sin(2 * np.pi * t / p_altura + np.radians(fase))
    elif tipo == "vuelta_vertical": # pasa por encima de la cabeza y por debajo de los pies
        psi = np.radians(fase) + sentido * 2 * np.pi * t / periodo
        plano = np.radians(cfg.get("plano", 0))
        x = np.cos(psi) * np.cos(plano)
        y = np.cos(psi) * np.sin(plano)
        z = np.sin(psi) * escala_altura
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    else:
        raise ValueError(f"Trayectoria desconocida: {tipo}")
    return np.mod(az + 180, 360) - 180, np.clip(el, -90, 90)


def vector_direccion(az, el):
    """x = delante, y = izquierda, z = arriba."""
    az, el = np.radians(az), np.radians(el)
    return np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)


# =======================
# HRTF (cómo suena una dirección en cada oído)
# =======================
class HRTFParametrica:
    """Modelo de cabeza + oreja (Brown & Duda) con realce de elevación.

    No necesita archivos externos. Las pistas de arriba/abajo vienen de:
      - ecos de la oreja que cambian con la elevación (muescas espectrales),
      - bandas direccionales de Blauert: ~8 kHz suena "arriba",
        perder agudos suena "abajo".
    """
    def __init__(self, sr, intensidad_elevacion=1.0, normalizar=True):
        self.sr = sr
        self.nfft = 2 * FRAME
        self.retardo = 64  # muestras de margen para que el filtro sea causal
        self.k_elev = intensidad_elevacion
        self.normalizar = normalizar
        self.w = 2 * np.pi * np.fft.rfftfreq(self.nfft, 1 / sr)

    def espectros(self, az, el):
        x, y, z = vector_direccion(az, el)
        w = self.w[None, :]
        f = self.w / (2 * np.pi)
        w0 = VEL_SONIDO / RADIO_CABEZA

        # Coordenadas polares interaurales: ángulo polar 0 = delante, 90 = encima, 180 = detrás
        polar = np.degrees(np.arctan2(z, x))
        polar = np.where(polar < -90, polar + 360, polar)
        elev = np.degrees(np.arcsin(np.clip(z, -1, 1)))

        # Ecualización de elevación (igual para ambos oídos)
        arriba = np.sin(np.radians(np.clip(elev, 0, 90)))[:, None]
        abajo = np.sin(np.radians(np.clip(-elev, 0, 90)))[:, None]
        oct8k = np.log2(np.maximum(f, 1.0) / 8000.0)[None, :]
        campana = np.exp(-0.5 * (oct8k / 0.6) ** 2)
        estante = 1 / (1 + np.exp(-(np.log2(np.maximum(f, 1.0) / 4000.0)) * 4))[None, :]
        ganancia_db = self.k_elev * (8.0 * arriba * campana - 9.0 * abajo * estante)
        eq = 10 ** (ganancia_db / 20)

        salida = []
        for lado in (1, -1):  # 1 = oído izquierdo, -1 = derecho
            incidencia = np.arccos(np.clip(lado * y, -1, 1))  # 0 = el sonido apunta a este oído
            # Sombra de la cabeza
            alpha = 1.05 + 0.95 * np.cos(incidencia / np.radians(150) * np.pi)
            sombra = (1 + 1j * alpha[:, None] * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
            # Diferencia de tiempo entre oídos
            itd = np.where(incidencia < np.pi / 2,
                           (RADIO_CABEZA / VEL_SONIDO) * (1 - np.cos(incidencia)),
                           (RADIO_CABEZA / VEL_SONIDO) * (1 + incidencia - np.pi / 2))
            # Ecos de la oreja (retardos en muestras a 44.1 kHz)
            lateral = np.degrees(np.arcsin(np.clip(lado * y, -1, 1)))
            tau = (PINNA_A[None, :] * np.cos(np.radians(lateral / 2))[:, None]
                   * np.sin(np.radians(PINNA_D[None, :] * (90 - polar[:, None])))
                   + PINNA_B[None, :])
            tau = np.maximum(tau, 0.5) / 44100.0
            pinna = 1 + np.einsum("k,fkw->fw", PINNA_RHO,
                                  np.exp(-1j * w[:, None, :] * tau[:, :, None]))
            retardo = np.exp(-1j * w * (itd[:, None] + self.retardo / self.sr))
            salida.append(sombra * pinna * retardo * eq)

        hl, hr = salida
        if self.normalizar:
            # Energía constante en cualquier dirección: misma intensidad, como el 8D
            energia = np.mean(np.abs(hl) ** 2 + np.abs(hr) ** 2, axis=1, keepdims=True)
            g = np.sqrt(2.0 / np.maximum(energia, 1e-12))
            hl, hr = hl * g, hr * g
        return hl, hr


class HRTFSofa:
    """HRTF medida de una persona/maniquí real, en formato SOFA (más realista).
    Descargas gratis en https://www.sofaconventions.org (p. ej. MIT KEMAR, SADIE II)."""
    def __init__(self, ruta, sr, normalizar=True):
        import h5py
        with h5py.File(ruta, "r") as f:
            ir = np.asarray(f["Data.IR"], dtype=np.float64)          # (M, 2, N)
            fs = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
            pos = np.asarray(f["SourcePosition"], dtype=np.float64)  # (M, 3)
            tipo = f["SourcePosition"].attrs.get("Type", b"spherical")
        tipo = tipo.decode() if isinstance(tipo, bytes) else str(tipo)
        if fs != sr:
            g = math.gcd(int(sr), int(fs))
            ir = resample_poly(ir, int(sr) // g, int(fs) // g, axis=2)
        if tipo.lower().startswith("cart"):
            v = pos / np.linalg.norm(pos, axis=1, keepdims=True)
        else:
            v = np.stack(vector_direccion(pos[:, 0], pos[:, 1]), axis=1)
        self.dirs = v
        self.retardo = 0
        self.nfft = 1 << int(np.ceil(np.log2(FRAME + ir.shape[2])))
        hl = np.fft.rfft(ir[:, 0, :], self.nfft)
        hr = np.fft.rfft(ir[:, 1, :], self.nfft)
        if normalizar:
            energia = np.mean(np.abs(hl) ** 2 + np.abs(hr) ** 2, axis=1, keepdims=True)
            g = np.sqrt(2.0 / np.maximum(energia, 1e-12))
            hl, hr = hl * g, hr * g
        self.hl, self.hr = hl, hr
        log(f"HRTF SOFA cargada: {len(v)} direcciones, {ir.shape[2]} muestras")

    def espectros(self, az, el):
        d = np.stack(vector_direccion(az, el), axis=1)
        idx = np.argmax(d @ self.dirs.T, axis=1)  # dirección medida más cercana
        return self.hl[idx], self.hr[idx]


# =======================
# Render
# =======================
def render_binaural(senal, sr, cfg, hrtf, velocidad, escala_altura, bloque=2048):
    """Mueve una señal mono por su trayectoria y devuelve estéreo (N, 2)."""
    n = len(senal)
    nfft = hrtf.nfft
    x = np.concatenate([np.zeros(HOP), senal, np.zeros(FRAME + HOP)])
    nframes = (len(x) - FRAME) // HOP + 1
    ventana = np.hanning(FRAME + 1)[:-1]  # Hann periódica: suma 1 con 50 % de solape
    tramas = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP][:nframes]
    t_centro = (np.arange(nframes) * HOP + FRAME / 2 - HOP) / sr
    az, el = posicion(cfg, t_centro, velocidad, escala_altura)

    salida = np.zeros((len(x) + nfft, 2))
    for i0 in range(0, nframes, bloque):
        i1 = min(i0 + bloque, nframes)
        espectro = np.fft.rfft(tramas[i0:i1] * ventana, nfft)
        hl, hr = hrtf.espectros(az[i0:i1], el[i0:i1])
        yl = np.fft.irfft(espectro * hl, nfft)
        yr = np.fft.irfft(espectro * hr, nfft)
        for k in range(i1 - i0):
            p = (i0 + k) * HOP
            salida[p:p + nfft, 0] += yl[k]
            salida[p:p + nfft, 1] += yr[k]
    inicio = HOP + hrtf.retardo
    return salida[inicio:inicio + n]


def armonicos_ambix(az, el):
    """Armónicos esféricos reales hasta orden 3 (ACN/SN3D) -> 16 canales."""
    a, e = np.radians(az), np.radians(el)
    se, ce = np.sin(e), np.cos(e)
    s3, s38, s58, s15 = math.sqrt(3) / 2, math.sqrt(3 / 8), math.sqrt(5 / 8), math.sqrt(15) / 2
    return np.stack([
        np.ones_like(a),
        np.sin(a) * ce, se, np.cos(a) * ce,
        s3 * np.sin(2 * a) * ce ** 2, s3 * np.sin(a) * np.sin(2 * e), 0.5 * (3 * se ** 2 - 1),
        s3 * np.cos(a) * np.sin(2 * e), s3 * np.cos(2 * a) * ce ** 2,
        s58 * np.sin(3 * a) * ce ** 3, s15 * np.sin(2 * a) * se * ce ** 2,
        s38 * np.sin(a) * ce * (5 * se ** 2 - 1), 0.5 * se * (5 * se ** 2 - 3),
        s38 * np.cos(a) * ce * (5 * se ** 2 - 1), s15 * np.cos(2 * a) * se * ce ** 2,
        s58 * np.cos(3 * a) * ce ** 3,
    ])


def exportar_ambisonics(fuentes, sr, ruta, velocidad, escala_altura, bloque=1 << 16):
    n = max(len(s) for _, s, _ in fuentes)
    with sf.SoundFile(ruta, "w", sr, 16, subtype="FLOAT") as f:
        for i0 in range(0, n, bloque):
            i1 = min(i0 + bloque, n)
            t = np.arange(i0, i1) / sr
            acumulado = np.zeros((16, i1 - i0))
            for _, senal, cfg in fuentes:
                trozo = senal[i0:i1]
                if len(trozo) == 0:
                    continue
                az, el = posicion(cfg, t[:len(trozo)], velocidad, escala_altura)
                acumulado[:, :len(trozo)] += armonicos_ambix(az, el) * trozo
            f.write((acumulado * 0.5).T.astype(np.float32))
    log(f"Ambisonics 3er orden (16 canales AmbiX) -> {ruta}")


def reverb(mono, sr, rt60=1.6, semilla=7):
    """Reverb sintética descorrelada L/R: ayuda a oír el sonido fuera de la cabeza."""
    rng = np.random.default_rng(semilla)
    largo = int(rt60 * sr)
    t = np.arange(largo) / sr
    caida = np.exp(-6.91 * t / rt60)
    pre = int(0.02 * sr)
    salida = []
    for _ in range(2):
        ir = np.concatenate([np.zeros(pre), rng.standard_normal(largo) * caida])
        ir /= np.sqrt(np.sum(ir ** 2))
        salida.append(fftconvolve(mono, ir)[:len(mono)])
    return np.stack(salida, axis=1)


# =======================
# Entrada: separar o cargar
# =======================
def cargar_mono(ruta, sr=None):
    datos, sr_archivo = sf.read(str(ruta), always_2d=True, dtype="float64")
    mono = datos.mean(axis=1)
    if sr and sr_archivo != sr:
        g = math.gcd(int(sr), int(sr_archivo))
        mono = resample_poly(mono, int(sr) // g, int(sr_archivo) // g)
        sr_archivo = sr
    return mono, sr_archivo


def separar_con_demucs(ruta, carpeta, modelo):
    log(f"Separando instrumentos con Demucs ({modelo})... puede tardar unos minutos")
    cmd = [sys.executable, "-m", "demucs", "-n", modelo, "-o", str(carpeta), str(ruta)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("No se encontró Python para ejecutar Demucs.")
    except subprocess.CalledProcessError:
        sys.exit("Demucs falló. ¿Está instalado? pip install demucs  (o usa --sin-separar)")
    return Path(carpeta) / modelo / Path(ruta).stem


def senal_demo(sr, segundos=20):
    """Ráfagas de ruido rosa y clics: lo más fácil de localizar para el oído."""
    rng = np.random.default_rng(1)
    n = int(segundos * sr)
    blanco = rng.standard_normal(n)
    rosa = np.fft.irfft(np.fft.rfft(blanco) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / sr), 20)), n)
    rosa /= np.max(np.abs(rosa))
    t = np.arange(n) / sr
    envolvente = (np.mod(t, 0.25) < 0.12).astype(float)
    return rosa * envolvente * 0.8


# =======================
# Main
# =======================
def main():
    ap = argparse.ArgumentParser(description="Separa una canción por instrumentos y la mueve en 3D (16D).")
    ap.add_argument("cancion", nargs="?", help="archivo de audio (mp3, wav, flac...)")
    ap.add_argument("-o", "--salida", help="archivo de salida (por defecto <cancion>_16d.wav)")
    ap.add_argument("--demo", action="store_true", help="genera una prueba de 20 s sin necesitar canción")
    ap.add_argument("--sin-separar", action="store_true", help="no separa: mueve la mezcla completa")
    ap.add_argument("--stems", help="carpeta con stems ya separados (*.wav), salta Demucs")
    ap.add_argument("--modelo", default="htdemucs_6s",
                    help="modelo Demucs: htdemucs (4 stems) o htdemucs_6s (6, con guitarra y piano)")
    ap.add_argument("--sofa", help="archivo HRTF .sofa (más realista que el modelo interno)")
    ap.add_argument("--velocidad", type=float, default=1.0, help="multiplica la velocidad de movimiento")
    ap.add_argument("--altura", type=float, default=1.0, help="multiplica cuánto suben y bajan (0 = 8D plano)")
    ap.add_argument("--realce-elevacion", type=float, default=1.0,
                    help="exagera las pistas de arriba/abajo del modelo interno (0-2)")
    ap.add_argument("--reverb", type=float, default=0.12, help="cantidad de reverb (0 = nada)")
    ap.add_argument("--config", help="JSON con trayectorias por instrumento (ver README)")
    ap.add_argument("--ambisonics", action="store_true", help="exporta también 16 canales AmbiX")
    ap.add_argument("--sin-normalizar", action="store_true",
                    help="deja que la intensidad cambie con la dirección (por defecto constante)")
    args = ap.parse_args()

    trayectorias = dict(TRAYECTORIAS)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            for nombre, cfg in json.load(f).items():
                trayectorias[nombre] = {**trayectorias.get(nombre, {}), **cfg}

    # 1) Fuentes de sonido
    fuentes = []  # (nombre, señal mono, trayectoria)
    if args.demo:
        sr = 44100
        fuentes.append(("demo", senal_demo(sr), trayectorias["demo"]))
        base = Path("demo")
    else:
        if not args.cancion and not args.stems:
            ap.error("indica una canción, --stems o --demo")
        base = Path(args.cancion or args.stems)
        if args.sin_separar:
            mezcla, sr = cargar_mono(args.cancion)
            fuentes.append(("mezcla", mezcla, trayectorias["mezcla"]))
        else:
            carpeta = Path(args.stems) if args.stems else separar_con_demucs(
                args.cancion, Path(args.cancion).parent / "separado", args.modelo)
            archivos = sorted(carpeta.glob("*.wav"))
            if not archivos:
                sys.exit(f"No hay stems .wav en {carpeta}")
            sr = sf.info(str(archivos[0])).samplerate
            for i, archivo in enumerate(archivos):
                senal, _ = cargar_mono(archivo, sr)
                cfg = trayectorias.get(archivo.stem, {"tipo": "espiral", "fase": 72 * i})
                fuentes.append((archivo.stem, senal, cfg))

    # 2) HRTF
    normalizar = not args.sin_normalizar
    hrtf = HRTFSofa(args.sofa, sr, normalizar) if args.sofa else \
        HRTFParametrica(sr, args.realce_elevacion, normalizar)

    # 3) Render binaural de cada instrumento
    n = max(len(s) for _, s, _ in fuentes)
    mezcla = np.zeros((n, 2))
    seco = np.zeros(n)
    for nombre, senal, cfg in fuentes:
        log(f"Moviendo '{nombre}' ({cfg.get('tipo')})")
        mezcla[:len(senal)] += render_binaural(senal, sr, cfg, hrtf, args.velocidad, args.altura)
        seco[:len(senal)] += senal
    if args.reverb > 0:
        mezcla += args.reverb * reverb(seco, sr)

    pico = np.max(np.abs(mezcla))
    if pico > 0:
        mezcla *= 10 ** (-1 / 20) / pico  # normaliza a -1 dBFS

    salida = Path(args.salida) if args.salida else base.with_name(base.stem + "_16d.wav")
    sf.write(str(salida), mezcla.astype(np.float32), sr, subtype="PCM_24")
    log(f"Listo (escuchar con auriculares) -> {salida}")

    if args.ambisonics:
        exportar_ambisonics(fuentes, sr, salida.with_name(salida.stem + "_ambix16.wav"),
                            args.velocidad, args.altura)


if __name__ == "__main__":
    main()

"""
interfaz16d.py — Ventana fácil de usar para crear audio 16D.

Se abre con doble clic en INICIAR_16D.bat (Windows) o con:
    python interfaz16d.py
"""
import os, queue, subprocess, sys, threading, time, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
    import espacial16d as motor
except ImportError as e:
    raiz = tk.Tk(); raiz.withdraw()
    messagebox.showerror("Audio 16D", "Faltan librerías. Abre INICIAR_16D.bat o ejecuta:\n\n"
                         f"python -m pip install -r requirements.txt\n\n({e})")
    sys.exit(1)

# =======================
# Configuración de la ventana
# =======================
FONDO, PANEL, TEXTO, SUAVE, ACENTO = "#14161c", "#1e212b", "#e8e9ee", "#9aa0ae", "#7c5cff"

INSTRUMENTOS = [  # (nombre del stem, nombre visible, color)
    ("vocals", "Voz", "#ff6b8b"),
    ("drums", "Batería", "#ffb84d"),
    ("bass", "Bajo", "#4dd4ff"),
    ("guitar", "Guitarra", "#7dff8a"),
    ("piano", "Piano", "#d38cff"),
    ("other", "Otros", "#f0f0f0"),
]
MEZCLA = ("mezcla", "Canción completa", "#ff6b8b")

MOVIMIENTOS = {  # nombre visible -> tipo de trayectoria
    "Espiral (gira y sube/baja)": "espiral",
    "Vuelta vertical (encima y debajo)": "vuelta_vertical",
    "Arriba y abajo (sin girar)": "vertical",
    "Órbita 8D clásica (sin altura)": "orbita",
    "Quieto delante": "fijo",
}
NOMBRE_MOVIMIENTO = {v: k for k, v in MOVIMIENTOS.items()}

SEPARACION = {
    "6 instrumentos (voz, batería, bajo, guitarra, piano, otros)": "htdemucs_6s",
    "4 instrumentos (voz, batería, bajo, otros) — más rápido": "htdemucs",
    "No separar: mover la canción completa — lo más rápido": None,
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio 16D")
        self.configure(bg=FONDO)
        self.minsize(900, 640)
        self.mensajes = queue.Queue()
        self.resultado = None
        self.trabajando = False
        self.t0 = time.time()

        self.var_cancion = tk.StringVar()
        self.var_separacion = tk.StringVar(value=list(SEPARACION)[0])
        self.var_velocidad = tk.DoubleVar(value=1.0)
        self.var_altura = tk.DoubleVar(value=1.0)
        self.var_realce = tk.DoubleVar(value=1.0)
        self.var_reverb = tk.DoubleVar(value=0.12)
        self.var_ambisonics = tk.BooleanVar(value=False)
        self.var_mov = {}
        for nombre, _, _ in INSTRUMENTOS + [MEZCLA]:
            tipo = motor.TRAYECTORIAS[nombre]["tipo"]
            self.var_mov[nombre] = tk.StringVar(value=NOMBRE_MOVIMIENTO[tipo])

        self._estilos()
        self._construir()
        motor.SALIDA_LOG = lambda m: self.mensajes.put(("log", m))
        self.after(100, self._revisar_mensajes)
        self.after(40, self._animar)

    # ---------- apariencia ----------
    def _estilos(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=FONDO, foreground=TEXTO, fieldbackground=PANEL, font=("Segoe UI", 10))
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Panel.TLabel", background=PANEL, foreground=TEXTO)
        s.configure("Suave.TLabel", background=PANEL, foreground=SUAVE, font=("Segoe UI", 9))
        s.configure("Titulo.TLabel", background=FONDO, font=("Segoe UI", 20, "bold"))
        s.configure("Sub.TLabel", background=FONDO, foreground=SUAVE)
        s.configure("Seccion.TLabel", background=PANEL, foreground=ACENTO, font=("Segoe UI", 11, "bold"))
        s.configure("TButton", background=PANEL, foreground=TEXTO, borderwidth=1, padding=6)
        s.map("TButton", background=[("active", "#2a2e3b"), ("disabled", FONDO)],
              foreground=[("disabled", "#555")])
        s.configure("Grande.TButton", background=ACENTO, foreground="white",
                    font=("Segoe UI", 12, "bold"), padding=10)
        s.map("Grande.TButton", background=[("active", "#9277ff"), ("disabled", "#3a3456")])
        s.configure("TRadiobutton", background=PANEL, foreground=TEXTO)
        s.map("TRadiobutton", background=[("active", PANEL)])
        s.configure("TCheckbutton", background=PANEL, foreground=TEXTO)
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("TScale", background=PANEL, troughcolor=FONDO)
        s.configure("TCombobox", fieldbackground=FONDO, background=PANEL, foreground=TEXTO, arrowcolor=TEXTO)
        s.map("TCombobox", fieldbackground=[("readonly", FONDO)], foreground=[("readonly", TEXTO)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXTO)
        s.configure("Horizontal.TProgressbar", background=ACENTO, troughcolor=PANEL)

    def _panel(self, padre, titulo):
        marco = ttk.Frame(padre, style="Panel.TFrame", padding=12)
        ttk.Label(marco, text=titulo, style="Seccion.TLabel").pack(anchor="w", pady=(0, 6))
        return marco

    # ---------- ventana ----------
    def _construir(self):
        cabecera = ttk.Frame(self, padding=(16, 12, 16, 4))
        cabecera.pack(fill="x")
        ttk.Label(cabecera, text="Audio 16D", style="Titulo.TLabel").pack(side="left")
        ttk.Label(cabecera, text="   Separa la canción y mueve cada instrumento a tu alrededor: "
                  "izquierda, derecha, arriba y abajo. Escucha con auriculares 🎧",
                  style="Sub.TLabel").pack(side="left", pady=(8, 0))

        cuerpo = ttk.Frame(self, padding=(16, 8))
        cuerpo.pack(fill="both", expand=True)
        izq = ttk.Frame(cuerpo)
        izq.pack(side="left", fill="both", expand=True)
        der = ttk.Frame(cuerpo)
        der.pack(side="left", fill="both", padx=(12, 0))

        # 1. Canción
        p = self._panel(izq, "1. Elige tu canción")
        p.pack(fill="x")
        fila = ttk.Frame(p, style="Panel.TFrame")
        fila.pack(fill="x")
        ttk.Entry(fila, textvariable=self.var_cancion).pack(side="left", fill="x", expand=True, ipady=3)
        ttk.Button(fila, text="Buscar…", command=self._elegir).pack(side="left", padx=(8, 0))
        ttk.Label(p, text="MP3, WAV, FLAC, M4A…", style="Suave.TLabel").pack(anchor="w", pady=(4, 0))

        # 2. Separación
        p = self._panel(izq, "2. ¿Separar por instrumentos?")
        p.pack(fill="x", pady=10)
        for texto in SEPARACION:
            ttk.Radiobutton(p, text=texto, value=texto, variable=self.var_separacion,
                            command=self._actualizar_instrumentos).pack(anchor="w", pady=1)

        # 3. Movimiento por instrumento
        p = self._panel(izq, "3. ¿Cómo se mueve cada instrumento?")
        p.pack(fill="x")
        self.marco_instr = ttk.Frame(p, style="Panel.TFrame")
        self.marco_instr.pack(fill="x")
        self._actualizar_instrumentos()

        # 4. Ajustes
        p = self._panel(izq, "4. Ajustes")
        p.pack(fill="x", pady=10)
        rejilla = ttk.Frame(p, style="Panel.TFrame")
        rejilla.pack(fill="x")
        self._deslizador(rejilla, 0, "Velocidad del movimiento", self.var_velocidad, 0.25, 3.0, "{:.2f}×")
        self._deslizador(rejilla, 1, "Cuánto sube y baja", self.var_altura, 0.0, 1.5, "{:.0%}")
        self._deslizador(rejilla, 2, "Realce arriba/abajo", self.var_realce, 0.0, 2.0, "{:.1f}")
        self._deslizador(rejilla, 3, "Reverb (sala)", self.var_reverb, 0.0, 0.5, "{:.0%}")
        ttk.Checkbutton(p, text="Exportar también 16 canales Ambisonics (para sistemas envolventes)",
                        variable=self.var_ambisonics).pack(anchor="w", pady=(6, 0))

        # Derecha: vista animada + acciones + registro
        p = self._panel(der, "Vista previa del movimiento")
        p.pack(fill="x")
        self.lienzo = tk.Canvas(p, width=400, height=210, bg=FONDO, highlightthickness=0)
        self.lienzo.pack()
        self.leyenda = ttk.Frame(p, style="Panel.TFrame")
        self.leyenda.pack(fill="x", pady=(6, 0))
        self._actualizar_leyenda()

        p = self._panel(der, "5. ¡Listo!")
        p.pack(fill="both", expand=True, pady=10)
        self.btn_crear = ttk.Button(p, text="🎧  Crear audio 16D", style="Grande.TButton", command=self._crear)
        self.btn_crear.pack(fill="x")
        fila = ttk.Frame(p, style="Panel.TFrame")
        fila.pack(fill="x", pady=8)
        self.btn_demo = ttk.Button(fila, text="Probar demo (20 s)", command=self._demo)
        self.btn_demo.pack(side="left", expand=True, fill="x")
        self.btn_escuchar = ttk.Button(fila, text="▶ Escuchar", command=self._escuchar, state="disabled")
        self.btn_escuchar.pack(side="left", expand=True, fill="x", padx=6)
        self.btn_carpeta = ttk.Button(fila, text="📂 Carpeta", command=self._abrir_carpeta, state="disabled")
        self.btn_carpeta.pack(side="left", expand=True, fill="x")
        self.barra = ttk.Progressbar(p, mode="indeterminate")
        self.barra.pack(fill="x", pady=(0, 6))
        self.registro = tk.Text(p, height=9, bg=FONDO, fg=SUAVE, relief="flat", wrap="word",
                                font=("Consolas", 9), state="disabled")
        self.registro.pack(fill="both", expand=True)
        self._escribir("Elige una canción y pulsa «Crear audio 16D». "
                       "O pulsa «Probar demo» para oír el efecto.")

    def _deslizador(self, padre, fila, texto, var, desde, hasta, formato):
        ttk.Label(padre, text=texto, style="Panel.TLabel").grid(row=fila, column=0, sticky="w", pady=3)
        ttk.Scale(padre, from_=desde, to=hasta, variable=var, length=220).grid(row=fila, column=1, padx=10)
        valor = ttk.Label(padre, width=6, style="Suave.TLabel")
        valor.grid(row=fila, column=2, sticky="w")
        actualizar = lambda *_: valor.configure(text=formato.format(var.get()))
        var.trace_add("write", actualizar)
        actualizar()

    def _instrumentos_activos(self):
        modelo = SEPARACION[self.var_separacion.get()]
        if modelo is None:
            return [MEZCLA]
        if modelo == "htdemucs":
            return [i for i in INSTRUMENTOS if i[0] not in ("guitar", "piano")]
        return INSTRUMENTOS

    def _actualizar_instrumentos(self):
        for w in self.marco_instr.winfo_children():
            w.destroy()
        for fila, (nombre, visible, color) in enumerate(self._instrumentos_activos()):
            tk.Label(self.marco_instr, text="●", fg=color, bg=PANEL).grid(row=fila, column=0)
            ttk.Label(self.marco_instr, text=visible, style="Panel.TLabel", width=16).grid(
                row=fila, column=1, sticky="w")
            ttk.Combobox(self.marco_instr, textvariable=self.var_mov[nombre], values=list(MOVIMIENTOS),
                         state="readonly", width=34).grid(row=fila, column=2, sticky="w", pady=2)
        if hasattr(self, "leyenda"):
            self._actualizar_leyenda()

    def _actualizar_leyenda(self):
        for w in self.leyenda.winfo_children():
            w.destroy()
        for nombre, visible, color in self._instrumentos_activos():
            tk.Label(self.leyenda, text=f"● {visible}", fg=color, bg=PANEL,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))

    # ---------- animación ----------
    def _trayectorias(self):
        """Trayectorias elegidas en la ventana, en el formato del motor."""
        resultado = {}
        for nombre, _, _ in INSTRUMENTOS + [MEZCLA]:
            tipo = MOVIMIENTOS[self.var_mov[nombre].get()]
            cfg = dict(motor.TRAYECTORIAS[nombre])
            if tipo != cfg["tipo"]:
                cfg["tipo"] = tipo
                cfg.setdefault("altura", 50)
                cfg.setdefault("periodo_altura", 8)
                if tipo == "fijo":
                    cfg["elev_base"] = 0
            resultado[nombre] = cfg
        return resultado

    def _animar(self):
        c = self.lienzo
        c.delete("all")
        R, cy = 75, 105
        vistas = ((100, "Desde arriba", "delante ↑"), (300, "De lado", "arriba ↑"))
        for cx, titulo, guia in vistas:
            c.create_oval(cx - R, cy - R, cx + R, cy + R, outline="#2c3040", dash=(3, 3))
            c.create_oval(cx - 16, cy - 18, cx + 16, cy + 18, fill="#3a3f52", outline="")
            c.create_text(cx, 12, text=titulo, fill=SUAVE, font=("Segoe UI", 9))
            c.create_text(cx, 198, text=guia, fill="#5c6274", font=("Segoe UI", 8))
        c.create_polygon(100, cy - 26, 94, cy - 17, 106, cy - 17, fill="#3a3f52")  # nariz (arriba)
        c.create_polygon(316 + 8, cy, 316, cy - 6, 316, cy + 6, fill="#3a3f52")  # nariz (lado)

        t = time.time() - self.t0
        trayectorias = self._trayectorias()
        for nombre, _, color in self._instrumentos_activos():
            az, el = motor.posicion(trayectorias[nombre], np.array([t]),
                                    self.var_velocidad.get(), self.var_altura.get())
            x, y, z = (v[0] for v in motor.vector_direccion(az, el))
            # desde arriba: delante hacia arriba en pantalla, izquierda a la izquierda
            px, py = 100 - y * R, cy - x * R
            tam = 5 + 2 * z  # más grande si está encima
            c.create_oval(px - tam, py - tam, px + tam, py + tam, fill=color, outline="")
            # de lado: delante a la derecha, arriba hacia arriba
            px, py = 300 + x * R, cy - z * R
            c.create_oval(px - 6, py - 6, px + 6, py + 6, fill=color, outline="")
        self.after(40, self._animar)

    # ---------- acciones ----------
    def _elegir(self):
        ruta = filedialog.askopenfilename(
            title="Elige una canción",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"), ("Todos", "*.*")])
        if ruta:
            self.var_cancion.set(ruta)

    def _crear(self):
        cancion = self.var_cancion.get().strip().strip('"')
        if not cancion or not Path(cancion).is_file():
            messagebox.showwarning("Audio 16D", "Primero elige una canción con el botón «Buscar…».")
            return
        modelo = SEPARACION[self.var_separacion.get()]
        self._lanzar(dict(cancion=cancion, sin_separar=modelo is None, modelo=modelo or "htdemucs"))

    def _demo(self):
        carpeta = Path(self.var_cancion.get().strip().strip('"')).parent \
            if self.var_cancion.get().strip() else Path.home() / "Music"
        if not carpeta.is_dir():
            carpeta = Path.home()
        self._lanzar(dict(demo=True, salida=str(carpeta / "demo_16d.wav")))

    def _lanzar(self, opciones):
        if self.trabajando:
            return
        opciones.update(velocidad=self.var_velocidad.get(), altura=self.var_altura.get(),
                        realce_elevacion=self.var_realce.get(), cantidad_reverb=self.var_reverb.get(),
                        ambisonics=self.var_ambisonics.get(), trayectorias_extra=self._trayectorias())
        self.trabajando = True
        for b in (self.btn_crear, self.btn_demo, self.btn_escuchar, self.btn_carpeta):
            b.configure(state="disabled")
        self.barra.start(12)
        self._escribir("— Empezando —")
        if not opciones.get("demo") and not opciones["sin_separar"]:
            self._escribir("La separación puede tardar varios minutos (la primera vez también "
                           "descarga el modelo). No cierres la ventana.")

        def trabajo():
            try:
                self.mensajes.put(("fin", motor.procesar(**opciones)))
            except Exception as e:
                traceback.print_exc()
                self.mensajes.put(("error", str(e) or repr(e)))
        threading.Thread(target=trabajo, daemon=True).start()

    def _revisar_mensajes(self):
        try:
            while True:
                tipo, dato = self.mensajes.get_nowait()
                if tipo == "log":
                    self._escribir(dato.replace("[16D] ", ""))
                else:
                    self.trabajando = False
                    self.barra.stop()
                    self.btn_crear.configure(state="normal")
                    self.btn_demo.configure(state="normal")
                    if tipo == "fin":
                        self.resultado = Path(dato).resolve()
                        self.btn_escuchar.configure(state="normal")
                        self.btn_carpeta.configure(state="normal")
                        self._escribir(f"✔ Guardado en: {self.resultado}")
                        if messagebox.askyesno("Audio 16D", "¡Listo! Ponte los auriculares.\n\n"
                                               "¿Quieres escucharlo ahora?"):
                            self._escuchar()
                    else:
                        self._escribir(f"✖ Error: {dato}")
                        messagebox.showerror("Audio 16D", dato)
        except queue.Empty:
            pass
        self.after(100, self._revisar_mensajes)

    def _escribir(self, texto):
        self.registro.configure(state="normal")
        # las barras de progreso de Demucs reemplazan la línea anterior
        if texto.startswith("Separando: ") and self.registro.get("end-2l", "end-1c").startswith("Separando: "):
            self.registro.delete("end-2l", "end-1c")
        self.registro.insert("end", texto + "\n")
        self.registro.see("end")
        self.registro.configure(state="disabled")

    def _abrir(self, ruta):
        if sys.platform == "win32":
            os.startfile(ruta)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", ruta])
        else:
            subprocess.Popen(["xdg-open", ruta])

    def _escuchar(self):
        if self.resultado:
            self._abrir(str(self.resultado))

    def _abrir_carpeta(self):
        if self.resultado:
            self._abrir(str(self.resultado.parent))


if __name__ == "__main__":
    App().mainloop()

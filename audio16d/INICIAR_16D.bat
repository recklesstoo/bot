@echo off
title Audio 16D
cd /d "%~dp0"

where python >nul 2>nul || (
  echo No se encontro Python. Instalalo desde https://www.python.org y marca "Add Python to PATH".
  pause
  exit /b 1
)

python -c "import numpy, scipy, soundfile, demucs" >nul 2>nul || (
  echo Instalando lo necesario por primera vez. Puede tardar varios minutos...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Hubo un error al instalar. Copia el mensaje de arriba y envialo.
    pause
    exit /b 1
  )
)

for /f "usebackq delims=" %%i in (`python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))"`) do set "PYW=%%i"
if exist "%PYW%" (
  start "" "%PYW%" interfaz16d.py
) else (
  python interfaz16d.py
  if errorlevel 1 pause
)

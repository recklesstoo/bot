@echo off
cd /d "%~dp0ai_server"
if not exist ".venv\Scripts\python.exe" (
  echo Primero ejecuta INSTALAR.bat
  pause
  exit /b 1
)
echo Servidor de IA en http://127.0.0.1:8000  (no cierres esta ventana)
".venv\Scripts\python.exe" server.py
pause

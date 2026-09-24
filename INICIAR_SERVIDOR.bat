@echo off
cd /d "%~dp0ai_server"
if not exist ".venv\Scripts\python.exe" (
  echo Primero ejecuta INSTALAR.bat
  pause
  exit /b 1
)
netstat -ano | findstr /R /C:"127.0.0.1:8000 .*LISTENING" /C:"0.0.0.0:8000 .*LISTENING" >nul
if not errorlevel 1 (
  echo El servidor YA esta funcionando en otra ventana. No hace falta abrirlo otra vez.
  echo Comprueba: http://127.0.0.1:8000/health
  pause
  exit /b 0
)
echo Servidor de IA en http://127.0.0.1:8000  (no cierres esta ventana)
".venv\Scripts\python.exe" server.py
pause

@echo off
cd /d "%~dp0ai_server"
if not exist ".venv\Scripts\python.exe" (
  echo Primero ejecuta INSTALAR.bat
  pause
  exit /b 1
)

rem Libera el puerto 8000 si quedo ocupado por un servidor anterior
rem (sesion colgada, ventana cerrada mal, reinicio, etc.).
rem Solo se cierran procesos de Python; si es otro programa, se avisa.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  tasklist /FI "PID eq %%p" /NH | findstr /I "python" >nul
  if not errorlevel 1 (
    echo Cerrando servidor anterior en el puerto 8000 - PID %%p
    taskkill /F /T /PID %%p >nul 2>&1
  ) else (
    echo AVISO: el puerto 8000 lo usa otro programa - PID %%p. Cierralo o cambia PORT en ai_server\.env
  )
)
timeout /t 1 /nobreak >nul

echo Servidor de IA en http://127.0.0.1:8000  (no cierres esta ventana)
".venv\Scripts\python.exe" server.py
pause

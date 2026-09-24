# Instalador del bot de futuros con IA para NinjaTrader 8 (Windows).
# Uso: doble clic en INSTALAR.bat (o: powershell -ExecutionPolicy Bypass -File install.ps1)
$ErrorActionPreference = "Stop"
$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Model = "qwen2.5:7b-instruct"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    AVISO: $msg" -ForegroundColor Yellow }

# 1. Copiar la estrategia a la carpeta de NinjaTrader 8
Step "Copiando la estrategia a NinjaTrader 8"
# MyDocuments respeta Documentos redirigido a OneDrive
$docs = [Environment]::GetFolderPath("MyDocuments")
$nt   = Join-Path $docs "NinjaTrader 8"
if (-not (Test-Path $nt)) {
    throw "No se encontro la carpeta '$nt'. Abre NinjaTrader 8 al menos una vez y vuelve a ejecutar el instalador."
}
$dest = Join-Path $nt "bin\Custom\Strategies"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Force (Join-Path $Root "ninjatrader\AIFuturesTrader.cs") $dest
$target = Join-Path $dest "AIFuturesTrader.cs"
# NinjaTrader compila todo bin\Custom junto: cualquier otra copia de la clase
# provoca errores CS0101/CS0111. Se mueven fuera de Custom como respaldo.
$backup = Join-Path $nt "AIFuturesTrader_backup"
Get-ChildItem (Join-Path $nt "bin\Custom") -Recurse -Filter *.cs |
    Where-Object { $_.FullName -ne $target } |
    Where-Object { Select-String -Path $_.FullName -Pattern "class\s+AIFuturesTrader\b" -Quiet } |
    ForEach-Object {
        New-Item -ItemType Directory -Force -Path $backup | Out-Null
        $bk = Join-Path $backup ("{0}_{1}_{2:yyyyMMddHHmmss}.cs.bak" -f $_.Directory.Name, $_.BaseName, (Get-Date))
        Move-Item -Force $_.FullName $bk
        Warn "Copia duplicada movida: $($_.FullName) -> $bk"
    }
Ok "Copiado a $dest\AIFuturesTrader.cs"

# 2. Python + dependencias del servidor
Step "Preparando el servidor de IA (Python)"
$py = $null
foreach ($c in @("py", "python")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
        try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { $py = $c; break } } catch { }
    }
}
if (-not $py) {
    Warn "Python no esta instalado. Intentando instalarlo con winget..."
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    throw "Python instalado. Cierra esta ventana y vuelve a ejecutar INSTALAR.bat."
}
$srv  = Join-Path $Root "ai_server"
$venv = Join-Path $srv ".venv"
if (-not (Test-Path $venv)) { & $py -m venv $venv }
& (Join-Path $venv "Scripts\python.exe") -m pip install --upgrade pip -q
& (Join-Path $venv "Scripts\python.exe") -m pip install -r (Join-Path $srv "requirements.txt") -q
if ($LASTEXITCODE -ne 0) { throw "Fallo la instalacion de dependencias de Python." }
if (-not (Test-Path (Join-Path $srv ".env"))) {
    Copy-Item (Join-Path $srv ".env.example") (Join-Path $srv ".env")
}
Ok "Servidor listo en $srv"

# 3. Ollama + modelo
Step "Preparando la IA local (Ollama, modelo $Model)"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Warn "Ollama no esta instalado. Instalando con winget..."
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull $Model
    Ok "Modelo $Model descargado"
} else {
    Warn "No se pudo instalar Ollama automaticamente. Descargalo de https://ollama.com/download y ejecuta: ollama pull $Model"
}

Step "Instalacion terminada"
Write-Host "  1. Ejecuta INICIAR_SERVIDOR.bat (dejalo abierto mientras operes)."
Write-Host "  2. En NinjaTrader: New > NinjaScript Editor > Strategies > AIFuturesTrader > pulsa F5 para compilar."
Write-Host "  3. Grafico MNQ 5 minutos > clic derecho > Strategies > AIFuturesTrader > elige cuenta > Enabled."

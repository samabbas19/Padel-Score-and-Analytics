param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvDir = if ($env:PADEL_VENV_DIR) { $env:PADEL_VENV_DIR } else { ".venv" }
$pythonBin = Join-Path $venvDir "Scripts\python.exe"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $pythonBin)) {
    Write-Error "Virtual environment not found at $venvDir. Create it with: py -3.12 -m venv .venv"
}

if (Test-Path -LiteralPath $activateScript) {
    . $activateScript
}

Write-Host "Starting Padel Analytics backend at http://$HostAddress`:$Port"
& $pythonBin -m uvicorn backend.api:app --host $HostAddress --port $Port
exit $LASTEXITCODE

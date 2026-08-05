#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper -- installs all required packages via uv.

.DESCRIPTION
    Creates a local .venv and installs every package listed in
    requirements.txt using uv.  Run this once before running the app.

    Pass -Python to target a specific interpreter (default: 3.10).

.EXAMPLE
    # One-time setup
    .\run.ps1

    # Run the Two-Brain router demo
    .venv\Scripts\python.exe -m two_brain_router

    # Run the test suite
    .venv\Scripts\python.exe -m pytest tests/ -q
#>

[CmdletBinding()]
param(
    [string]$Python = "3.10"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  [INFO]  $msg" }

$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
$VenvDir = Join-Path $PSScriptRoot ".venv"

# 1. Verify requirements.txt exists
if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found: $ReqFile"
    exit 1
}

# 2. Verify uv -- auto-install if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found -- installing ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Refresh PATH so uv is visible in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv install failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    }
}
Write-Ok "uv $(uv --version)"

# 3. Create venv if absent
if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv (Python $Python) ..."
    uv venv "$VenvDir" --python $Python
    if ($LASTEXITCODE -ne 0) { Write-Error "uv venv failed."; exit 1 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists -- skipping creation"
}

# 4. Install dependencies
Write-Info "Installing dependencies from requirements.txt ..."
uv pip install --system-certs --python "$VenvDir\Scripts\python.exe" -r "$ReqFile"
if ($LASTEXITCODE -ne 0) { Write-Error "uv pip install failed."; exit 1 }
Write-Ok "All dependencies installed"

Write-Host ""
Write-Host "  Setup complete.  Run the router demo:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .venv\Scripts\python.exe -m two_brain_router"
Write-Host "    .venv\Scripts\python.exe -m pytest tests/ -q"
Write-Host ""

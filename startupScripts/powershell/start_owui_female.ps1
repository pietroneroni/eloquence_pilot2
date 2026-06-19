# Start Open WebUI SDialog counselor - FEMALE
# Run from PowerShell: .\start_owui_female.ps1

$ErrorActionPreference = "Stop"

# ---- Project / environment ----
$ProjectDir = "C:\Users\Drako\PycharmProjects\Sdialog"
$CondaEnv = "Sdialog"

# ---- Server configuration ----
$env:DIALOG_LANGUAGE = "Italian"
$env:COUNSELOR_MODEL = "ollama:qwen3:30b-thinking"
$env:OWUI_AGENT_PORT = "1334"
$env:OWUI_MODEL_ID = "university-counselor-female:latest"
$env:OWUI_FORCED_GENDER = "female"

# Optional logging level
$env:OWUI_SESSION_LOG_LEVEL = "INFO"

Set-Location $ProjectDir

# Make conda activate work inside a .ps1 script.
$condaHook = (& conda "shell.powershell" "hook") | Out-String
Invoke-Expression $condaHook
conda activate $CondaEnv

Write-Host "Starting FEMALE counselor server..." -ForegroundColor Magenta
Write-Host "Project: $ProjectDir"
Write-Host "Model ID: $env:OWUI_MODEL_ID"
Write-Host "Forced gender: $env:OWUI_FORCED_GENDER"
Write-Host "URL: http://localhost:$env:OWUI_AGENT_PORT/v1"
Write-Host "Open WebUI URL: http://host.docker.internal:$env:OWUI_AGENT_PORT/v1" -ForegroundColor Yellow
Write-Host ""

python -m university_counseling.owui_session_server

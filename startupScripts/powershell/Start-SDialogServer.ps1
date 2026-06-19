<#
Start SDialog/OpenWebUI session server with either:
  - an OpenAI-compatible backend, e.g. openai:gemma-3-27b-it-q8_0
  - an Ollama backend, e.g. ollama:qwen3:30b-thinking

Examples:
  ./Start-SDialogServer.ps1 -Provider openai -Model gemma-3-27b-it-q8_0 -BaseUrl http://127.0.0.1:10007/v1
  ./Start-SDialogServer.ps1 -Provider ollama -Model qwen3:30b-thinking -OllamaHost http://127.0.0.1:11434
  ./Start-SDialogServer.ps1 -Provider ollama -Model qwen3:30b-thinking -Server gender

OpenWebUI should point to:
  Base URL: http://<host>:1333/v1
  API Key: empty or any value
  Model ID: university-counselor:latest
For -Server gender, model IDs are usually:
  university-counselor-male:latest
  university-counselor-female:latest
#>

[CmdletBinding()]
param(
    [ValidateSet('openai','ollama')]
    [string]$Provider = 'openai',

    [string]$Model = 'gemma-3-27b-it-q8_0',
    [string]$BaseUrl = 'http://127.0.0.1:10007/v1',
    [string]$OllamaHost = 'http://127.0.0.1:11434',

    [ValidateSet('single','gender')]
    [string]$Server = 'single',

    [string]$Module = '',
    [string]$ProjectDir = (Join-Path $HOME 'PycharmProjects/Sdialog'),
    [string]$HostName = '0.0.0.0',
    [int]$Port = 1333,
    [string]$Python = '',
    [string]$DialogLanguage = 'Italian',
    [string]$LanceDbDir = '',
    [string]$LanceDbTable = 'universities',
    [string]$OwuiModelId = '',
    [ValidateSet('', 'male', 'female')]
    [string]$ForcedGender = '',
    [string]$OpenAiApiKey = 'kk',
    [switch]$RequireBackend,
    [switch]$SkipBackendCheck
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Module)) {
    if ($Server -eq 'gender') {
        $Module = 'university_counseling.owui_session_server_two_gender_models'
    } else {
        $Module = 'university_counseling.owui_session_server'
    }
}

if ([string]::IsNullOrWhiteSpace($LanceDbDir)) {
    $LanceDbDir = Join-Path $ProjectDir 'RAG_University_Eng/Embeddings/lancedb'
}

if (-not (Test-Path -LiteralPath $ProjectDir -PathType Container)) {
    throw "Project directory not found: $ProjectDir"
}

Set-Location -LiteralPath $ProjectDir

if ([string]::IsNullOrWhiteSpace($Python)) {
    $candidates = @(
        (Join-Path $ProjectDir 'venv/Scripts/python.exe'),
        (Join-Path $ProjectDir 'venv/bin/python3.12'),
        (Join-Path $ProjectDir 'venv/bin/python'),
        'python3.12',
        'python'
    )

    foreach ($candidate in $candidates) {
        if ($candidate -like '*/*' -or $candidate -like '*\*') {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $Python = $candidate
                break
            }
        } else {
            $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($null -ne $cmd) {
                $Python = $candidate
                break
            }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    throw 'Could not find a Python executable. Use -Python to specify one.'
}

# Activate venv when a PowerShell activation script exists. This is optional
# because we also call the venv Python directly when available.
$activateCandidates = @(
    (Join-Path $ProjectDir 'venv/Scripts/Activate.ps1'),
    (Join-Path $ProjectDir 'venv/bin/Activate.ps1'),
    (Join-Path $ProjectDir 'venv/bin/activate.ps1')
)
foreach ($activate in $activateCandidates) {
    if (Test-Path -LiteralPath $activate -PathType Leaf) {
        . $activate
        break
    }
}

$env:OWUI_AGENT_HOST = $HostName
$env:OWUI_AGENT_PORT = [string]$Port
$env:DIALOG_LANGUAGE = $DialogLanguage
$env:SDIALOG_ENABLE_RAG = '1'
$env:SDIALOG_RAG_BACKEND = 'lancedb'
$env:SDIALOG_LANCEDB_DIR = $LanceDbDir
$env:SDIALOG_LANCEDB_TABLE = $LanceDbTable
$env:OPENAI_API_KEY = $OpenAiApiKey
$env:SDIALOG_OPENAI_API_KEY = $OpenAiApiKey

if (-not [string]::IsNullOrWhiteSpace($OwuiModelId)) {
    $env:OWUI_MODEL_ID = $OwuiModelId
}

if (-not [string]::IsNullOrWhiteSpace($ForcedGender)) {
    $env:OWUI_FORCED_GENDER = $ForcedGender
} else {
    Remove-Item Env:OWUI_FORCED_GENDER -ErrorAction SilentlyContinue
}

if ($Model.StartsWith('openai:') -or $Model.StartsWith('ollama:')) {
    $ModelUri = $Model
} elseif ($Provider -eq 'openai') {
    $ModelUri = "openai:$Model"
} else {
    $ModelUri = "ollama:$Model"
}

if ($Provider -eq 'openai') {
    $env:SDIALOG_OPENAI_API_BASE = $BaseUrl.TrimEnd('/')
    $env:OPENAI_API_BASE = $env:SDIALOG_OPENAI_API_BASE
    $env:OPENAI_BASE_URL = $env:SDIALOG_OPENAI_API_BASE
    $checkUrl = "$($env:SDIALOG_OPENAI_API_BASE)/models"
} else {
    $env:OLLAMA_HOST = $OllamaHost.TrimEnd('/')
    $env:SDIALOG_OPENAI_API_BASE = "$($env:OLLAMA_HOST)/v1"
    $env:OPENAI_API_BASE = $env:SDIALOG_OPENAI_API_BASE
    $env:OPENAI_BASE_URL = $env:SDIALOG_OPENAI_API_BASE
    $checkUrl = "$($env:OLLAMA_HOST)/api/tags"
}

$env:SDIALOG_MODEL_URI = $ModelUri
$env:COUNSELOR_MODEL = $ModelUri

if (-not $SkipBackendCheck) {
    try {
        Invoke-WebRequest -Uri $checkUrl -UseBasicParsing -TimeoutSec 3 | Out-Null
        Write-Host "Backend check OK: $checkUrl"
    } catch {
        Write-Warning "Backend check failed: $checkUrl"
        Write-Warning "The SDialog server can start, but chat completions will fail until the LLM backend is reachable."
        if ($RequireBackend) {
            throw "Backend is required but not reachable: $checkUrl"
        }
    }
}

$lanceTablePath = Join-Path $env:SDIALOG_LANCEDB_DIR "$($env:SDIALOG_LANCEDB_TABLE).lance"
if (-not (Test-Path -LiteralPath $env:SDIALOG_LANCEDB_DIR -PathType Container)) {
    Write-Warning "LanceDB directory not found: $($env:SDIALOG_LANCEDB_DIR)"
} elseif (-not (Test-Path -LiteralPath $lanceTablePath)) {
    Write-Warning "LanceDB table not found: $lanceTablePath"
}

Write-Host ''
Write-Host 'Starting SDialog OpenWebUI session server'
Write-Host "  Project:       $ProjectDir"
Write-Host "  Python:        $Python"
Write-Host "  Module:        $Module"
Write-Host "  Bind:          http://$HostName`:$Port"
Write-Host "  Provider:      $Provider"
Write-Host "  Model URI:     $ModelUri"
Write-Host "  LLM base URL:  $($env:OPENAI_BASE_URL)"
Write-Host "  LanceDB:       $lanceTablePath"
Write-Host ''
Write-Host 'OpenWebUI connection:'
Write-Host "  Base URL:      http://<this-host>:$Port/v1"
Write-Host '  API Key:       empty or any value'
Write-Host ''

& $Python -m $Module
exit $LASTEXITCODE

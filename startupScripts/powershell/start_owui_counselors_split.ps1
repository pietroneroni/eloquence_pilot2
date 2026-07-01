<#
Starts two isolated OpenWebUI-compatible counselor servers:
  - ITA on port 1333
  - ENG on port 1334

Run from PowerShell:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\start_owui_counselors_split.ps1

OpenWebUI endpoints:
  http://localhost:1333/v1  -> university-counselor-ITA:latest
  http://localhost:1334/v1  -> university-counselor-ENG:latest
#>

param(
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$PyScript = $(Join-Path $PSScriptRoot "owui_session_server_split.py"),

    [string]$HostAddress = $(if ($env:OWUI_AGENT_HOST) { $env:OWUI_AGENT_HOST } else { "0.0.0.0" }),
    [int]$ItaPort = $(if ($env:OWUI_ITA_PORT) { [int]$env:OWUI_ITA_PORT } else { 1333 }),
    [int]$EngPort = $(if ($env:OWUI_ENG_PORT) { [int]$env:OWUI_ENG_PORT } else { 1334 }),

    # Use the same OpenAI-compatible/Ollama endpoint for both processes by default.
    [string]$OpenAIBase = $(if ($env:SDIALOG_OPENAI_API_BASE) { $env:SDIALOG_OPENAI_API_BASE } elseif ($env:OPENAI_API_BASE) { $env:OPENAI_API_BASE } else { "http://127.0.0.1:10007/v1" }),
    [string]$ItaOpenAIBase = $(if ($env:SDIALOG_OPENAI_API_BASE_ITA) { $env:SDIALOG_OPENAI_API_BASE_ITA } else { $OpenAIBase }),
    [string]$EngOpenAIBase = $(if ($env:SDIALOG_OPENAI_API_BASE_ENG) { $env:SDIALOG_OPENAI_API_BASE_ENG } else { $OpenAIBase }),

    [string]$ItaModel = $(if ($env:COUNSELOR_MODEL_ITA) { $env:COUNSELOR_MODEL_ITA } elseif ($env:COUNSELOR_MODEL) { $env:COUNSELOR_MODEL } elseif ($env:SDIALOG_MODEL_URI) { $env:SDIALOG_MODEL_URI } else { "openai:gemma-3-27b-it-q8_0" }),
    [string]$EngModel = $(if ($env:COUNSELOR_MODEL_ENG) { $env:COUNSELOR_MODEL_ENG } elseif ($env:COUNSELOR_MODEL) { $env:COUNSELOR_MODEL } elseif ($env:SDIALOG_MODEL_URI) { $env:SDIALOG_MODEL_URI } else { "openai:gemma-3-27b-it-q8_0" }),

    [int]$SessionTtlSeconds = $(if ($env:OWUI_SESSION_TTL_SECONDS) { [int]$env:OWUI_SESSION_TTL_SECONDS } else { 3600 }),
    [int]$CleanupIntervalSeconds = $(if ($env:OWUI_SESSION_CLEANUP_INTERVAL_SECONDS) { [int]$env:OWUI_SESSION_CLEANUP_INTERVAL_SECONDS } else { 60 }),
    [int]$MaxPatchHistory = $(if ($env:OWUI_MAX_PATCH_HISTORY) { [int]$env:OWUI_MAX_PATCH_HISTORY } else { 80 }),
    [int]$MaxPatchBuffer = $(if ($env:OWUI_MAX_PATCH_BUFFER) { [int]$env:OWUI_MAX_PATCH_BUFFER } else { 80 }),
    [int]$MaxPatchEvents = $(if ($env:OWUI_MAX_PATCH_EVENTS) { [int]$env:OWUI_MAX_PATCH_EVENTS } else { 80 }),
    [int]$MaxDebugSnapshots = $(if ($env:OWUI_MAX_DEBUG_SNAPSHOTS) { [int]$env:OWUI_MAX_DEBUG_SNAPSHOTS } else { 0 }),

    [int]$TopK = $(if ($env:SDIALOG_COUNSELOR_TOP_K) { [int]$env:SDIALOG_COUNSELOR_TOP_K } else { 6 }),
    [int]$CandidatePoolSize = $(if ($env:SDIALOG_CANDIDATE_POOL_SIZE) { [int]$env:SDIALOG_CANDIDATE_POOL_SIZE } else { 4 }),
    [int]$MaxCtxChars = $(if ($env:SDIALOG_MAX_CTX_CHARS) { [int]$env:SDIALOG_MAX_CTX_CHARS } else { 2200 }),

    [string]$LogDir = $(Join-Path $PSScriptRoot "logs"),
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PyScript)) {
    throw "Python server script not found: $PyScript"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Quote-ForPowerShellCommand {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Start-CounselorProcess {
    param(
        [Parameter(Mandatory=$true)][ValidateSet("ITA", "ENG")][string]$Language,
        [Parameter(Mandatory=$true)][int]$Port,
        [Parameter(Mandatory=$true)][string]$BaseUrl,
        [Parameter(Mandatory=$true)][string]$Model,
        [Parameter(Mandatory=$true)][string]$LegacyTarget
    )

    $stdout = Join-Path $LogDir ("counselor_{0}_{1}.out.log" -f $Language.ToLowerInvariant(), $Port)
    $stderr = Join-Path $LogDir ("counselor_{0}_{1}.err.log" -f $Language.ToLowerInvariant(), $Port)

    $pythonQ = Quote-ForPowerShellCommand $PythonBin
    $scriptQ = Quote-ForPowerShellCommand $PyScript

    $command = @"
`$env:OWUI_PROCESS_MODEL = '$Language'
`$env:OWUI_AGENT_PORT = '$Port'
`$env:OWUI_AGENT_HOST = '$HostAddress'
`$env:OWUI_LEGACY_MODEL_TARGET = '$LegacyTarget'
`$env:SDIALOG_OPENAI_API_BASE = '$BaseUrl'
`$env:OPENAI_API_BASE = '$BaseUrl'
`$env:OPENAI_BASE_URL = '$BaseUrl'
`$env:COUNSELOR_MODEL = '$Model'
`$env:SDIALOG_MODEL_URI = '$Model'
`$env:OWUI_SESSION_TTL_SECONDS = '$SessionTtlSeconds'
`$env:OWUI_SESSION_CLEANUP_INTERVAL_SECONDS = '$CleanupIntervalSeconds'
`$env:OWUI_MAX_PATCH_HISTORY = '$MaxPatchHistory'
`$env:OWUI_MAX_PATCH_BUFFER = '$MaxPatchBuffer'
`$env:OWUI_MAX_PATCH_EVENTS = '$MaxPatchEvents'
`$env:OWUI_MAX_DEBUG_SNAPSHOTS = '$MaxDebugSnapshots'
`$env:SDIALOG_DEBUG = '0'
`$env:SDIALOG_COUNSELOR_TOP_K = '$TopK'
`$env:SDIALOG_CANDIDATE_POOL_SIZE = '$CandidatePoolSize'
`$env:SDIALOG_MAX_CTX_CHARS = '$MaxCtxChars'
& $pythonQ $scriptQ
"@

    $launcher = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell.exe" }
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command)

    $proc = Start-Process -FilePath $launcher `
        -ArgumentList $args `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden

    Write-Host ("{0} server started: pid={1}, port={2}, model={3}, base={4}" -f $Language, $proc.Id, $Port, $Model, $BaseUrl)
    Write-Host ("  logs: {0} | {1}" -f $stdout, $stderr)
    return $proc
}

$processes = @()
try {
    $processes += Start-CounselorProcess -Language "ITA" -Port $ItaPort -BaseUrl $ItaOpenAIBase -Model $ItaModel -LegacyTarget "ITA"
    $processes += Start-CounselorProcess -Language "ENG" -Port $EngPort -BaseUrl $EngOpenAIBase -Model $EngModel -LegacyTarget "ENG"

    Write-Host ""
    Write-Host "OpenWebUI endpoints:"
    Write-Host ("  ITA: http://localhost:{0}/v1  -> university-counselor-ITA:latest" -f $ItaPort)
    Write-Host ("  ENG: http://localhost:{0}/v1  -> university-counselor-ENG:latest" -f $EngPort)
    Write-Host ""
    Write-Host "Health checks:"
    Write-Host ("  Invoke-RestMethod http://127.0.0.1:{0}/health" -f $ItaPort)
    Write-Host ("  Invoke-RestMethod http://127.0.0.1:{0}/health" -f $EngPort)

    if ($NoWait) {
        Write-Host ""
        Write-Host "NoWait enabled. Processes remain running in the background."
        return
    }

    Write-Host ""
    Write-Host "Press Ctrl+C to stop both servers."
    while ($true) {
        foreach ($p in $processes) {
            if ($p.HasExited) {
                throw "A counselor server exited unexpectedly. Check logs in $LogDir"
            }
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    if (-not $NoWait) {
        foreach ($p in $processes) {
            try {
                if ($p -and -not $p.HasExited) {
                    Write-Host ("Stopping pid={0}" -f $p.Id)
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                }
            } catch {}
        }
    }
}

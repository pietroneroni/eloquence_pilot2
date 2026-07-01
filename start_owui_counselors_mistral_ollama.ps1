<#
Starts two isolated OpenWebUI-compatible counselor servers using the direct SDialog Ollama backend by default:
  - ITA on port 1333
  - ENG on port 1334

Default SDialog model:
  ollama:qwen3:30b-thinking

This launcher starts the Python server as a module:
  python -m university_counseling.owui_session_server

Run from the project root, for example:
  cd C:\Users\Drako\PycharmProjects\Sdialog
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\start_owui_counselors_ollama_direct_module.ps1

OpenWebUI endpoints:
  http://localhost:1333/v1  -> university-counselor-ITA:latest
  http://localhost:1334/v1  -> university-counselor-ENG:latest
#>

param(
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$ProjectRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { $PSScriptRoot }),
    [string]$ModuleName = $(if ($env:OWUI_SERVER_MODULE) { $env:OWUI_SERVER_MODULE } else { "university_counseling.owui_session_server" }),

    [string]$HostAddress = $(if ($env:OWUI_AGENT_HOST) { $env:OWUI_AGENT_HOST } else { "0.0.0.0" }),
    [int]$ItaPort = $(if ($env:OWUI_ITA_PORT) { [int]$env:OWUI_ITA_PORT } else { 1333 }),
    [int]$EngPort = $(if ($env:OWUI_ENG_PORT) { [int]$env:OWUI_ENG_PORT } else { 1334 }),

    # Direct Ollama. This is not an OpenAI/OpenAI-compatible base URL.
    [string]$OllamaHost = $(if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "127.0.0.1:11434" }),

    [string]$ItaModel = $(if ($env:COUNSELOR_MODEL_ITA) { $env:COUNSELOR_MODEL_ITA } else { "ollama:qwen3:30b-thinking" }),
    [string]$EngModel = $(if ($env:COUNSELOR_MODEL_ENG) { $env:COUNSELOR_MODEL_ENG } else { "ollama:qwen3:30b-thinking" }),

    [int]$SessionTtlSeconds = $(if ($env:OWUI_SESSION_TTL_SECONDS) { [int]$env:OWUI_SESSION_TTL_SECONDS } else { 3600 }),
    [int]$CleanupIntervalSeconds = $(if ($env:OWUI_SESSION_CLEANUP_INTERVAL_SECONDS) { [int]$env:OWUI_SESSION_CLEANUP_INTERVAL_SECONDS } else { 60 }),
    [int]$MaxPatchHistory = $(if ($env:OWUI_MAX_PATCH_HISTORY) { [int]$env:OWUI_MAX_PATCH_HISTORY } else { 80 }),
    [int]$MaxPatchBuffer = $(if ($env:OWUI_MAX_PATCH_BUFFER) { [int]$env:OWUI_MAX_PATCH_BUFFER } else { 80 }),
    [int]$MaxPatchEvents = $(if ($env:OWUI_MAX_PATCH_EVENTS) { [int]$env:OWUI_MAX_PATCH_EVENTS } else { 80 }),
    [int]$MaxDebugSnapshots = $(if ($env:OWUI_MAX_DEBUG_SNAPSHOTS) { [int]$env:OWUI_MAX_DEBUG_SNAPSHOTS } else { 0 }),

    [int]$TopK = $(if ($env:SDIALOG_COUNSELOR_TOP_K) { [int]$env:SDIALOG_COUNSELOR_TOP_K } else { 6 }),
    [int]$CandidatePoolSize = $(if ($env:SDIALOG_CANDIDATE_POOL_SIZE) { [int]$env:SDIALOG_CANDIDATE_POOL_SIZE } else { 4 }),
    [int]$MaxCtxChars = $(if ($env:SDIALOG_MAX_CTX_CHARS) { [int]$env:SDIALOG_MAX_CTX_CHARS } else { 2200 }),

    [string]$LogDir = $(Join-Path $ProjectRoot "logs"),
    [switch]$SkipOllamaCheck,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

$packageDir = Join-Path $ProjectRoot "university_counseling"
if (-not (Test-Path $packageDir)) {
    throw "Package folder not found: $packageDir"
}

$serverFile = Join-Path $packageDir "owui_session_server.py"
if (-not (Test-Path $serverFile)) {
    throw "Python server module file not found: $serverFile"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Quote-ForPowerShellCommand {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Get-OllamaApiRoot {
    param([string]$HostValue)
    $h = $HostValue.Trim()
    if ($h -notmatch '^https?://') {
        $h = "http://$h"
    }
    return $h.TrimEnd('/')
}

function Get-OllamaModelName {
    param([string]$SdialogModel)
    if ($SdialogModel.StartsWith("ollama:")) {
        return $SdialogModel.Substring("ollama:".Length)
    }
    return $SdialogModel
}

function Test-OllamaDirect {
    param([string]$HostValue, [string[]]$RequiredModels)

    Write-Host "Ollama check:"

    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host ("  ollama.exe: {0}" -f $cmd.Source)
    } else {
        Write-Warning "  ollama command not found in PATH. If Ollama is running, the API check may still work."
    }

    $apiRoot = Get-OllamaApiRoot $HostValue
    try {
        $version = Invoke-RestMethod -Uri "$apiRoot/api/version" -TimeoutSec 5
        Write-Host ("  API: {0}/api/version -> OK, version={1}" -f $apiRoot, $version.version)
    } catch {
        Write-Warning ("  Cannot reach Ollama API at {0}/api/version" -f $apiRoot)
        Write-Warning "  Start Ollama first, for example: ollama serve"
        return
    }

    try {
        $tags = Invoke-RestMethod -Uri "$apiRoot/api/tags" -TimeoutSec 10
        $names = @()
        if ($tags.models) {
            $names = @($tags.models | ForEach-Object { $_.name })
        }
        foreach ($m in $RequiredModels) {
            $plain = Get-OllamaModelName $m
            if ($names -contains $plain) {
                Write-Host ("  model present: {0}" -f $plain)
            } else {
                Write-Warning ("  model not listed: {0}. Run: ollama pull {0}" -f $plain)
            }
        }
    } catch {
        Write-Warning "  Could not read /api/tags. Continuing."
    }
}

function Start-CounselorProcess {
    param(
        [Parameter(Mandatory=$true)][ValidateSet("ITA", "ENG")][string]$Language,
        [Parameter(Mandatory=$true)][int]$Port,
        [Parameter(Mandatory=$true)][string]$Model,
        [Parameter(Mandatory=$true)][string]$LegacyTarget
    )

    $stdout = Join-Path $LogDir ("counselor_{0}_{1}.out.log" -f $Language.ToLowerInvariant(), $Port)
    $stderr = Join-Path $LogDir ("counselor_{0}_{1}.err.log" -f $Language.ToLowerInvariant(), $Port)

    $pythonQ = Quote-ForPowerShellCommand $PythonBin
    $moduleQ = Quote-ForPowerShellCommand $ModuleName
    $projectRootQ = Quote-ForPowerShellCommand $ProjectRoot

    $command = @"
Set-Location $projectRootQ
`$env:PYTHONPATH = '$ProjectRoot' + [System.IO.Path]::PathSeparator + `$env:PYTHONPATH
`$env:OWUI_PROCESS_MODEL = '$Language'
`$env:OWUI_AGENT_PORT = '$Port'
`$env:OWUI_AGENT_HOST = '$HostAddress'
`$env:OWUI_LEGACY_MODEL_TARGET = '$LegacyTarget'
`$env:OLLAMA_HOST = '$OllamaHost'
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
& $pythonQ -m $moduleQ
"@

    $launcher = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell.exe" }
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command)

    $proc = Start-Process -FilePath $launcher `
        -ArgumentList $args `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden

    Write-Host ("{0} server started: pid={1}, port={2}, model={3}" -f $Language, $proc.Id, $Port, $Model)
    Write-Host ("  logs: {0} | {1}" -f $stdout, $stderr)
    return $proc
}

if (-not $SkipOllamaCheck) {
    Test-OllamaDirect -HostValue $OllamaHost -RequiredModels @($ItaModel, $EngModel)
    Write-Host ""
}

$processes = @()
try {
    $processes += Start-CounselorProcess -Language "ITA" -Port $ItaPort -Model $ItaModel -LegacyTarget "ITA"
    $processes += Start-CounselorProcess -Language "ENG" -Port $EngPort -Model $EngModel -LegacyTarget "ENG"

    Start-Sleep -Seconds 2
    foreach ($p in $processes) {
        if ($p.HasExited) {
            Write-Host ""
            Write-Warning "A server exited immediately. Recent error logs:"
            Get-ChildItem $LogDir -Filter "*.err.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 4 | ForEach-Object {
                Write-Host ("--- {0} ---" -f $_.FullName)
                Get-Content $_.FullName -Tail 40 -ErrorAction SilentlyContinue
            }
            throw "A counselor server exited immediately. Check logs in $LogDir"
        }
    }

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

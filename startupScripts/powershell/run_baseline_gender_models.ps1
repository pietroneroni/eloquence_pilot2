<#
Run baseline gender/model experiment for SDialog.

Place this script in either:
  - project root, or
  - startupScripts/

Examples:
  .\startupScripts\run_baseline_gender_models_fixed.ps1 -Mode Pilot
  .\startupScripts\run_baseline_gender_models_fixed.ps1 -Mode Full

Pilot = first 5 personas, Mistral only, male/female.
Full  = all baseline personas, Mistral + Salamandra, male/female.
#>

param(
    [ValidateSet("Pilot", "Full")]
    [string]$Mode = "Pilot",

    [string]$InputDir = "PERSONAS_DATASET_baseline_02_riasec_l2_only",
    [string]$ExperimentName = "baseline_02_riasec_l2_only",

    [string]$MistralModel = "ollama:mistral-small3.1:24b",
    [string]$SalamandraModel = "ollama:salamandra-7b-instruct",
    [string]$StudentModel = "ollama:qwen3:30b",

    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Set-EnvVar {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    Write-Host "$Name=$Value"
}

function Require-Path {
    param([Parameter(Mandatory = $true)][string]$PathToCheck)
    if (-not (Test-Path -Path $PathToCheck)) {
        throw "Required path not found: $PathToCheck"
    }
}

# Resolve project root robustly.
# If launched from project root, use current directory.
# If launched from startupScripts, use parent directory.
$LaunchDir = (Get-Location).Path
$ScriptDir = Split-Path -Parent $PSCommandPath
$CandidateRoots = @(
    $LaunchDir,
    (Split-Path -Parent $ScriptDir),
    $ScriptDir
) | Select-Object -Unique

$ProjectRoot = $null
foreach ($candidate in $CandidateRoots) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
    $candidateScript = Join-Path $candidate "university_counseling\SdialogExample.py"
    $candidatePackage = Join-Path $candidate "university_counseling\__init__.py"
    if ((Test-Path $candidateScript) -and (Test-Path $candidatePackage)) {
        $ProjectRoot = $candidate
        break
    }
}

if (-not $ProjectRoot) {
    throw "Could not find project root. Expected university_counseling\SdialogExample.py and university_counseling\__init__.py. Current dir: $LaunchDir; script dir: $ScriptDir"
}

Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"

$SdialogExample = Join-Path $ProjectRoot "university_counseling\SdialogExample.py"
$ResolvedInputDir = Join-Path $ProjectRoot $InputDir

Require-Path $SdialogExample
Require-Path $ResolvedInputDir

# Clean/evaluation defaults.
Set-EnvVar "SDIALOG_DEBUG" "0"
Set-EnvVar "SDIALOG_LOG_ORCHESTRATION" "0"
Set-EnvVar "SDIALOG_LOG_THINKING" "0"
Set-EnvVar "SDIALOG_EXPERT_THINK" "0"
Set-EnvVar "SDIALOG_EXPOSE_SENSITIVE_CONTEXT" "0"
Set-EnvVar "SDIALOG_ADD_SYNTHETIC_SCHOOL_TYPE" "0"

# Experiment constants.
Set-EnvVar "SDIALOG_INPUT_DIR" $InputDir
Set-EnvVar "SDIALOG_EXPERIMENT_NAME" $ExperimentName
Set-EnvVar "SDIALOG_GENDER_VARIANTS" "male,female"
Set-EnvVar "SDIALOG_STUDENT_MODEL" $StudentModel
Set-EnvVar "SDIALOG_STUDENT_MODEL_SAME_AS_EXPERT" "0"

# RIASEC policy: keep inferred-only behavior.
# Do not set any variable that injects source RIASEC values.

if ($Mode -eq "Pilot") {
    Set-EnvVar "SDIALOG_MAX_PERSONAS" "5"
    Set-EnvVar "SDIALOG_MODEL_CONDITIONS" "mistral_small_3_1_24b=$MistralModel"
}
else {
    Set-EnvVar "SDIALOG_MAX_PERSONAS" "all"
    Set-EnvVar "SDIALOG_MODEL_CONDITIONS" "mistral_small_3_1_24b=$MistralModel;salamandra_7b=$SalamandraModel"
}

Write-Host ""
Write-Host "Running SDialog experiment..."
Write-Host "Mode: $Mode"
Write-Host "Script: $SdialogExample"
Write-Host "Input:  $ResolvedInputDir"
Write-Host ""

# Run as module so relative imports inside university_counseling work reliably.
& $PythonExe -m university_counseling.SdialogExample

if ($LASTEXITCODE -ne 0) {
    throw "Python run failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Check output under: Generated Dialogs\$ExperimentName"

<#
Run the offline SDialog gender experiment with a local Ollama model.

Typical usage:
  .\run_baseline_gender_models.ps1
  .\run_baseline_gender_models.ps1 -CounselorModel "ollama:gemma-4-31B-it" -ModelLabel "gemma_4_31b"
  .\run_baseline_gender_models.ps1 -Mode Full -CounselorModel "ollama:llama3.1:8b" -ModelLabel "llama31_8b"

By default the student uses the same model as the counselor. Pass -StudentModel
only if you intentionally want a separate simulated-student model.
#>

param(
    [ValidateSet("Pilot", "Full")]
    [string]$Mode = "Pilot",

    [string]$CounselorModel = "ollama:gemma-4-31B-it",
    [string]$ModelLabel = "gemma_4_31b",
    [string]$StudentModel = "",

    [string]$InputDir = "PERSONAS_DATASET_baseline_02_riasec_l2_only",
    [string]$ExperimentName = "baseline_gender_ollama",
    [string]$GenderVariants = "male,female",
    [string]$DialogLanguage = "Italian",

    [int]$PilotPersonas = 5,
    [int]$MaxTurns = 100,
    [string]$OllamaHost = "http://127.0.0.1:11434",
    [string]$PythonExe = "python",

    [string[]]$ExtraModelConditions = @()
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

function Normalize-ModelLabel {
    param([Parameter(Mandatory = $true)][string]$Value)
    $label = $Value -replace "[^A-Za-z0-9_.-]+", "_"
    $label = $label.Trim("_")
    if ([string]::IsNullOrWhiteSpace($label)) {
        return "model"
    }
    return $label
}

$LaunchDir = (Get-Location).Path
$ScriptDir = Split-Path -Parent $PSCommandPath
$CandidateRoots = @(
    $LaunchDir,
    (Split-Path -Parent $ScriptDir),
    $ScriptDir
) | Select-Object -Unique

$ProjectRoot = $null
foreach ($candidate in $CandidateRoots) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        continue
    }
    $candidateScript = Join-Path $candidate "university_counseling\SdialogExample.py"
    $candidatePackage = Join-Path $candidate "university_counseling\__init__.py"
    if ((Test-Path $candidateScript) -and (Test-Path $candidatePackage)) {
        $ProjectRoot = $candidate
        break
    }
}

if (-not $ProjectRoot) {
    throw "Could not find project root. Expected university_counseling\SdialogExample.py and university_counseling\__init__.py."
}

Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"

$SdialogExample = Join-Path $ProjectRoot "university_counseling\SdialogExample.py"
$ResolvedInputDir = Join-Path $ProjectRoot $InputDir
Require-Path $SdialogExample
Require-Path $ResolvedInputDir

$CleanModelLabel = Normalize-ModelLabel $ModelLabel
$ModelConditions = @("$CleanModelLabel=$CounselorModel")
foreach ($condition in $ExtraModelConditions) {
    $trimmed = ($condition | Out-String).Trim()
    if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $ModelConditions += $trimmed
    }
}

$UseSameStudentModel = [string]::IsNullOrWhiteSpace($StudentModel)
if ($UseSameStudentModel) {
    $StudentModel = $CounselorModel
}

Set-EnvVar "OLLAMA_HOST" $OllamaHost

Set-EnvVar "SDIALOG_INPUT_DIR" $InputDir
Set-EnvVar "SDIALOG_EXPERIMENT_NAME" $ExperimentName
Set-EnvVar "SDIALOG_DIALOG_LANGUAGE" $DialogLanguage
Set-EnvVar "SDIALOG_MODEL_CONDITIONS" ($ModelConditions -join ";")
Set-EnvVar "SDIALOG_STUDENT_MODEL" $StudentModel
Set-EnvVar "SDIALOG_STUDENT_MODEL_SAME_AS_EXPERT" ($(if ($UseSameStudentModel) { "1" } else { "0" }))
Set-EnvVar "SDIALOG_GENDER_VARIANTS" $GenderVariants
Set-EnvVar "SDIALOG_MAX_TURNS" ([string]$MaxTurns)

if ($Mode -eq "Pilot") {
    Set-EnvVar "SDIALOG_MAX_PERSONAS" ([string]$PilotPersonas)
}
else {
    Set-EnvVar "SDIALOG_MAX_PERSONAS" "all"
}

if ($DialogLanguage.Trim().ToLower().StartsWith("it")) {
    Set-EnvVar "SDIALOG_RAG_LANG" "ita"
}
else {
    Set-EnvVar "SDIALOG_RAG_LANG" "eng"
}

# Evaluation defaults: keep logs clean and keep the gender condition hidden from
# the visible student persona.
Set-EnvVar "SDIALOG_DEBUG" "0"
Set-EnvVar "SDIALOG_LOG_ORCHESTRATION" "0"
Set-EnvVar "SDIALOG_LOG_THINKING" "0"
Set-EnvVar "SDIALOG_EXPERT_THINK" "0"
Set-EnvVar "SDIALOG_STUDENT_THINK" "0"
Set-EnvVar "SDIALOG_EXPOSE_SENSITIVE_CONTEXT" "0"
Set-EnvVar "SDIALOG_ALLOW_FORCED_LISTENER_GENDER" "1"
Set-EnvVar "SDIALOG_HIDE_VISIBLE_GENDER_CUES" "1"
Set-EnvVar "SDIALOG_USE_GENDERED_NAMES" "0"
Set-EnvVar "SDIALOG_ADD_SYNTHETIC_SCHOOL_TYPE" "0"

Write-Host ""
Write-Host "Running SDialog gender experiment"
Write-Host "Mode:      $Mode"
Write-Host "Counselor: $CounselorModel"
Write-Host "Student:   $StudentModel"
Write-Host "Input:     $ResolvedInputDir"
Write-Host "Output:    Generated Dialogs\$ExperimentName"
Write-Host ""

& $PythonExe -m university_counseling.SdialogExample

if ($LASTEXITCODE -ne 0) {
    throw "Python run failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Manifest: Generated Dialogs\$ExperimentName\run_manifest.csv"

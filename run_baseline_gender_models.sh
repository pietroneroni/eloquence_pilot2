#!/usr/bin/env bash
#
# Run the offline SDialog gender experiment with a local Ollama model.
#
# Typical usage:
#   ./run_baseline_gender_models.sh
#   ./run_baseline_gender_models.sh -CounselorModel "ollama:gemma-4-31B-it" -ModelLabel "gemma_4_31b"
#   ./run_baseline_gender_models.sh -Mode Full -CounselorModel "ollama:llama3.1:8b" -ModelLabel "llama31_8b"
#
# By default the student uses the same model as the counselor. Pass -StudentModel
# only if you intentionally want a separate simulated-student model.

set -euo pipefail

Mode="Pilot"
CounselorModel="ollama:gemma-4-31B-it"
ModelLabel="gemma_4_31b"
StudentModel=""

InputDir="PERSONAS_DATASET_baseline_02_riasec_l2_only"
ExperimentName="baseline_gender_ollama"
GenderVariants="male,female"
DialogLanguage="Italian"

PilotPersonas=5
MaxTurns=100
OllamaHost="http://127.0.0.1:11434"
PythonExe="python"

ExtraModelConditions=()

usage() {
    cat <<EOF
Usage:
  ./run_baseline_gender_models.sh [options]

Options:
  -Mode Pilot|Full
  -CounselorModel VALUE
  -ModelLabel VALUE
  -StudentModel VALUE
  -InputDir VALUE
  -ExperimentName VALUE
  -GenderVariants VALUE
  -DialogLanguage VALUE
  -PilotPersonas VALUE
  -MaxTurns VALUE
  -OllamaHost VALUE
  -PythonExe VALUE
  -ExtraModelConditions VALUE

Examples:
  ./run_baseline_gender_models.sh
  ./run_baseline_gender_models.sh -CounselorModel "ollama:gemma-4-31B-it" -ModelLabel "gemma_4_31b"
  ./run_baseline_gender_models.sh -Mode Full -CounselorModel "ollama:llama3.1:8b" -ModelLabel "llama31_8b"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -Mode|--Mode)
            Mode="${2:?Missing value for $1}"
            shift 2
            ;;
        -CounselorModel|--CounselorModel)
            CounselorModel="${2:?Missing value for $1}"
            shift 2
            ;;
        -ModelLabel|--ModelLabel)
            ModelLabel="${2:?Missing value for $1}"
            shift 2
            ;;
        -StudentModel|--StudentModel)
            StudentModel="${2:?Missing value for $1}"
            shift 2
            ;;
        -InputDir|--InputDir)
            InputDir="${2:?Missing value for $1}"
            shift 2
            ;;
        -ExperimentName|--ExperimentName)
            ExperimentName="${2:?Missing value for $1}"
            shift 2
            ;;
        -GenderVariants|--GenderVariants)
            GenderVariants="${2:?Missing value for $1}"
            shift 2
            ;;
        -DialogLanguage|--DialogLanguage)
            DialogLanguage="${2:?Missing value for $1}"
            shift 2
            ;;
        -PilotPersonas|--PilotPersonas)
            PilotPersonas="${2:?Missing value for $1}"
            shift 2
            ;;
        -MaxTurns|--MaxTurns)
            MaxTurns="${2:?Missing value for $1}"
            shift 2
            ;;
        -OllamaHost|--OllamaHost)
            OllamaHost="${2:?Missing value for $1}"
            shift 2
            ;;
        -PythonExe|--PythonExe)
            PythonExe="${2:?Missing value for $1}"
            shift 2
            ;;
        -ExtraModelConditions|--ExtraModelConditions)
            ExtraModelConditions+=("${2:?Missing value for $1}")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$Mode" != "Pilot" && "$Mode" != "Full" ]]; then
    echo "Invalid Mode: $Mode. Allowed values: Pilot, Full" >&2
    exit 1
fi

set_env_var() {
    local name="$1"
    local value="$2"

    export "$name=$value"
    echo "$name=$value"
}

require_path() {
    local path_to_check="$1"

    if [[ ! -e "$path_to_check" ]]; then
        echo "Required path not found: $path_to_check" >&2
        exit 1
    fi
}

trim_string() {
    local value="$1"
    printf '%s' "$value" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

normalize_model_label() {
    local value="$1"
    local label

    label="$(printf '%s' "$value" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^_+//; s/_+$//')"

    if [[ -z "$(trim_string "$label")" ]]; then
        printf '%s\n' "model"
    else
        printf '%s\n' "$label"
    fi
}

join_by_semicolon() {
    local IFS=";"
    echo "$*"
}

LaunchDir="$(pwd -P)"
ScriptDir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ParentScriptDir="$(cd -- "$ScriptDir/.." && pwd -P)"

CandidateRoots=(
    "$LaunchDir"
    "$ParentScriptDir"
    "$ScriptDir"
)

ProjectRoot=""

for candidate in "${CandidateRoots[@]}"; do
    if [[ -z "$(trim_string "$candidate")" ]]; then
        continue
    fi

    candidate_script="$candidate/university_counseling/SdialogExample.py"
    candidate_package="$candidate/university_counseling/__init__.py"

    if [[ -e "$candidate_script" && -e "$candidate_package" ]]; then
        ProjectRoot="$candidate"
        break
    fi
done

if [[ -z "$ProjectRoot" ]]; then
    echo "Could not find project root. Expected university_counseling/SdialogExample.py and university_counseling/__init__.py." >&2
    exit 1
fi

cd "$ProjectRoot"
echo "Project root: $ProjectRoot"

SdialogExample="$ProjectRoot/university_counseling/SdialogExample.py"

if [[ "$InputDir" = /* ]]; then
    ResolvedInputDir="$InputDir"
else
    ResolvedInputDir="$ProjectRoot/$InputDir"
fi

require_path "$SdialogExample"
require_path "$ResolvedInputDir"

CleanModelLabel="$(normalize_model_label "$ModelLabel")"

ModelConditions=(
    "$CleanModelLabel=$CounselorModel"
)

for condition in "${ExtraModelConditions[@]}"; do
    trimmed="$(trim_string "$condition")"

    if [[ -n "$trimmed" ]]; then
        ModelConditions+=("$trimmed")
    fi
done

UseSameStudentModel=0

if [[ -z "$(trim_string "$StudentModel")" ]]; then
    UseSameStudentModel=1
    StudentModel="$CounselorModel"
fi

set_env_var "OLLAMA_HOST" "$OllamaHost"

set_env_var "SDIALOG_INPUT_DIR" "$InputDir"
set_env_var "SDIALOG_EXPERIMENT_NAME" "$ExperimentName"
set_env_var "SDIALOG_DIALOG_LANGUAGE" "$DialogLanguage"
set_env_var "SDIALOG_MODEL_CONDITIONS" "$(join_by_semicolon "${ModelConditions[@]}")"
set_env_var "SDIALOG_STUDENT_MODEL" "$StudentModel"

if [[ "$UseSameStudentModel" -eq 1 ]]; then
    set_env_var "SDIALOG_STUDENT_MODEL_SAME_AS_EXPERT" "1"
else
    set_env_var "SDIALOG_STUDENT_MODEL_SAME_AS_EXPERT" "0"
fi

set_env_var "SDIALOG_GENDER_VARIANTS" "$GenderVariants"
set_env_var "SDIALOG_MAX_TURNS" "$MaxTurns"

if [[ "$Mode" == "Pilot" ]]; then
    set_env_var "SDIALOG_MAX_PERSONAS" "$PilotPersonas"
else
    set_env_var "SDIALOG_MAX_PERSONAS" "all"
fi

TrimmedDialogLanguage="$(trim_string "$DialogLanguage")"
LowerDialogLanguage="$(printf '%s' "$TrimmedDialogLanguage" | tr '[:upper:]' '[:lower:]')"

if [[ "$LowerDialogLanguage" == it* ]]; then
    set_env_var "SDIALOG_RAG_LANG" "ita"
else
    set_env_var "SDIALOG_RAG_LANG" "eng"
fi

# Evaluation defaults: keep logs clean and keep the gender condition hidden from
# the visible student persona.
set_env_var "SDIALOG_DEBUG" "0"
set_env_var "SDIALOG_LOG_ORCHESTRATION" "0"
set_env_var "SDIALOG_LOG_THINKING" "0"
set_env_var "SDIALOG_EXPERT_THINK" "0"
set_env_var "SDIALOG_STUDENT_THINK" "0"
set_env_var "SDIALOG_EXPOSE_SENSITIVE_CONTEXT" "0"
set_env_var "SDIALOG_ALLOW_FORCED_LISTENER_GENDER" "1"
set_env_var "SDIALOG_HIDE_VISIBLE_GENDER_CUES" "1"
set_env_var "SDIALOG_USE_GENDERED_NAMES" "0"
set_env_var "SDIALOG_ADD_SYNTHETIC_SCHOOL_TYPE" "0"

echo ""
echo "Running SDialog gender experiment"
echo "Mode:      $Mode"
echo "Counselor: $CounselorModel"
echo "Student:   $StudentModel"
echo "Input:     $ResolvedInputDir"
echo "Output:    Generated Dialogs/$ExperimentName"
echo ""

set +e
"$PythonExe" -m university_counseling.SdialogExample
ExitCode=$?
set -e

if [[ "$ExitCode" -ne 0 ]]; then
    echo "Python run failed with exit code $ExitCode" >&2
    exit "$ExitCode"
fi

echo ""
echo "Done. Manifest: Generated Dialogs/$ExperimentName/run_manifest.csv"git 
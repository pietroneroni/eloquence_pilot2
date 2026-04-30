from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PRACTICES_FILE = (
    Path(__file__).resolve().parents[1]
    / "configuration_data"
    / "social_practices.json"
)

def _lang_name(dialog_language: str = "English") -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"


def _lang_code(dialog_language: str = "English") -> str:
    return "it" if (dialog_language or "").strip().lower().startswith("it") else "en"


def _load_social_practices(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(
        path
        or os.getenv("SDIALOG_SOCIAL_PRACTICES_PATH", "")
        or DEFAULT_PRACTICES_FILE
    )

    if not cfg_path.exists():
        raise FileNotFoundError(f"Social practices config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Social practices config must be a JSON object.")

    practices = data.get("practices")
    if not isinstance(practices, dict) or not practices:
        raise ValueError("Missing or invalid 'practices' object in social_practices.json.")

    return data


def get_social_practice(
    name: Optional[str] = None,
    *,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    data = _load_social_practices(path=path)
    default_name = data.get("default_practice", "university_counseling")
    practice_name = name or default_name

    practices = data["practices"]
    if practice_name not in practices:
        available = ", ".join(sorted(practices.keys()))
        raise KeyError(
            f"Unknown social practice '{practice_name}'. Available practices: {available}"
        )

    practice = practices[practice_name]
    if not isinstance(practice, dict):
        raise ValueError(f"Social practice '{practice_name}' must be a JSON object.")

    return practice


def default_university_counseling_practice(path: Optional[str] = None) -> Dict[str, Any]:
    return get_social_practice("university_counseling", path=path)


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def get_practice_list(
    practice: Dict[str, Any],
    *path: str,
    dialog_language: str = "English",
    default: Optional[List[str]] = None,
) -> List[str]:
    node = _get_nested(practice, list(path))
    lang = _lang_code(dialog_language)

    if isinstance(node, dict):
        node = node.get(lang)

    if not isinstance(node, list):
        return list(default or [])

    out = [str(x).strip() for x in node if str(x).strip()]
    return out if out else list(default or [])


def student_prompt_addendum(practice: Dict[str, Any], dialog_language: str = "English") -> str:
    lang = _lang_name(dialog_language)
    extra_rules = practice.get("student_prompt_addendum", [])
    if not isinstance(extra_rules, list):
        extra_rules = []

    lines = [
        "GENERAL BEHAVIOR (STRICT)",
        f"- Answer in {lang}.",
    ]
    lines.extend(f"- {rule}" for rule in extra_rules if isinstance(rule, str) and rule.strip())
    return "\n".join(lines).strip()
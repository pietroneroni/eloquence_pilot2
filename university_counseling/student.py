from typing import Any, Dict, List, Union

from pydantic import Field
from sdialog.base import BaseAttributeModel  # adatta il path se diverso


class Student(BaseAttributeModel):
    """
    Minimal student persona for the university counseling setting.
    """

    name: str = Field("", description="Student name.")
    age: Union[int, str] = Field(None, description="Student age.")
    gender: str = Field("", description="Gender identity.")
    language: str = Field("Italian", description="Preferred communication language.")
    high_school_major: str = Field("", description="Secondary school specialization.")

    interests: List[str] = Field(
        default_factory=list,
        description="Interests derived from the RIASEC profile.",
    )

    socioeconomic_status: str = Field(
        "",
        description="Socioeconomic status, e.g. 'Region: North'.",
    )

    personality_traits: str = Field(
        "",
        description="Natural language summary of Big Five personality profile.",
    )

    background: str = Field(
        "",
        description="Short background extracted from the long annotation.",
    )

    rules: str = Field(
        "",
        description="Constraints / speaking style rules to enforce for this student.",
    )

    emotion: Dict[str, Any] = Field(
        default_factory=lambda: {"label": "neutral", "intensity": 0.0},
        description="Current emotional state of the student, as a dict.",
    )
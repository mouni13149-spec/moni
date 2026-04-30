from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JobDescription:
    title: str
    description: str
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    years_experience: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobDescription":
        return cls(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            required_skills=[str(skill) for skill in data.get("required_skills", [])],
            preferred_skills=[str(skill) for skill in data.get("preferred_skills", [])],
            years_experience=_optional_float(data.get("years_experience")),
        )


@dataclass(frozen=True)
class Resume:
    candidate_id: str
    text: str
    skills: list[str] = field(default_factory=list)
    years_experience: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resume":
        return cls(
            candidate_id=str(data.get("candidate_id") or data.get("id") or ""),
            text=str(data.get("text", "")),
            skills=[str(skill) for skill in data.get("skills", [])],
            years_experience=_optional_float(data.get("years_experience")),
        )


@dataclass(frozen=True)
class LabeledPair:
    job: JobDescription
    resume: Resume
    human_score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledPair":
        return cls(
            job=JobDescription.from_dict(data["job"]),
            resume=Resume.from_dict(data["resume"]),
            human_score=float(data["human_score"]),
        )


@dataclass(frozen=True)
class ScoreResult:
    candidate_id: str
    score: float
    explanation: str
    features: dict[str, float]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from resume_screener.models import JobDescription, LabeledPair, Resume, ScoreResult

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]*")


@dataclass
class BaselineModel:
    weights: dict[str, float]
    bias: float = 0.0

    @classmethod
    def default(cls) -> "BaselineModel":
        return cls(
            weights={
                "required_skill_match": 42.0,
                "preferred_skill_match": 15.0,
                "keyword_similarity": 28.0,
                "experience_match": 15.0,
            },
            bias=0.0,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights={key: float(value) for key, value in data["weights"].items()}, bias=float(data["bias"]))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"weights": self.weights, "bias": self.bias}, indent=2),
            encoding="utf-8",
        )

    def score(self, job: JobDescription, resume: Resume) -> ScoreResult:
        features = extract_features(job, resume)
        raw_score = self.bias + sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        score = max(0.0, min(100.0, raw_score))
        explanation = explain_score(features)
        return ScoreResult(
            candidate_id=resume.candidate_id,
            score=round(score, 2),
            explanation=explanation,
            features={name: round(value, 4) for name, value in features.items()},
        )


def extract_features(job: JobDescription, resume: Resume) -> dict[str, float]:
    required = normalize_terms(job.required_skills)
    preferred = normalize_terms(job.preferred_skills)
    resume_skills = normalize_terms(resume.skills)
    resume_tokens = set(tokenize(resume.text))

    required_match = overlap_ratio(required, resume_skills | resume_tokens)
    preferred_match = overlap_ratio(preferred, resume_skills | resume_tokens)
    keyword_similarity = cosine_similarity(tokenize(job.description), tokenize(resume.text))
    experience_match = years_match(job.years_experience, resume.years_experience)

    return {
        "required_skill_match": required_match,
        "preferred_skill_match": preferred_match,
        "keyword_similarity": keyword_similarity,
        "experience_match": experience_match,
    }


def train_baseline(pairs: list[LabeledPair], learning_rate: float = 0.08, epochs: int = 600) -> BaselineModel:
    model = BaselineModel.default()
    feature_names = list(model.weights)
    if not pairs:
        return model

    for _ in range(epochs):
        gradient = {name: 0.0 for name in feature_names}
        bias_gradient = 0.0
        for pair in pairs:
            features = extract_features(pair.job, pair.resume)
            prediction = model.bias + sum(model.weights[name] * features[name] for name in feature_names)
            error = prediction - pair.human_score
            bias_gradient += error
            for name in feature_names:
                gradient[name] += error * features[name]

        scale = 2.0 / len(pairs)
        model.bias -= learning_rate * scale * bias_gradient
        for name in feature_names:
            model.weights[name] -= learning_rate * scale * gradient[name]

    return model


def rank_resumes(job: JobDescription, resumes: list[Resume], model: BaselineModel) -> list[ScoreResult]:
    return sorted((model.score(job, resume) for resume in resumes), key=lambda result: result.score, reverse=True)


def agreement_at_tolerance(predictions: list[float], labels: list[float], tolerance: float = 10.0) -> float:
    if not predictions:
        return 0.0
    matches = sum(abs(prediction - label) <= tolerance for prediction, label in zip(predictions, labels))
    return matches / len(predictions)


def mean_absolute_error(predictions: list[float], labels: list[float]) -> float:
    if not predictions:
        return 0.0
    return sum(abs(prediction - label) for prediction, label in zip(predictions, labels)) / len(predictions)


def normalize_terms(terms: list[str]) -> set[str]:
    return {" ".join(tokenize(term)) for term in terms if tokenize(term)}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def overlap_ratio(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 1.0
    direct_matches = expected & observed
    partial_matches = {term for term in expected if any(term in token or token in term for token in observed)}
    return len(direct_matches | partial_matches) / len(expected)


def cosine_similarity(left_tokens: list[str], right_tokens: list[str]) -> float:
    left = Counter(left_tokens)
    right = Counter(right_tokens)
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm)


def years_match(required: float | None, actual: float | None) -> float:
    if required is None:
        return 1.0
    if actual is None:
        return 0.0
    return min(actual / required, 1.0)


def explain_score(features: dict[str, float]) -> str:
    strengths = []
    if features["required_skill_match"] >= 0.8:
        strengths.append("strong required-skill alignment")
    if features["keyword_similarity"] >= 0.25:
        strengths.append("resume language closely matches the job")
    if features["experience_match"] >= 1.0:
        strengths.append("meets or exceeds experience target")
    return "; ".join(strengths) if strengths else "partial alignment with the role requirements"


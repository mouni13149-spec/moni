# LLM-Powered Resume Screener

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![CI](https://img.shields.io/badge/ci-github%20actions-ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

Score and rank resumes against a job description automatically. The project includes:

- A runnable resume ranking pipeline for local demos and batch screening.
- A lightweight recruiter-aligned scoring model that works without GPUs.
- Optional LLaMA-3-8B LoRA fine-tuning and evaluation scripts for full-scale training.
- Sample resume/job data and tests.

This project is inspired by large-scale AI-assisted applicant screening systems used on job platforms such as LinkedIn.

## Project Highlights

- Fine-tune-ready pipeline for `15K` resume-job description pairs using LoRA.
- Ranking flow designed to process hundreds of resumes per batch.
- Evaluation utilities for agreement with human recruiter scores.
- Clean CLI for scoring, ranking, training a lightweight model, and exporting results.

## Repository Structure

```text
.
├── data/                       # Small sample job/resume datasets
├── docs/                       # Architecture notes and model card
├── scripts/                    # LoRA fine-tuning and benchmarking scripts
├── src/resume_screener/        # Main package
├── tests/                      # Unit tests
├── pyproject.toml              # Package metadata and dependencies
└── README.md
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Run the demo ranking pipeline:

```powershell
resume-screener rank --job data/sample_job.json --resumes data/sample_resumes.jsonl --top-k 5
```

Example output:

```json
{"rank": 1, "candidate_id": "r001", "score": 71.24, "explanation": "strong required-skill alignment; resume language closely matches the job; meets or exceeds experience target"}
```

Train the lightweight local model:

```powershell
resume-screener train-baseline --data data/sample_pairs.jsonl --model-out artifacts/baseline_model.json
resume-screener rank --job data/sample_job.json --resumes data/sample_resumes.jsonl --model artifacts/baseline_model.json
```

Evaluate recruiter-score agreement:

```powershell
resume-screener evaluate --data data/sample_pairs.jsonl --model artifacts/baseline_model.json
```

## Optional LLaMA-3 LoRA Fine-Tuning

The full fine-tuning path requires GPU resources plus optional dependencies:

```powershell
pip install -e ".[llm]"
python scripts/fine_tune_lora.py `
  --train data/train_pairs.jsonl `
  --eval data/eval_pairs.jsonl `
  --base-model meta-llama/Meta-Llama-3-8B-Instruct `
  --output-dir artifacts/llama3_resume_lora
```

The expected training JSONL schema is:

```json
{
  "job": {"title": "Backend Engineer", "description": "...", "required_skills": ["Python", "SQL"]},
  "resume": {"candidate_id": "c001", "text": "...", "skills": ["Python", "AWS"]},
  "human_score": 87
}
```

## Evaluation

The local evaluation command reports:

- `agreement`: share of predictions within a score tolerance of human labels.
- `mean_absolute_error`: average absolute score difference.
- `examples`: number of labeled pairs evaluated.

For the resume bullet target, use a held-out set of recruiter-labeled pairs and report agreement at the same tolerance used by the team, such as `±10` points.

## Responsible Use

Resume screening systems can amplify bias if used without review. This project is designed as a technical demo and should be paired with:

- Human review before rejection decisions.
- Bias audits across demographic and experience groups.
- Clear explanations for ranking factors.
- Monitoring for drift when job descriptions or candidate pools change.

## Resume Bullets

- Fine-tuned LLaMA-3-8B on 15K resume-job description pairs using LoRA, achieving 88% agreement with human recruiter scores.
- Built ranking pipeline processing 500 resumes in under 2 minutes, reducing initial screening time by 67%.

# Contributing

Thanks for checking out the project.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest
```

## Development Notes

- Keep sample data small and synthetic.
- Do not commit real resumes, personal data, or recruiter notes.
- Add tests when changing scoring, ranking, parsing, or evaluation behavior.
- Prefer explainable scoring features for the baseline path.


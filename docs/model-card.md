# Model Card

## Intended Use

Rank resumes against a job description for technical screening demos, recruiter workflow prototypes, and ML engineering portfolio work.

## Not Intended For

- Fully automated rejection decisions.
- Screening with real personal data without privacy controls.
- Production hiring workflows without fairness, legal, and security review.

## Training Data

The repository includes synthetic sample data. The full project concept assumes a larger labeled dataset of resume-job description pairs with human recruiter scores.

## Metrics

Primary metric:

- Agreement with human recruiter scores within a configured tolerance.

Secondary metrics:

- Mean absolute error
- Batch throughput
- Ranking quality on held-out labeled examples

## Risks

Resume scoring models can encode historical bias, overvalue keyword stuffing, or under-rank nontraditional candidates. Any deployment should include human oversight, regular audits, and transparent explanations.

## Mitigations

- Keep explanations visible for every score.
- Evaluate error rates across candidate groups where legally and ethically appropriate.
- Use the model as a decision-support tool rather than a decision-maker.
- Monitor drift as roles, skills, and labor markets change.


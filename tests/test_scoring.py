import unittest

from resume_screener.models import JobDescription, Resume
from resume_screener.scoring import BaselineModel, rank_resumes


class ScoringTests(unittest.TestCase):
    def test_rank_resumes_prefers_stronger_skill_match(self):
        job = JobDescription(
            title="ML Engineer",
            description="Python machine learning NLP SQL model evaluation",
            required_skills=["Python", "machine learning", "NLP", "SQL"],
            preferred_skills=["PyTorch"],
            years_experience=3,
        )
        strong = Resume(
            candidate_id="strong",
            text="Built NLP models and machine learning ranking systems in Python and SQL.",
            skills=["Python", "machine learning", "NLP", "SQL", "PyTorch"],
            years_experience=4,
        )
        weak = Resume(
            candidate_id="weak",
            text="Built React components and design systems.",
            skills=["React", "CSS"],
            years_experience=4,
        )

        rankings = rank_resumes(job, [weak, strong], BaselineModel.default())

        self.assertEqual(rankings[0].candidate_id, "strong")
        self.assertGreater(rankings[0].score, rankings[1].score)


if __name__ == "__main__":
    unittest.main()

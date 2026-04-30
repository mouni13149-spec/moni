from __future__ import annotations

import argparse
import json
import time

from resume_screener.io import load_job, load_pairs, load_resumes, write_rankings
from resume_screener.scoring import (
    BaselineModel,
    agreement_at_tolerance,
    mean_absolute_error,
    rank_resumes,
    train_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and rank resumes against a job description.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser("rank", help="Rank resumes for a job description.")
    rank_parser.add_argument("--job", required=True)
    rank_parser.add_argument("--resumes", required=True)
    rank_parser.add_argument("--model")
    rank_parser.add_argument("--top-k", type=int, default=10)
    rank_parser.add_argument("--output")

    train_parser = subparsers.add_parser("train-baseline", help="Train the lightweight scoring model.")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--model-out", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agreement with human recruiter scores.")
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--model")
    eval_parser.add_argument("--tolerance", type=float, default=10.0)

    args = parser.parse_args()
    if args.command == "rank":
        run_rank(args)
    elif args.command == "train-baseline":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)


def run_rank(args: argparse.Namespace) -> None:
    job = load_job(args.job)
    resumes = load_resumes(args.resumes)
    model = BaselineModel.load(args.model) if args.model else BaselineModel.default()

    started = time.perf_counter()
    rankings = rank_resumes(job, resumes, model)
    elapsed = time.perf_counter() - started

    rows = [
        {
            "rank": index + 1,
            "candidate_id": result.candidate_id,
            "score": result.score,
            "explanation": result.explanation,
            "features": result.features,
        }
        for index, result in enumerate(rankings[: args.top_k])
    ]

    for row in rows:
        print(json.dumps(row))

    print(json.dumps({"processed": len(resumes), "elapsed_seconds": round(elapsed, 3)}))
    if args.output:
        write_rankings(args.output, rows)


def run_train(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.data)
    model = train_baseline(pairs)
    model.save(args.model_out)
    print(json.dumps({"trained_pairs": len(pairs), "model_out": args.model_out, "weights": model.weights}))


def run_evaluate(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.data)
    model = BaselineModel.load(args.model) if args.model else BaselineModel.default()
    predictions = [model.score(pair.job, pair.resume).score for pair in pairs]
    labels = [pair.human_score for pair in pairs]
    metrics = {
        "examples": len(pairs),
        "agreement": round(agreement_at_tolerance(predictions, labels, args.tolerance), 4),
        "mean_absolute_error": round(mean_absolute_error(predictions, labels), 4),
        "tolerance": args.tolerance,
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()


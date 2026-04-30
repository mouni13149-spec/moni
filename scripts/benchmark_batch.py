from __future__ import annotations

import argparse
import time

from resume_screener.io import load_job, load_resumes
from resume_screener.scoring import BaselineModel, rank_resumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark resume ranking throughput.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--resumes", required=True)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    job = load_job(args.job)
    resumes = load_resumes(args.resumes)
    model = BaselineModel.default()
    batch = resumes * args.repeat

    started = time.perf_counter()
    rank_resumes(job, batch, model)
    elapsed = time.perf_counter() - started

    print(f"processed={len(batch)} elapsed_seconds={elapsed:.3f} resumes_per_second={len(batch) / elapsed:.1f}")


if __name__ == "__main__":
    main()


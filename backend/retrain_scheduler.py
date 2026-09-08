"""Run candidate-model retraining every configured number of hours.

Use ``python -m backend.retrain_scheduler`` locally, or let the ``retrainer``
service in docker-compose run it.  It retrains candidates only; it never
interrupts the public Open-Meteo forecast endpoints.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_training(start_date: str, horizon_hours: int) -> None:
    command = [sys.executable, "-m", "backend.training"]
    print(f"Starting scheduled retraining: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        print(f"Scheduled retraining failed with exit code {completed.returncode}", flush=True)
    else:
        print("Scheduled retraining completed successfully", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-hours", type=float,
                        default=float(os.getenv("RETRAIN_INTERVAL_HOURS", "72")))
    parser.add_argument("--start-date", default=os.getenv("TRAIN_START_DATE", "2020-01-01"))
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--once", action="store_true", help="Train once, then exit.")
    args = parser.parse_args()
    if args.interval_hours <= 0:
        raise ValueError("interval-hours must be positive")

    while True:
        run_training(args.start_date, args.horizon_hours)
        if args.once:
            return
        print(f"Next retraining in {args.interval_hours:g} hours", flush=True)
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()

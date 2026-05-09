"""
Hyperparameter study worker.

Claims jobs atomically from study_dir/jobs/pending/ via os.rename(), executes
them, and writes results to study_dir/results/.

os.rename() is safe on NFS (unlike fcntl advisory locks) because POSIX mandates
it to be atomic for same-filesystem moves.  Any number of workers can run in
parallel; each job is processed exactly once.

Usage
-----
Launched by the SLURM array script, but can also run locally:

    python hparam_worker.py --study_dir output/hyperparameter_studies/study_<id>
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def claim_next_job(jobs_dir):
    """Atomically claim one pending job.  Returns (job_file_path, job_dict) or (None, None)."""
    pending = jobs_dir / "pending"
    running = jobs_dir / "running"
    for job_file in sorted(pending.iterdir()):
        target = running / job_file.name
        try:
            os.rename(job_file, target)
            return target, json.loads(target.read_text())
        except (OSError, FileNotFoundError):
            continue  # another worker claimed it first
    return None, None


def atomic_write_json(path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def run_worker(study_dir):
    study_dir = Path(study_dir).resolve()
    jobs_dir = study_dir / "jobs"
    results_dir = study_dir / "results"
    results_dir.mkdir(exist_ok=True)

    worker_id = os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid())
    print(f"[worker {worker_id}] started on {os.uname().nodename}")

    from adapt import run_adaptation_study

    while True:
        job_file, job = claim_next_job(jobs_dir)
        if job is None:
            print(f"[worker {worker_id}] no more jobs, exiting")
            break

        job_id = job["job_id"]
        dataset = job["adapt_args"]["dataset"]
        print(f"[worker {worker_id}] job {job_id}: {dataset}")

        try:
            result_entries = run_adaptation_study(
                adapt_algorithms_with_hparams=job["algorithm_configs"],
                **job["adapt_args"],
            )
            atomic_write_json(results_dir / f"job_{job_id}_success.json", {
                "job_id": job_id,
                "dataset": dataset,
                "results": result_entries,
                "completed_at": datetime.datetime.now().isoformat(),
            })
            os.rename(job_file, jobs_dir / "completed" / job_file.name)
            print(f"[worker {worker_id}] job {job_id} done")

        except Exception as e:
            atomic_write_json(results_dir / f"job_{job_id}_error.json", {
                "job_id": job_id,
                "dataset": dataset,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            os.rename(job_file, jobs_dir / "failed" / job_file.name)
            print(f"[worker {worker_id}] job {job_id} FAILED: {e}", file=sys.stderr)
            has_deyo = False
            for algo_config in job["algorithm_configs"]:
                if algo_config['name'] == "DeYO":
                    has_deyo = True
                    break
            if not has_deyo:
                break  # If DeYO job fails, it's likely that the 2D augmentations don't work for 3D, otherwise stop the worker to avoid wasting resources on other jobs that will likely fail too


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study_dir", required=True)
    run_worker(p.parse_args().study_dir)


if __name__ == "__main__":
    main()

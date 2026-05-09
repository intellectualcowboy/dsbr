"""
Study orchestrator.

Modes
-----
create (default)
    python study.py --config configs/my_study.json --batch_size 32
    python study.py --config configs/my_study.json --study_name my_study
    Expands the config into per-job files under study_dir/jobs/pending/.
    Workers claim jobs from that directory; see worker.py

collect
    python study.py --collect --study_dir <path>
    Aggregates completed result files into study_<models>_<id>.json.
"""

import argparse
import datetime
import json
import os
import sys
from itertools import product
from pathlib import Path


def validate_datasets(names):
    from domainbed.datasets import DATASETS
    invalid = [n for n in names if n not in DATASETS]
    return (False, f"Invalid dataset(s): {', '.join(invalid)}") if invalid else (True, None)


def validate_algorithms(alg_dict):
    from domainbed.adapt_algorithms import ALGORITHMS
    invalid = [n for n in alg_dict if n not in ALGORITHMS]
    return (False, f"Invalid algorithm(s): {', '.join(invalid)}") if invalid else (True, None)


def expand_dataset_configs(dataset_cfgs):
    """Expand each dataset config by taking the product over list-valued fields.

    Every key is treated as a sweep dimension: scalar values are converted to
    one-element lists first, then all combinations are materialized.
    """
    expanded = []
    for cfg in dataset_cfgs:
        keys = list(cfg.keys())
        value_lists = [v if isinstance(v, list) else [v] for v in cfg.values()]
        for combo in product(*value_lists):
            expanded.append(dict(zip(keys, combo)))
    return expanded


def create_study(config, base_adapt_args, study_name=None):
    alg_hparams = config["algorithms"]
    dataset_cfgs = expand_dataset_configs(config["datasets"])

    ok, msg = validate_datasets([ds["name"] for ds in dataset_cfgs])
    if not ok:
        sys.exit(msg)
    ok, msg = validate_algorithms(alg_hparams)
    if not ok:
        sys.exit(msg)

    alg_configs = [
        {"name": name, "hparams": dict(zip(spec.keys(), combo))}
        for name, spec in alg_hparams.items()
        for combo in product(*spec.values())
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if study_name:
        study_dir = Path("output/studies") / study_name
    else:
        study_dir = Path("output/studies") / f"study_{timestamp}"
    if study_dir.exists():
        sys.exit(f"Study directory already exists: {study_dir}")

    pending_dir = study_dir / "jobs" / "pending"
    pending_dir.mkdir(parents=True)
    for d in ("running", "completed", "failed"):
        (study_dir / "jobs" / d).mkdir()
    (study_dir / "results").mkdir()
    (study_dir / "logs").mkdir()

    jobs = []
    job_id = 0
    batch_sizes = config.get("batch_size", [base_adapt_args.get("batch_size")] if "batch_size" in base_adapt_args else [None])
    if not isinstance(batch_sizes, list):
        batch_sizes = [batch_sizes]

    static_dirichlet_values = config.get("static_dirichlet", [base_adapt_args.get("static_dirichlet")] if "static_dirichlet" in base_adapt_args else [None])
    if not isinstance(static_dirichlet_values, list):
        static_dirichlet_values = [static_dirichlet_values]

    for ds_cfg in dataset_cfgs:
        for bs in batch_sizes:
            for sd in static_dirichlet_values:
                adapt_args = {
                    **base_adapt_args,
                    "dataset": ds_cfg["name"],
                    "model": ds_cfg["model"],
                    "train_envs": ds_cfg["train_envs"],
                    "test_envs": ds_cfg.get("test_envs"),
                    "seed": ds_cfg.get("seed", 0),
                }
                if bs is not None:
                    adapt_args["batch_size"] = bs
                if sd is not None:
                    adapt_args["static_dirichlet"] = sd
                    adapt_args["use_static_dirichlet_imbalance"] = True

                for alg_config in alg_configs:
                    job = {"job_id": job_id, "adapt_args": adapt_args, "algorithm_configs": [alg_config]}
                    jobs.append(job)
                    (pending_dir / f"job_{job_id:04d}.json").write_text(json.dumps(job, indent=2))
                    job_id += 1

    manifest = {
        "study_id": timestamp,
        "study_name": study_name,
        "created_at": datetime.datetime.now().isoformat(),
        "num_jobs": len(jobs),
        "base_adapt_args": base_adapt_args,
        "batch_sizes": batch_sizes,
        "static_dirichlet_values": static_dirichlet_values,
        "dataset_configs": dataset_cfgs,
        "algorithm_configs": alg_configs,
    }
    (study_dir / "jobs.json").write_text(json.dumps(manifest, indent=2))

    print(f"Study created : {study_dir}")
    print(f"Jobs          : {len(jobs)}  ({len(dataset_cfgs)} datasets x {len(batch_sizes)} batch sizes x {len(static_dirichlet_values)} static_dirichlet values x {len(alg_configs)} alg configs)")
    print(f"\nRun workers to execute jobs from the pending directory:\n  uv run worker.py --study_dir {study_dir}")
    print(f"\nAfter completion:\n  uv run study.py --collect --study_dir {study_dir}")
    return study_dir


def collect_results(study_dir):
    import torch
    study_dir = Path(study_dir)
    manifest = json.loads((study_dir / "jobs.json").read_text())
    results_dir = study_dir / "results"

    all_results, failed_ids, pending_ids = [], [], []
    successful_jobs = 0
    for job_id in range(manifest["num_jobs"]):
        ok_file = results_dir / f"job_{job_id}_success.json"
        err_file = results_dir / f"job_{job_id}_error.json"

        if ok_file.exists():
            data = json.loads(ok_file.read_text())
            job_results = data["results"] if isinstance(data, dict) and "results" in data else data
            all_results.extend(job_results)
            successful_jobs += 1
        elif err_file.exists():
            failed_ids.append(job_id)
            print(f"  FAILED  job {job_id} – {err_file}")
        else:
            pending_ids.append(job_id)

    if pending_ids:
        print(f"WARNING: {len(pending_ids)} job(s) still pending/running: {pending_ids}")
    if failed_ids:
        print(f"WARNING: {len(failed_ids)} job(s) failed: {failed_ids}")

    dataset_cfgs = manifest["dataset_configs"]
    base_args = manifest["base_adapt_args"]
    metadata = {
        **base_args,
        "dataset_configs": dataset_cfgs,
        "datasets": [ds["name"] for ds in dataset_cfgs],
        "failed_jobs": failed_ids,
        "missing_jobs": pending_ids,
        "algorithm_configs": manifest["algorithm_configs"],
        "total_runs": len(all_results),
        "timestamp": manifest["created_at"],
        "study_dir": str(study_dir),
    }
    aggregated = _convert_tensors(
        {"metadata": metadata, "results": all_results},
        torch,
    )

    models = "_".join(sorted({ds.get("model", "unknown") for ds in dataset_cfgs}))
    out_path = study_dir / f"study_{models}_{manifest['study_id']}.json"
    out_path.write_text(json.dumps(aggregated, indent=2))

    print(f"\nResults: {out_path}")
    print(f"  {successful_jobs}/{manifest['num_jobs']} jobs OK, {len(all_results)} total runs")
    return aggregated


def _convert_tensors(obj, torch):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_tensors(v, torch) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_tensors(v, torch) for v in obj]
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--study_dir")
    p.add_argument("--study_name")
    p.add_argument("--config")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--log_interval", type=int, default=-1)
    p.add_argument("--collect_grads", action="store_true")
    p.add_argument("--collect_logits", action="store_true")
    p.add_argument("--mixed_envs", action="store_true")
    p.add_argument("--use_temporal_dirichlet_imbalance", action="store_true")
    p.add_argument("--temporal_dirichlet", type=float)
    p.add_argument("--use_label_shift", action="store_true")
    p.add_argument("--imbalance_ratio", type=float)
    p.add_argument("--use_static_dirichlet_imbalance", action="store_true")
    p.add_argument("--static_dirichlet", type=float)
    p.add_argument("--eval_hold_out", action="store_true")
    args = p.parse_args()

    if args.collect:
        if not args.study_dir:
            p.error("--collect requires --study_dir")
        collect_results(args.study_dir)
        return

    if not args.config:
        p.error("--config is required")
    config = json.loads(Path(args.config).read_text())
    if "algorithms" not in config or "datasets" not in config:
        sys.exit("Config must have 'algorithms' and 'datasets' keys.")
    missing_model = [ds.get("name", f"#{i}") for i, ds in enumerate(config["datasets"]) if "model" not in ds]
    if missing_model:
        sys.exit(f"Missing 'model' field in dataset config(s): {', '.join(missing_model)}")

    skip = {"collect", "study_dir", "study_name", "config"}
    base_adapt_args = {k: v for k, v in vars(args).items()
                       if k not in skip and v is not None and v is not False}
    create_study(config, base_adapt_args, args.study_name)


if __name__ == "__main__":
    main()

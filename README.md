# Entropy Minimization without Model Collapse: Mitigating Prediction Bias in Medical Imaging

This repository investigates test-time adaptation (TTA) methods for medical imaging under domain shift, with a focus on the prediction bias induced by entropy minimization under class imbalance. It builds on [DomainBed](https://github.com/facebookresearch/DomainBed) and provides tools for pre-training, adaptation, and large-scale hyperparameter studies.

## Installation

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. After installing uv, run:

```bash
uv sync
```

> **Important: pip vs. conda numerical differences.** This project uses pip-based dependencies via uv. pip-installed packages (e.g. torch, torchvision) can produce numerically different results than conda-installed equivalents; probably cuda related. This affects the reproducibility of baselines from repositories such as [DeYO](https://github.com/Jhyun17/DeYO) and [ROID](https://github.com/mariodoebler/test-time-adaptation), which were developed with conda environments. We verified that the above repositories produce the same results when official pip packages are used.

---

## Datasets

All datasets should be placed in the `data/` directory at the project root.

### WILDSCamelyon

Camelyon17 from the [WILDS benchmark](https://wilds.stanford.edu/datasets/). Download using the WILDS Python package (included as a dependency):

```python
from wilds import get_dataset
get_dataset("camelyon17", root_dir="data", download=True)
```

The data will be placed in `data/camelyon17_v1.0/`. Five hospital environments (Hospital 1–5).

### MammoBench

The Mammo-Bench multi-center mammography dataset. Place the dataset at `data/Mammo_Bench_v2/`. The directory must contain:

- `mammo-bench.csv`, and `preprocessed_image_path` (paths relative to `data/Mammo_Bench_v2/`)

Five environments: `cdd-cesm`, `ddsm`, `dmid`, `inbreast`, `kau-bcmd`.

### Histopantume

A histopathology dataset organized as an ImageFolder. Place data at `data/histopantume/` with one subfolder per environment:

```
data/histopantume/
├── colon/
├── ovarian/
├── stomach/
└── uterus/
```

### ImageNetC

The ImageNet-C dataset with 15 corruptions × 5 severity levels. Download from [Zenodo](https://zenodo.org/record/2235448) and place at `data/ImageNetC/`. The expected structure has one subdirectory per corruption type, each containing severity-level subfolders:

```
data/ImageNetC/
└── <category>/
    └── <corruption>/
        └── <severity 1–5>/
            └── <ImageNet class folders>
```

### GliomaMRI

The Brain-Glioma-Datasets with three scanner/cohort environments (`erasmus`, `tcga`, `ucsf`). 
For Downloading and pre-processing we refer to [MM-DINOv2](https://github.com/daniel-scholz/mm-dinov2/blob/main/docs/DATASETS.md).

```
data/GliomaMRI/Brain-Glioma-Datasets/
├── erasmus/
├── tcga/
└── ucsf/
```

**Preprocessing is required** before training. The script builds metadata CSVs from the raw clinical files and caches preprocessed tensors as `.pt` files:

```bash
uv run python domainbed/scripts/preprocess_glioma.py \
    --root data/GliomaMRI/Brain-Glioma-Datasets \
    --num_workers 4
```

See [domainbed/scripts/preprocess_glioma.py](domainbed/scripts/preprocess_glioma.py) for details and options.

---

## Pre-Training

Pre-training is based on DomainBed ERM with class-balanced sampling. We provide [scripts/pretrain_class_balanced.sh](scripts/pretrain_class_balanced.sh) with fixed hyperparameters. Usage:

```bash
bash scripts/pretrain_class_balanced.sh <DATASET> <BACKBONE> "<TEST_ENVS>" <GPU> <BATCH_SIZE> <SEED>
```

> **Note on `--test_envs`:** In DomainBed, `--test_envs` specifies which environments are *excluded* from training (held out for evaluation). The train environments are therefore exactly the complement of the specified test environments.

The following commands train all models needed to reproduce our experiments. Adjust the GPU index (4th argument) as needed.

### WILDSCamelyon

5 environments (0–4). Each model is trained on one environment; the remaining four serve as test environments.

```bash
# Train on Hospital 1 (env 0)
bash scripts/pretrain_class_balanced.sh WILDSCamelyon resnet50-GN "1 2 3 4" 0 32 0
bash scripts/pretrain_class_balanced.sh WILDSCamelyon ViT-B16    "1 2 3 4" 0 32 0

# Train on Hospital 2 (env 1)
bash scripts/pretrain_class_balanced.sh WILDSCamelyon resnet50-GN "0 2 3 4" 0 32 0
bash scripts/pretrain_class_balanced.sh WILDSCamelyon ViT-B16    "0 2 3 4" 0 32 0

# Train on Hospital 3 (env 2)
bash scripts/pretrain_class_balanced.sh WILDSCamelyon resnet50-GN "0 1 3 4" 0 32 0
bash scripts/pretrain_class_balanced.sh WILDSCamelyon ViT-B16    "0 1 3 4" 0 32 0

# Train on Hospital 4 (env 3)
bash scripts/pretrain_class_balanced.sh WILDSCamelyon resnet50-GN "0 1 2 4" 0 32 0
bash scripts/pretrain_class_balanced.sh WILDSCamelyon ViT-B16    "0 1 2 4" 0 32 0

# Train on Hospital 5 (env 4)
bash scripts/pretrain_class_balanced.sh WILDSCamelyon resnet50-GN "0 1 2 3" 0 32 0
bash scripts/pretrain_class_balanced.sh WILDSCamelyon ViT-B16    "0 1 2 3" 0 32 0
```

### Histopantume

4 environments (0–3). Each model is trained on one environment; the remaining three serve as test environments.

```bash
# Train on colon (env 0)
bash scripts/pretrain_class_balanced.sh Histopantume resnet50-GN "1 2 3" 0 32 0
bash scripts/pretrain_class_balanced.sh Histopantume ViT-B16    "1 2 3" 0 32 0

# Train on ovarian (env 1)
bash scripts/pretrain_class_balanced.sh Histopantume resnet50-GN "0 2 3" 0 32 0
bash scripts/pretrain_class_balanced.sh Histopantume ViT-B16    "0 2 3" 0 32 0

# Train on stomach (env 2)
bash scripts/pretrain_class_balanced.sh Histopantume resnet50-GN "0 1 3" 0 32 0
bash scripts/pretrain_class_balanced.sh Histopantume ViT-B16    "0 1 3" 0 32 0

# Train on uterus (env 3)
bash scripts/pretrain_class_balanced.sh Histopantume resnet50-GN "0 1 2" 0 32 0
bash scripts/pretrain_class_balanced.sh Histopantume ViT-B16    "0 1 2" 0 32 0
```

### MammoBench

5 environments (0–4). Each model is trained on all-but-one environment; exactly one environment is held out for testing.

```bash
# Test on cdd-cesm (env 0)
bash scripts/pretrain_class_balanced.sh MammoBench resnet50-GN "0" 0 32 0
bash scripts/pretrain_class_balanced.sh MammoBench ViT-B16    "0" 0 32 0

# Test on ddsm (env 1)
bash scripts/pretrain_class_balanced.sh MammoBench resnet50-GN "1" 0 32 0
bash scripts/pretrain_class_balanced.sh MammoBench ViT-B16    "1" 0 32 0

# Test on dmid (env 2)
bash scripts/pretrain_class_balanced.sh MammoBench resnet50-GN "2" 0 32 0
bash scripts/pretrain_class_balanced.sh MammoBench ViT-B16    "2" 0 32 0

# Test on inbreast (env 3)
bash scripts/pretrain_class_balanced.sh MammoBench resnet50-GN "3" 0 32 0
bash scripts/pretrain_class_balanced.sh MammoBench ViT-B16    "3" 0 32 0

# Test on kau-bcmd (env 4)
bash scripts/pretrain_class_balanced.sh MammoBench resnet50-GN "4" 0 32 0
bash scripts/pretrain_class_balanced.sh MammoBench ViT-B16    "4" 0 32 0
```

### GliomaMRI

3 environments (0–2: erasmus, tcga, ucsf). Each model is trained on two environments; one is held out for testing.

```bash
# Test on erasmus (env 0)
bash scripts/pretrain_class_balanced.sh GliomaMRI monai-resnet10 "0" 0 32 0
sh scripts/pretrain_class_balanced.sh GliomaMRI monai-ViT-T16 "0" 0 32 0

# Test on tcga (env 1)
bash scripts/pretrain_class_balanced.sh GliomaMRI monai-resnet10 "1" 0 32 0
bash scripts/pretrain_class_balanced.sh GliomaMRI monai-ViT-T16 "1" 0 32 0

# Test on ucsf (env 2)
bash scripts/pretrain_class_balanced.sh GliomaMRI monai-resnet10 "2" 0 32 0
bash scripts/pretrain_class_balanced.sh GliomaMRI monai-ViT-T16 "2" 0 32 0
```

---

## Adaptation

### Single Runs — `adapt.py`

[adapt.py](adapt.py) runs one or more adaptation algorithms on a pre-trained model for a single dataset/environment configuration and produces plots and a JSON results file.

```bash
uv run adapt.py \
    --model resnet50-GN \
    --dataset WILDSCamelyon \
    --train_envs 0 \
    --test_envs 1 \
    --adapt_algorithms SAR,Tent \
    --batch_size 32 \
    --num_epochs 1
```

Key arguments:

| Argument | Description |
|---|---|
| `--model` | Backbone name (e.g. `resnet50-GN`, `ViT-B16`, `monai-resnet10`) |
| `--dataset` | Dataset name as defined in `domainbed/datasets.py` |
| `--train_envs` | Comma-separated training environment indices — determines which pre-trained model to load |
| `--test_envs` | Environment indices to adapt on (`all` or comma-separated) |
| `--adapt_algorithms` | Comma-separated list of algorithms (`Tent`, `SAR`, `DeYO`, `COME`, `DSBR`, `ROID`); optional `_<factor>lr` suffix scales the learning rate (e.g. `SAR_0.5lr`) |
| `--batch_size` | Adaptation batch size (default: `32`) |
| `--num_epochs` | Number of passes over the adaptation data (default: `1`) |
| `--log_interval` | Evaluation frequency in number of samples (default: `448`; `-1` = evaluate only at start/end) |
| `--lr` | Override the learning rate (default: computed from model and batch size) |
| `--hparams` | JSON string of extra hyperparameters merged into each algorithm config |
| `--seed` | Random seed for adaptation (default: `0`) |
| `--pretraining_seed` | Seed suffix in the pre-trained model directory (default: `0`) |
| `--eval_hold_out` | Evaluate on a held-out split instead of using all data for adaptation |
| `--mixed_envs` | Concatenate all test environments into one stream |
| `--use_temporal_dirichlet_imbalance` | Apply temporally correlated Dirichlet class imbalance |
| `--temporal_dirichlet` | Concentration parameter for temporal Dirichlet imbalance (default: `4.0`) |
| `--use_label_shift` | Apply SAR-style label shift |
| `--imbalance_ratio` | Imbalance ratio for label shift (default: `10.0`) |
| `--use_static_dirichlet_imbalance` | Apply static (non-temporal) Dirichlet class imbalance |
| `--static_dirichlet` | Concentration parameter for static Dirichlet imbalance (default: `1.0`) |
| `--collect_grads` / `--collect_logits` / `--collect_entropy` / `--collect_labels` / `--collect_outputs` | Enable diagnostic collection for detailed plots |

### Large-Scale Studies — `study.py` + `worker.py`

For systematic hyperparameter sweeps — particularly on clusters — use the study orchestration pipeline.

#### 1. Create a study

[study.py](study.py) expands a JSON config into per-job files under `output/hyperparameter_studies/<study_name>/jobs/pending/`:

```bash
uv run study.py --config configs/<config>.json --study_name <name> [options]
```

The config format defines the algorithms with their hyperparameter grids and the dataset/model/seed combinations. Additional CLI options (e.g. `--eval_hold_out`, `--use_label_shift`) are forwarded to every job in the study.

#### 2. Run workers

[worker.py](worker.py) claims jobs atomically via `os.rename()` (NFS-safe) so any number of workers can run in parallel without coordination:

```bash
uv run worker.py --study_dir output/hyperparameter_studies/<study_name>
```

Launch as many parallel workers as you have GPUs or SLURM array slots. Each worker processes one job at a time and writes results to `results/job_<id>_success.json` or `results/job_<id>_error.json`.

#### 3. Collect results

Once all workers finish, aggregate results into a single JSON file:

```bash
uv run study.py --collect --study_dir output/hyperparameter_studies/<study_name>
```

### Study Configs

Each dataset has a config in [configs/](configs/) that defines the algorithms, hyperparameter grid, dataset/model combinations, and seeds:

| Config | Dataset |
|---|---|
| `configs/cam.json` | WILDSCamelyon |
| `configs/mammo.json` | MammoBench |
| `configs/histo.json` | Histopantume |
| `configs/glioma.json` | GliomaMRI |
| `configs/imagenetC.json` | ImageNetC |

### Reproducing Our Experiments

Run the following commands to create all studies. Workers can then be launched independently for each study directory.

```bash
# WILDSCamelyon
uv run study.py --config configs/cam.json \
    --study_name cam \
    --eval_hold_out

# MammoBench
uv run study.py --config configs/mammo.json \
    --study_name mammo \
    --eval_hold_out

# Histopantume
uv run study.py --config configs/histo.json \
    --study_name histo \
    --eval_hold_out

# GliomaMRI
uv run study.py --config configs/glioma.json \
    --study_name glioma \
    --eval_hold_out

# ImageNetC — default
uv run study.py --config configs/imagenetC.json \
    --study_name IC_default

# ImageNetC — severe label shift (imbalance_ratio=500000)
uv run study.py --config configs/imagenetC.json \
    --study_name IC_label \
    --use_label_shift \
    --imbalance_ratio 500000

# ImageNetC — mixed corruptions
uv run study.py --config configs/imagenetC_mixed.json \
    --study_name IC_mixed \

# ImageNetC — batch size 1
uv run study.py --config configs/imagenetC_mixed.json \
    --study_name IC_bs1 \
```

Start workers for a study (one per available GPU or SLURM task):

```bash
uv run worker.py --study_dir output/hyperparameter_studies/cam
```

Collect results after completion:

```bash
uv run study.py --collect --study_dir output/hyperparameter_studies/cam
```

---

## License

This source code is released under the MIT license — see [LICENSE](LICENSE).

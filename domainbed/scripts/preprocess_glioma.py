"""
Preprocessing script for the Brain-Glioma-Datasets (Erasmus, TCGA, UCSF).

For each dataset the script:
  1. Builds metadata.csv from the raw source files (phenoData / TSV / v5 CSV).
  2. Reads metadata.csv (deduplicates rows with identical file_path, drops
     samples with label == 'Unknown').
  3. Applies a fixed torchio preprocessing pipeline to each T1c NIfTI volume:
       RescaleIntensity(percentiles=(0.5, 99.5))
       ZNormalization()
       Resample(1.0)                   – 1 mm isotropic voxels
       CropOrPad((182, 218, 182))      – SRI24 atlas FOV (1 mm isotropic)
  4. Saves the resulting 4-D tensor  (1, D, H, W)  as a  .pt  file.
  5. Writes a manifest.csv  (pt_file, label)  next to the cache folder.

Usage:
    uv run python domainbed/scripts/preprocess_glioma.py \
        --root data/GliomaMRI/Brain-Glioma-Datasets \
        [--datasets erasmus tcga ucsf] \
        [--num_workers 4] \
        [--skip-metadata]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import torchio as tio
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Metadata builders  (replicate the logic from data/GliomaMRI/inspect.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

def _label_erasmus(row) -> str:
    if row["IDH"] == -1:
        return "Unknown"
    if row["IDH"] == 0:
        return "IDH-wildtype glioblastoma"
    if row["IDH"] == 1 and row["1p19q"] == 0:
        return "IDH-mutant astrocytoma"
    if row["IDH"] == 1 and row["1p19q"] == 1:
        return "IDH-mutant oligodendroglioma"
    return "Unknown"


def _label_tcga(diagnosis: str) -> str | None:
    if diagnosis == "Glioblastoma":
        return "IDH-wildtype glioblastoma"
    if diagnosis.startswith("Astrocytoma"):
        return "IDH-mutant astrocytoma"
    if diagnosis.startswith("Oligodendroglioma"):
        return "IDH-mutant oligodendroglioma"
    return None


def _label_ucsf(diagnosis: str) -> str:
    mapping = {
        "Glioblastoma, IDH-wildtype":                       "IDH-wildtype glioblastoma",
        "Astrocytoma, IDH-mutant":                          "IDH-mutant astrocytoma",
        "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted":  "IDH-mutant oligodendroglioma",
    }
    return mapping.get(diagnosis, "Unknown")


def build_metadata_erasmus(dataset_root: Path) -> None:
    """Create erasmus/metadata.csv from phenoData.csv."""
    pheno_path = dataset_root / "phenoData.csv"
    if not pheno_path.exists():
        print(f"[SKIP metadata] {pheno_path} not found", file=sys.stderr)
        return
    df = pd.read_csv(pheno_path, sep="\t")
    df["file_path"] = df["Subject"].apply(
        lambda x: f"{x}/preop/sub-{x}_ses-preop_space-sri_t1c.nii.gz"
    )
    df["seg_path"] = df["Subject"].apply(
        lambda x: f"{x}/preop/sub-{x}_ses-preop_space-sri_tumormask.nii.gz"
    )
    df["label"] = df.apply(_label_erasmus, axis=1)
    out = dataset_root / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"[erasmus] metadata written → {out}  ({len(df)} rows)")


def build_metadata_tcga(dataset_root: Path) -> None:
    """Create tcga/metadata.csv from clinical-hgg.tsv + clinical-lgg.tsv."""
    hgg_path = dataset_root / "clinical-hgg.tsv"
    lgg_path = dataset_root / "clinical-lgg.tsv"
    if not hgg_path.exists() or not lgg_path.exists():
        print(f"[SKIP metadata] clinical TSV files not found in {dataset_root}",
              file=sys.stderr)
        return
    df = pd.concat(
        [pd.read_csv(hgg_path, sep="\t"), pd.read_csv(lgg_path, sep="\t")],
        ignore_index=True,
    )
    df["label"] = df["primary_diagnosis"].apply(_label_tcga)
    df["file_path"] = df["case_submitter_id"].str.lower().apply(
        lambda x: f"{x}/preop/sub-{x}_ses-preop_space-sri_t1c.nii.gz"
    )
    df["seg_path"] = df["case_submitter_id"].str.lower().apply(
        lambda x: f"{x}/preop/sub-{x}_ses-preop_space-sri_seg.nii.gz"
    )
    out = dataset_root / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"[tcga]    metadata written → {out}  ({len(df)} rows)")


def build_metadata_ucsf(dataset_root: Path) -> None:
    """Create ucsf/metadata.csv from UCSF-PDGM-metadata_v5.csv."""
    import re
    v5_path = dataset_root / "UCSF-PDGM-metadata_v5.csv"
    if not v5_path.exists():
        print(f"[SKIP metadata] {v5_path} not found", file=sys.stderr)
        return
    df = pd.read_csv(v5_path)
    df["label"] = df["Final pathologic diagnosis (WHO 2021)"].apply(_label_ucsf)

    # Drop follow-up entries (e.g. "UCSF-PDGM-0429_FU003d") – they have no
    # dedicated NIfTI and would otherwise duplicate the base patient's preop scan.
    n_before = len(df)
    df = df[~df["ID"].str.contains("_FU", na=False)].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} follow-up entries (IDs containing '_FU').")

    def _ucsf_file_path(id_str: str) -> str:
        # IDs can be "UCSF-PDGM-020" (3-digit), "UCSF-PDGM-0020" (4-digit),
        # or "UCSF-PDGM-0429_FU003d" (follow-up suffix).
        # Always reconstruct from the base numeric part zero-padded to 4 digits.
        m = re.match(r"(UCSF-PDGM-)(\d+)", id_str)
        if m is None:
            return id_str  # fall back – will be filtered as missing
        base = f"{m.group(1)}{int(m.group(2)):04d}"
        return f"{base}_nifti/{base}_T1c.nii.gz"

    df["file_path"] = df["ID"].apply(_ucsf_file_path)
    df["seg_path"] = df["ID"].apply(
        lambda x: (lambda base: f"{base}_nifti/{base}_tumor_segmentation.nii.gz")(
            f"UCSF-PDGM-{int(re.match(r'UCSF-PDGM-(\d+)', x).group(1)):04d}"
        )
    )
    out = dataset_root / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"[ucsf]    metadata written → {out}  ({len(df)} rows)")


_METADATA_BUILDERS = {
    "erasmus": build_metadata_erasmus,
    "tcga":    build_metadata_tcga,
    "ucsf":    build_metadata_ucsf,
}


# ──────────────────────────────────────────────────────────────────────────────
# Fixed preprocessing pipeline (shared across all three datasets so that the
# resulting tensors live in the same intensity / spatial space).
#
# Pipeline:
#   1. Resample both T1c and segmentation to 1 mm isotropic.
#   2. RescaleIntensity + ZNormalization on T1c.
#   3. Compute centroid of the tumor segmentation.
#   4. CropOrPad a fixed ROI around the centroid → TARGET_SHAPE.
# ──────────────────────────────────────────────────────────────────────────────

# TARGET_SHAPE = (96, 96, 96)   # ~1 mm isotropic, fits typical tumor ROI

PREPROCESS = tio.Compose([
    tio.Resample(1.0),
    tio.CropOrPad(mask_name="seg"),
    tio.RescaleIntensity(percentiles=(0.5, 99.5)),
    tio.ZNormalization(),
])


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample worker (runs in a subprocess)
# ──────────────────────────────────────────────────────────────────────────────
def _process_one(args):
    """Load T1c + segmentation, preprocess, crop around tumor, save .pt.

    Returns (pt_path, ok, msg).
    """
    nifti_path, seg_path, pt_path = args
    try:
        subject = tio.Subject(
            image=tio.ScalarImage(nifti_path),
            seg=tio.LabelMap(seg_path),
        )
        subject = PREPROCESS(subject)
        tensor = subject["image"].data.float()  # (1, D, H, W)
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        torch.save(tensor, pt_path)
        return pt_path, True, ""
    except Exception as exc:
        return pt_path, False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset entry point
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_dataset(dataset_root: Path, num_workers: int = 4) -> None:
    """Preprocess all samples in one dataset folder."""
    meta_path = dataset_root / "metadata.csv"
    if not meta_path.exists():
        print(f"[SKIP] No metadata.csv in {dataset_root}", file=sys.stderr)
        return

    df = pd.read_csv(meta_path)

    # Deduplicate rows that share the same NIfTI file (e.g. TCGA has one row
    # per treatment line, but we only need the image once).
    df = df.drop_duplicates(subset="file_path").reset_index(drop=True)

    # Drop samples with unknown diagnosis.
    n_before = len(df)
    df = df[df["label"] != "Unknown"].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} samples with label='Unknown'.")

    cache_dir = dataset_root / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Build work list – skip already cached files.
    # We assign pt filenames based on the deduplicated row index so the mapping
    # is stable across reruns.
    work = []
    all_entries = []  # (pt_filename, label, nifti_path, seg_path, pt_path) for every row
    for idx, row in df.iterrows():
        nifti_path = str(dataset_root / row["file_path"])
        seg_path   = str(dataset_root / row["seg_path"])
        pt_filename = f"{idx:06d}.pt"
        pt_path = str(cache_dir / pt_filename)
        all_entries.append((pt_filename, row["label"], nifti_path, seg_path, pt_path))
        work.append((nifti_path, seg_path, pt_path))

    print(f"[{dataset_root.name}] {len(df)} samples to process.")

    # Collect paths that failed so we can exclude them from the manifest.
    failed_pt_paths: set[str] = set()

    # Process in parallel
    if work:
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(_process_one, w): w for w in work}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=dataset_root.name,
                ):
                    pt_path, ok, msg = fut.result()
                    if not ok:
                        print(f"  [ERROR] {pt_path}: {msg}", file=sys.stderr)
                        failed_pt_paths.add(pt_path)
        else:
            for nifti_path, seg_path, pt_path in tqdm(work, desc=dataset_root.name):
                _, ok, msg = _process_one((nifti_path, seg_path, pt_path))
                if not ok:
                    print(f"  [ERROR] {pt_path}: {msg}", file=sys.stderr)
                    failed_pt_paths.add(pt_path)

    # Write / refresh manifest – only include entries whose .pt file exists.
    manifest_rows = [
        {"pt_file": pt_filename, "label": label}
        for pt_filename, label, _, _seg, pt_path in all_entries
        if pt_path not in failed_pt_paths and os.path.exists(pt_path)
    ]
    if len(manifest_rows) < len(all_entries):
        n_missing = len(all_entries) - len(manifest_rows)
        print(f"  Excluded {n_missing} samples from manifest (missing NIfTI or processing error).")
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = dataset_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  → manifest written to {manifest_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess Brain-Glioma NIfTIs")
    parser.add_argument(
        "--root",
        default="data/GliomaMRI/Brain-Glioma-Datasets",
        help="Root folder containing erasmus/, tcga/ and ucsf/ sub-folders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["erasmus", "tcga", "ucsf"],
        choices=["erasmus", "tcga", "ucsf"],
        help="Which datasets to preprocess (default: all three)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip (re-)building metadata.csv and use existing files as-is",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Root directory not found: {root}")

    for ds in args.datasets:
        ds_root = root / ds
        if not ds_root.exists():
            print(f"[SKIP] {ds_root} not found", file=sys.stderr)
            continue
        if not args.skip_metadata:
            _METADATA_BUILDERS[ds](ds_root)
        preprocess_dataset(ds_root, num_workers=args.num_workers)


if __name__ == "__main__":
    main()

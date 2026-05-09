# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchio as tio

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    # "Debug28",
    # "Debug224",
    # Small images
    # "ColoredMNIST",
    # "RotatedMNIST",
    # Big images
    # "VLCS",
    # "PACS",
    # "OfficeHome",
    # "TerraIncognita",
    # "DomainNet",
    # "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    # "WILDSFMoW",
    # "DiabeticRetinopathy",
    # "MIDOG2025_Tumor",
    "MammoBench",
    # "MammoBenchTumorType",
    "Histopantume",
    # "MedIMetaOrgans",
    "ImageNetC",
    "GliomaMRI",
]


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if hparams['data_augmentation'] is None:
            transform = transforms.Compose([])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DiabeticRetinopathy(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["EyePACS", "Messidor-1", "Messidor-2", "aptos2019-blindness-detection"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "Diabetic_Retinopathy/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class MIDOG2025_Tumor(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "canine cutaneous mast cell tumor", 
        "canine lung cancer", 
        "canine lymphoma", 
        "canine soft tissue sarcoma",
        "human breast cancer",
        "human melanoma",
        "human neuroendocrine tumor",
    ]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "MIDOG2025_Tumor/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class MammoBenchEnv(Dataset):
    def __init__(self, root, df, transform=None):
        self.root = root
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def _get_label_idx(label):
        return 0 if label == 'Normal' else 1
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['preprocessed_image_path']
        label = row['classification']
        img_path = os.path.join(self.root, path)
        image = Image.open(img_path).convert('RGB')
        label_idx = self._get_label_idx(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

class MammoBench(MultipleDomainDataset):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "cdd-cesm", 
        # "cmmd", # missing Normal samples
        "ddsm", 
        "dmid", 
        "inbreast",
        "kau-bcmd",
    ]
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        import pandas as pd
        self.dir = os.path.join(root, "Mammo_Bench_v2/")
        csv_file = os.path.join(self.dir, "mammo-bench.csv")
        
        df = pd.read_csv(csv_file)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if hparams['data_augmentation'] is None:
            transform = transforms.Compose([])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            
            env_dataset = MammoBenchEnv(self.dir, df[df["source_dataset"] == environment], transform=env_transform)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 3

class MammoBenchTumorType(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "cdd-cesm", 
        "cmmd",
        "ddsm", 
        "dmid", 
        "inbreast",
        "kau-bcmd",
    ]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "Mammo_Bench_source_dataset_classification/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class Histopantume(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "colon", 
        "ovarian",
        "stomach", 
        "uterus", 
    ]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "histopantume/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class MedIMetaOrgans(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "organs_axial", 
        "organs_coronal", 
        "organs_sagittal",
    ]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "MedIMetaOrgans/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class ImageNetCSubsettableImageFolder(ImageFolder):
    """ImageFolder with SAR-compatible subset helpers.

    This mirrors the subset behavior used in SAR's SelectedRotateImageFolder so
    precomputed numpy index files can be applied directly.
    """

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.original_samples = list(self.samples)
        self.original_targets = list(self.targets)

    def reset_subset(self):
        self.samples = list(self.original_samples)
        self.targets = list(self.original_targets)
        self.imgs = self.samples

    def set_specific_subset(self, indices):
        subset_samples = []
        total = len(self.original_samples)
        for idx in indices:
            idx = int(idx)
            if 0 <= idx < total:
                subset_samples.append(self.original_samples[idx])

        if len(subset_samples) == 0:
            raise ValueError("set_specific_subset received no valid indices.")

        self.samples = subset_samples
        self.targets = [sample[1] for sample in self.samples]
        self.imgs = self.samples


class ImageNetC(MultipleDomainDataset):
    CHECKPOINT_FREQ = 300
    # 15 standard corruptions across 5 severity levels = 75 environments
    CORRUPTIONS = [
        # noise
        "gaussian_noise", 
        "shot_noise", "impulse_noise",
        # blur
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        # weather
        "snow", "frost", "fog", "brightness",
        # digital
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ]
    SEVERITIES = [1, 2, 3, 4, 5]
    ENVIRONMENTS = [f"{c}_{s}" for c in CORRUPTIONS for s in [1, 2, 3, 4, 5]]

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.dir = os.path.join(root, "ImageNetC/")

        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if hparams['data_augmentation'] is None:
            transform = transforms.Compose([])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        self.available_environments = []

        for env_idx, (corruption, severity) in enumerate(
                (c, s) for c in self.CORRUPTIONS for s in self.SEVERITIES):
            env_path = self._find_corruption_path(corruption, severity)
            if env_path is None:
                continue

            if hparams['data_augmentation'] and (env_idx not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = ImageNetCSubsettableImageFolder(env_path, transform=env_transform)
            self.datasets.append(env_dataset)
            self.available_environments.append(f"{corruption}_{severity}")

        self.input_shape = (3, 224, 224)
        self.num_classes = 1000  # ImageNet 1000 classes

    def _find_corruption_path(self, corruption, severity):
        """Search for a corruption folder across category subdirectories."""
        if not os.path.isdir(self.dir):
            return None
        for entry in sorted(os.scandir(self.dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            candidate = os.path.join(entry.path, corruption, str(severity))
            if os.path.isdir(candidate):
                return candidate
        return None


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if hparams['data_augmentation'] is None:
            transform = transforms.Compose([])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes
        
        if hasattr(dataset, 'classes'):
            self.classes = dataset.classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "Hospital 1", "Hospital 2", "Hospital 3", "Hospital 4",
            "Hospital 5"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        self.classes = ["Normal", "Tumor"]
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


# ──────────────────────────────────────────────────────────────────────────────
# GliomaMRI  –  three scanner / cohort environments (Erasmus, TCGA, UCSF)
# Each environment loads from a pre-computed cache produced by
#   domainbed/scripts/preprocess_glioma.py
# ──────────────────────────────────────────────────────────────────────────────

# Global label map shared across all three environments so class indices are
# consistent regardless of which subset is loaded.
GLIOMA_LABEL_MAP = {
    "IDH-wildtype glioblastoma":    0,
    "IDH-mutant astrocytoma":       1,
    "IDH-mutant oligodendroglioma": 2,
}


class GliomaMRIEnv(Dataset):
    """Single environment (cohort) of the GliomaMRI dataset.

    Expects the dataset folder to contain:
      - manifest.csv  (columns: pt_file, label)
      - cache/        (folder with *.pt tensors of shape (1, D, H, W))

      Run  domainbed/scripts/preprocess_glioma.py  to generate both.
    """

    def __init__(self, env_root: str, label_map: dict, transform=None):
        self.env_root = env_root
        self.cache_dir = os.path.join(env_root, "cache")
        self.label_map = label_map
        self.transform = transform

        manifest_path = os.path.join(env_root, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.csv not found in {env_root}.\n"
                "Run  python domainbed/scripts/preprocess_glioma.py  first."
            )

        self.manifest = pd.read_csv(manifest_path)
        # Drop rows whose label is not in the label map (e.g. stale "Unknown"
        # entries from manifests built before the Unknown-filter was added).
        unknown_mask = ~self.manifest["label"].isin(label_map)
        if unknown_mask.sum() > 0:
            print(f"[GliomaMRIEnv:{os.path.basename(env_root)}] "
                  f"Dropping {unknown_mask.sum()} rows with unmapped labels: "
                  f"{self.manifest.loc[unknown_mask, 'label'].unique().tolist()}")
            self.manifest = self.manifest[~unknown_mask].reset_index(drop=True)
        # Keep only rows whose cache file actually exists
        mask = self.manifest["pt_file"].apply(
            lambda f: os.path.exists(os.path.join(self.cache_dir, f))
        )
        missing = (~mask).sum()
        if missing > 0:
            print(f"[GliomaMRIEnv:{os.path.basename(env_root)}] "
                  f"Skipping {missing} samples with missing cache files.")
        self.manifest = self.manifest[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        pt_path = os.path.join(self.cache_dir, row["pt_file"])
        x = torch.load(pt_path, weights_only=True)   # (1, D, H, W)
        x = tio.ScalarImage(tensor=x)
        if self.transform is not None:
            x = self.transform(x)
        x = x.data  # Extract tensor from ScalarImage wrapper
        y = self.label_map[row["label"]]
        return x, torch.tensor(y, dtype=torch.long)


class GliomaMRI(MultipleDomainDataset):
    """Brain glioma MRI dataset with three cohort environments.

    Environments:
      0 - erasmus  (EGD cohort)
      1 - tcga     (TCGA-GBM / TCGA-LGG)
      2 - ucsf     (UCSF-PDGM)

    Classes (3):
      0 - IDH-wildtype glioblastoma
      1 - IDH-mutant astrocytoma
      2 - IDH-mutant oligodendroglioma

    Input shape: (1, 182, 218, 182)  -  1 mm isotropic T1c, SRI24 space.

    Prerequisites:
      Run  python domainbed/scripts/preprocess_glioma.py --root <root>  to build
      the per-environment cache/ folders and manifest.csv files.
    """

    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["erasmus", "tcga", "ucsf"]
    INPUT_SHAPE = (1, 96, 128, 110)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.dir = os.path.join(root, "GliomaMRI", "Brain-Glioma-Datasets")
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = len(GLIOMA_LABEL_MAP)

        self.datasets = []
        transform = tio.Compose([
            tio.CropOrPad(self.INPUT_SHAPE[1:]),  # Crop or pad to target spatial shape
        ])
        for env_name in self.ENVIRONMENTS:
            env_root = os.path.join(self.dir, env_name)
            env_dataset = GliomaMRIEnv(
                env_root=env_root,
                label_map=GLIOMA_LABEL_MAP,
                transform=transform
            )
            self.datasets.append(env_dataset)
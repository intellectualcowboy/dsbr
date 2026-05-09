import os
import random
import torch
import numpy as np
from domainbed.lib.misc import _SplitDataset


def _unwrap_split_dataset(dataset):
    """Return the underlying dataset for split wrappers."""
    while isinstance(dataset, _SplitDataset):
        dataset = dataset.underlying_dataset
    return dataset

def _find_label_imbalance_indices_file(imbalance_ratio, seed, label_imbalance_dir):
    """Resolve the expected SAR-style label-imbalance index filename."""
    filename = f"seed{seed}_total_100000_ir_{int(imbalance_ratio)}_class_order_shuffle_yes.npy"

    candidate = os.path.join(label_imbalance_dir, filename)
    if os.path.isfile(candidate):
        return candidate
    return None


def _is_imagenetc_dataset(dataset, dataset_name=None):
    """Detect whether a dataset (or split) comes from ImageNet-C."""
    if dataset_name == "ImageNetC":
        return True

    underlying = _unwrap_split_dataset(dataset)
    root = getattr(underlying, "root", None)
    if isinstance(root, str):
        root_parts = [part.lower() for part in os.path.normpath(root).split(os.sep) if part]
        if "imagenetc" in root_parts or "imagenet-c" in root_parts:
            return True
    return False

def _extract_targets(dataset):
    """Efficiently extract labels without loading images"""
    if isinstance(dataset, _SplitDataset):
        # For _SplitDataset, extract targets for only the subset of data
        underlying = dataset.underlying_dataset
        if hasattr(underlying, 'manifest') and hasattr(underlying, 'label_map'):
            manifest_labels = underlying.manifest["label"].to_numpy()
            return [underlying.label_map[manifest_labels[k]] for k in dataset.keys]
        if hasattr(underlying, 'targets'):
            # Get targets for the keys in the split
            all_targets = underlying.targets
            return [all_targets[k] for k in dataset.keys]
        elif hasattr(underlying, 'tensors'):
            all_targets = underlying.tensors[1].numpy()
            return [all_targets[k] for k in dataset.keys]
        elif hasattr(underlying, 'dataset') and hasattr(underlying, 'indices'):
            # WILDSEnvironment
            y_array = underlying.dataset.y_array[underlying.indices].numpy()
            return [y_array[k] for k in dataset.keys]
        else:
            # Fallback: iterate through the split
            return [dataset[i][1] for i in range(len(dataset))]
    
    # Check dataset type and extract accordingly
    if hasattr(dataset, 'manifest') and hasattr(dataset, 'label_map'):
        return [dataset.label_map[v] for v in dataset.manifest["label"].to_numpy()]
    if hasattr(dataset, 'targets'):
        # ImageFolder, MNIST
        return dataset.targets
    elif hasattr(dataset, 'tensors'):
        # TensorDataset
        return dataset.tensors[1].numpy()
    elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # WILDSEnvironment
        return dataset.dataset.y_array[dataset.indices].numpy()
    else:
        # Fallback: iterate (slow but works)
        return [dataset[i][1] for i in range(len(dataset))]

class TemporalDirichletImbalanceDataset(torch.utils.data.Dataset):
    """
    Simulate non-i.i.d. dataset with using dirichlet distribution.
    """

    def __init__(self, base_dataset, temporal_dirichlet=None):
        # Extract metadata
        self.base_dataset = base_dataset
        self.num_classes = self._get_num_classes(base_dataset)

        # Extract labels efficiently
        targets = _extract_targets(base_dataset)

        # Generate shifted indices
        self.indices = self._apply_dirichlet_shift(
            targets, temporal_dirichlet
        )

        self.targets = [targets[i] for i in self.indices]


    def _get_num_classes(self, dataset):
        """Determine number of classes"""
        targets = _extract_targets(dataset)
        return len(np.unique(np.array(targets)))

    # Adapted from NOTE paper
    # https://github.com/TaesikGong/NOTE/blob/a714a2a2a9406903ba787b0bc240a95dd0342de5/learner/dnn.py#L161
    def _apply_dirichlet_shift(self, cl_labels, temporal_dirichlet):
        """
        Returns INDICES of samples after Dirichlet shift.
        These indices are relative to the base_dataset (0 to len(base_dataset)-1).
        """

        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
        min_size = -1
        cl_labels_np = np.asarray(cl_labels)
        N = len(cl_labels_np)
        
        # number of chunks to split data into. DIFFERS FROM NOTE-paper CODE, which uses num_chunks = num_classes. We use more chunks to adequately handle binary classification
        dirichlet_numchunks = max(1, N // self.num_classes)
        
        min_size_thresh = 0 #if conf.args.dataset in ['tinyimagenet'] else 10
        while min_size < min_size_thresh:  # prevent any chunk having too less data
            idx_batch = [[] for _ in range(dirichlet_numchunks)]
            idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
            for k in range(self.num_classes):
                idx_k = np.where(cl_labels_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(temporal_dirichlet, dirichlet_numchunks))

                # balance
                proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                        zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

                # store class-wise data
                for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                    idx_j.append(idx)

        indices = []
        # create temporally correlated toy dataset by shuffling classes
        for chunk in idx_batch_cls:
            cls_seq = list(range(self.num_classes))
            np.random.shuffle(cls_seq)
            for cls in cls_seq:
                idx = chunk[cls]
                indices.extend(idx)

        # trim data if num_sample is smaller than the original data size
        # num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
        # result_feats = result_feats[:num_samples]
        # result_cl_labels = result_cl_labels[:num_samples]
        # result_do_labels = result_do_labels[:num_samples]
        return indices
    
    def __getitem__(self, index):
        # indices[index] gives us the position in base_dataset we want
        original_index = self.indices[index]
        return self.base_dataset[original_index]
    
    def __len__(self):
        return len(self.indices)


class StaticDirichletImbalanceDataset(torch.utils.data.Dataset):
    """
    Subsample the dataset to introduce static class imbalance.
    Class proportions are drawn once from Dirichlet(static_dirichlet); small values
    produce severe imbalance, larger values approach a uniform class distribution.
    Unlike TemporalDirichletImbalanceDataset, this does NOT create temporal class correlations.
    """

    def __init__(self, base_dataset, static_dirichlet: float, seed: int = 0):
        self.base_dataset = base_dataset
        targets = _extract_targets(base_dataset)
        self.indices = self._subsample(targets, static_dirichlet, seed)

    def _subsample(self, targets, static_dirichlet: float, seed: int) -> list:
        rng_np = np.random.RandomState(seed)
        rng_py = random.Random(seed)

        targets_np = np.asarray(targets)
        unique_classes = sorted(np.unique(targets_np).tolist())
        num_classes = len(unique_classes)
        total = len(targets_np)

        proportions = rng_np.dirichlet(np.repeat(static_dirichlet, num_classes))
        self.proportions = proportions
        print(f"Subsampling with static Dirichlet imbalance (alpha={static_dirichlet}): class proportions = {proportions}")
        class_to_indices = {c: np.where(targets_np == c)[0].tolist() for c in unique_classes}

        result = []
        for k, cls in enumerate(unique_classes):
            target = round(proportions[k] * total)
            available = class_to_indices[cls]
            n = min(target, len(available))
            if n > 0:
                result.extend(rng_py.sample(available, n))

        rng_py.shuffle(result)
        return result

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class SARLabelShiftDataset(torch.utils.data.Dataset):
    """
    Simulate label distribution shift using per-class imbalance.
    Based on NeurIPS 2021: Online adaptation to label distribution shifts.
    """

    def __init__(self, base_dataset, imbalance_ratio=1, shuffle_class_order=True,
                 seed=0, dataset_name=None, label_imbalance_dir="./label_imbalance"):
        # Extract metadata
        self.base_dataset = base_dataset
        self.num_classes = self._get_num_classes(base_dataset)
        self.imbalance_ratio = imbalance_ratio
        self.shuffle_class_order = shuffle_class_order
        self.seed = seed
        self.dataset_name = dataset_name
        self.label_imbalance_dir = label_imbalance_dir
        self.indices_path = None
        
        # Extract labels efficiently
        targets = _extract_targets(base_dataset)
        
        # Generate shifted indices
        self.indices = self._apply_label_shift(targets)
    
    def _get_num_classes(self, dataset):
        """Determine number of classes"""
        targets = _extract_targets(dataset)
        return len(np.unique(np.array(targets)))

    def _load_precomputed_imagenetc_indices(self):
        """Load SAR-style precomputed label-shift indices for ImageNet-C if available."""
        if not _is_imagenetc_dataset(self.base_dataset, self.dataset_name):
            return None

        indices_path = _find_label_imbalance_indices_file(
            self.imbalance_ratio,
            self.seed,
            self.label_imbalance_dir,
        )
        if indices_path is None:
            return None

        indices = np.load(indices_path).astype(int).tolist()
        if len(indices) == 0:
            return None

        # If operating on a split dataset, map global indices to local split positions.
        if isinstance(self.base_dataset, _SplitDataset):
            key_to_local_idx = {int(global_idx): local_idx for local_idx, global_idx in enumerate(self.base_dataset.keys)}
            mapped = [key_to_local_idx[idx] for idx in indices if idx in key_to_local_idx]
            if len(mapped) == 0:
                return None
            self.indices_path = indices_path
            return mapped

        if min(indices) < 0 or max(indices) >= len(self.base_dataset):
            return None

        self.indices_path = indices_path
        return indices
    
    def _apply_label_shift(self, targets):
        """
        Apply per-class label shift with imbalance ratio.
        Returns indices of samples after label shift.
        """
        precomputed_indices = self._load_precomputed_imagenetc_indices()
        assert precomputed_indices is not None, "Debug failure: precomputed indices should be available for ImageNet-C datasets. Check the label_imbalance_dir and naming conventions."
        if precomputed_indices is not None:
            return precomputed_indices

        targets = np.array(targets)
        T = 100 * 1000  # Total number of samples
        
        # Create probability distribution for each class
        minor_class_prob = 1 / (self.imbalance_ratio + self.num_classes - 1)
        major_class_prob = minor_class_prob * self.imbalance_ratio
        
        # q_for_all_classes[i, j] = probability of sampling class j when in phase i
        q_for_all_classes = np.ones([self.num_classes, self.num_classes]) * minor_class_prob
        for i in range(self.num_classes):
            q_for_all_classes[i, i] = major_class_prob
        
        # Optionally shuffle the class order
        if self.shuffle_class_order:
            class_indices = list(range(self.num_classes))
            np.random.shuffle(class_indices)
            q_for_all_classes = q_for_all_classes[class_indices, :]
        
        # Create temporal distribution: repeat each class probability distribution
        num_for_repeat_each_q = 100

        num_total_repeats = T // self.num_classes // num_for_repeat_each_q
        if num_total_repeats == 0:
            num_total_repeats = 1
        
        q_all = np.concatenate([
            np.expand_dims(q_for_all_classes[i % self.num_classes, :], axis=0) 
            for _ in range(num_total_repeats)
            for i in range(self.num_classes) 
            for _ in range(num_for_repeat_each_q)
        ], axis=0)
        
        # Trim or extend to match dataset size
        if len(q_all) > T:
            q_all = q_all[:T]
        elif len(q_all) < T:
            # Repeat the last distribution
            extra = T - len(q_all)
            q_all = np.concatenate([q_all, np.tile(q_all[-1:], (extra, 1))], axis=0)
        
        # Sample class labels according to temporal distribution
        ys = np.array([np.random.choice(self.num_classes, p=q) for q in q_all])
        
        # Create indices mapping: organize by class
        class_to_indices = {}
        for cls in range(self.num_classes):
            class_to_indices[cls] = np.where(targets == cls)[0]
        
        # Generate sample indices based on sampled labels
        generated_indices = np.zeros(T, dtype=int)
        for cls in range(self.num_classes):
            num_needed = (ys == cls).sum()
            if num_needed == 0:
                continue
            
            available_indices = class_to_indices[cls]
            if len(available_indices) == 0:
                continue
            
            # Sample with replacement if needed
            sampled = np.random.choice(
                available_indices, 
                size=num_needed, 
                replace=(num_needed > len(available_indices))
            )
            generated_indices[ys == cls] = sampled
        
        return generated_indices.tolist()
    
    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.base_dataset[original_index]
    
    def __len__(self):
        return len(self.indices)

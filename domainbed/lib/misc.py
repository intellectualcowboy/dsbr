# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
import io
from shutil import copyfile

import numpy as np
import torch
import tqdm
from collections import Counter

from sklearn.metrics import roc_auc_score

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    print("Counting class frequencies...")
    dl = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    for _, y in tqdm.tqdm(dl):
        for _y in y:
            _y = int(_y)
            counts[_y] += 1
            classes.append(_y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    all_probs = []
    all_labels = []
    all_auc_weights = []

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                preds = p.gt(0).squeeze(1)
                correct += (preds.eq(y).float() * batch_weights).sum().item()

                probs = torch.sigmoid(p).squeeze(1)
                all_probs.append(probs.detach().cpu())
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()

                probs = torch.softmax(p, dim=1)
                all_probs.append(probs.detach().cpu())

            all_labels.append(y.detach().cpu())
            all_auc_weights.append(batch_weights.detach().cpu())
            total += batch_weights.sum().item()
    network.train()

    roc_auc = float('nan')
    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        try: 
            if all_probs.shape[1] == 2:
                # Binary classification - use probability of positive class
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multi-class - use all class probabilities
                roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', labels=np.arange(all_probs.shape[-1]))
        except ValueError as e:
            print(f"Error occurred while calculating ROC AUC: {e}")
            roc_auc = float('nan')

    return correct / total, roc_auc

class Tee(io.TextIOBase):
    def __init__(self, fname, mode="a", stream=sys.stdout):
        super().__init__()
        self.stream = stream  # the original stdout/stderr
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    # Delegate attributes to the underlying stream
    def __getattr__(self, name):
        return getattr(self.stream, name)
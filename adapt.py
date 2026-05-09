import os
import json
from argparse import Namespace
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as kl_divergence

import warnings
# Suppress deprecation warnings from outdated and pydantic
warnings.filterwarnings('ignore', category=UserWarning, module='outdated')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic._internal._generate_schema')

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import FastDataLoader, DataParallelPassthrough
from domainbed import adapt_algorithms as all_adapt_algorithms
from domainbed.scripts.unsupervised_adaptation import softmax_entropy
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import seaborn as sns
import time

import datetime
from torch.utils.data import ConcatDataset
from adapt_config import plot_config
from distribution_shifts import TemporalDirichletImbalanceDataset, StaticDirichletImbalanceDataset, SARLabelShiftDataset
import re
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def resolve_dataset_class(dataset_name):
    if dataset_name in vars(datasets):
        return vars(datasets)[dataset_name]
    else:
        raise NotImplementedError

def load_dataset(args, hparams):
    dataset_class = resolve_dataset_class(args.dataset)
    # disable data augmentation for evaluation and online adaptation
    hparams["data_augmentation"] = False
    dataset = dataset_class(args.data_dir, args.test_envs, hparams)
    return dataset

def load_run_record(input_dir):
    epochs_path = os.path.join(input_dir, 'results.jsonl')
    records = []
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    return args, records

def resolve_run_hparams(args):
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    if args.hparams:
        if type(args.hparams) == str:
            args.hparams = json.loads(args.hparams)
        hparams.update(args.hparams)
    return hparams

def plot_training_curves(training_results, environments, target_envs, backbone, dataset_name, output_dir):
    source_envs = [i for i in range(len(environments)) if i not in target_envs]
    steps = [log['step'] for log in training_results]

    accuracy = {}
    for i, name in enumerate(environments):
        accuracy[f"env{i}_in_acc"] = [log[f"env{i}_in_acc"] for log in training_results]
        accuracy[f"env{i}_out_acc"] = [log[f"env{i}_out_acc"] for log in training_results]

    accuracy["target_train_acc"] = np.mean([accuracy[f"env{i}_in_acc"] for i in target_envs], axis=0)
    accuracy["target_val_acc"] = np.mean([accuracy[f"env{i}_out_acc"] for i in target_envs], axis=0)

    accuracy["source_train_acc"] = np.mean([accuracy[f"env{i}_in_acc"] for i in source_envs], axis=0)
    accuracy["source_val_acc"] = np.mean([accuracy[f"env{i}_out_acc"] for i in source_envs], axis=0)

    for key in ["source_train_acc", "source_val_acc", "target_train_acc", "target_val_acc"]:
        plt.plot(steps, accuracy[key], label=key)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(f'Pre-Training Accuracy for {backbone} on {dataset_name}\nwith test env(s) {", ".join([environments[i] for i in target_envs])}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{backbone}_{dataset_name}_{"_".join([str(env) for env in target_envs])}_pretraining_accuracy.svg'))
    plt.show()

def get_dataset_splits(dataset, args):
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        name = dataset.ENVIRONMENTS[env_i]

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        in_splits.append((f"{name}_in", in_))
        out_splits.append((f"{name}_out", out))
    return in_splits, out_splits

def build_dataloader(env_indices, splits, batch_size, num_workers, temporal_dirichlet=None,
                     imbalance_ratio=None, static_dirichlet=None, use_random=True, seed=0, dataset_name=None):
    names = [splits[env_idx][0] for env_idx in env_indices]
    datasets = [splits[env_idx][1] for env_idx in env_indices]

    # Apply temporally-correlated Dirichlet class imbalance
    if temporal_dirichlet is not None:
        datasets = [TemporalDirichletImbalanceDataset(ds, temporal_dirichlet=temporal_dirichlet)
                   for ds in datasets]
        use_random = False
    # Apply static Dirichlet class imbalance (no temporal correlation)
    elif static_dirichlet is not None:
        datasets = [StaticDirichletImbalanceDataset(ds, static_dirichlet=static_dirichlet)
                   for ds in datasets]
        use_random = True
    # Apply SARLabelShiftDataset if imbalance_ratio is provided
    elif imbalance_ratio is not None:
        datasets = [
            SARLabelShiftDataset(
                ds,
                imbalance_ratio=imbalance_ratio,
                seed=seed,
                dataset_name=dataset_name,
            )
            for ds in datasets
        ]
        use_random = False
    
    lengths = [len(d) for d in datasets]
    dataset = ConcatDataset(datasets)
    # dataloader = FastDataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     random=use_random
    # )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=use_random,
                                                    num_workers=num_workers, pin_memory=True)
    return names, lengths, dataloader

def get_base_algorithm(args, dataset, hparams, device, input_dir=None):
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    
    algorithm.to(device)
    
    # use_data_parallel = bool(hparams.get("use_data_parallel", False))
    if not dataset.__class__.__name__.startswith("ImageNetC"):
        if hasattr(algorithm, 'network'):
            algorithm.network = DataParallelPassthrough(algorithm.network)
        else:
            for m in algorithm.children():
                m = DataParallelPassthrough(m)

    if input_dir is not None:
        ckpt = torch.load(os.path.join(input_dir, 'IID_best.pkl'), weights_only=False)
        algorithm_dict = ckpt['model_dict']
        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)
    else:
        assert dataset.__class__.__name__ == "ImageNetC", "Pre-trained model loading only implemented for ImageNet-C in this context"
    
    # Ensure model is on the correct device after loading
    algorithm.to(device)

    return algorithm

def eval_base_model(input_dir, device, use_cached=True, args=None, env_indices=None):
    """
    Evaluate the base model on the specified environments.

    Parameters
    ----------
    env_indices : list of int, optional
        Indices of environments to evaluate.  When *None* (default) all
        environments in the dataset are evaluated.
    """
    original_input_dir = input_dir
    if input_dir is None:
        assert args is not None, "Args must be provided if input_dir is None"
        backbone = args.hparams.get("backbone", "unknown") if isinstance(args.hparams, dict) else "unknown"
        input_dir = f"output/models/{args.dataset}_{backbone}_seed_0"
        os.makedirs(input_dir, exist_ok=True)
        training_results = []
    else:
        assert os.path.exists(os.path.join(input_dir, 'done'))
        args, training_results = load_run_record(input_dir)
    hparams = resolve_run_hparams(args)

    set_seed(0)

    dataset = load_dataset(args, hparams)
    in_splits, out_splits = get_dataset_splits(dataset, args)
    if training_results:
        plot_training_curves(training_results, dataset.ENVIRONMENTS, args.test_envs, hparams.get("backbone"), dataset.__class__.__name__, input_dir)

    base_algorithm = get_base_algorithm(args, dataset, hparams, device, input_dir=original_input_dir)

    results = {}
    if os.path.exists(os.path.join(input_dir, 'results_IID_best.json')) and use_cached:
        with open(os.path.join(input_dir, 'results_IID_best.json'), 'r') as f:
            results = json.load(f)

    # Determine which environments to evaluate
    all_env_indices = list(range(len(dataset)))
    target_indices = all_env_indices if env_indices is None else [
        i for i in env_indices if 0 <= i < len(dataset)
    ]

    eval_metrics = ["accuracy", "entropy", "roc_auc", "balanced_accuracy", "pred_entropy", "pred_entropy_normalized", "kl_divergence"]
    all_datasets = list(itertools.product([in_splits, out_splits], target_indices))
    pbar = tqdm(all_datasets, desc="Evaluating base model", position=0)
    for splits, env_idx in pbar:
        pbar.set_postfix({"env": dataset.ENVIRONMENTS[env_idx]})
        name = splits[env_idx][0]

        if use_cached:
            if all(f"{name}_{metric}" in results for metric in eval_metrics) and f"{name}_len" in results:
                continue

        names, length, dataloader = build_dataloader([env_idx], splits, batch_size=64, num_workers=dataset.N_WORKERS, use_random=False)
        name, length = names[0], length[0]

        try:
            eval_result, _ = evaluate_algorithm(base_algorithm, dataloader, device)
        finally:
            if hasattr(dataloader, "close"):
                dataloader.close()
        for metric, value in eval_result.items():
            results[f"{name}_{metric}"] = value
        results[f"{name}_len"] = length

        with open(os.path.join(input_dir, 'results_IID_best.json'), 'w') as f:
            json.dump(results, f, indent=4)

    return results

def evaluate_algorithm(network, dataloader, device, sample=None, return_entropy=False, return_topk=False, collect_embeddings=False):
    correct = 0
    total = 0
    ent = 0
    
    # Collect all predictions and labels
    all_probs = []
    all_labels = []
    all_entropies = []
    all_embeddings = []

    network.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            features = network.featurizer(x)
            if collect_embeddings:
                all_embeddings.append(features.detach().cpu().numpy())
            if hasattr(network, 'classifier'):
                outputs = network.classifier(features)
            else:
                outputs = features

            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Collect probabilities and labels for ROC AUC
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            entropy = softmax_entropy(outputs)
            if return_entropy:
                all_entropies.append(entropy.cpu())
            ent += entropy.sum().item()
    
    # Compute ROC AUC on all aggregated data
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    num_classes = all_probs.shape[1]
    true_dist = np.bincount(all_labels, minlength=num_classes) / len(all_labels)
    predictions = np.argmax(all_probs, axis=1)
    pred_dist = np.bincount(predictions, minlength=num_classes) / len(predictions)
    
    mask = true_dist > 0
    true_dist = true_dist[mask]  # Filter out classes not present in true labels
    pred_dist = pred_dist[mask]
    num_classes = len(true_dist)
    
    tvd = np.abs(true_dist - pred_dist).sum() / 2
    jensenshannon_divergence = jensenshannon(true_dist, pred_dist)
    kl = kl_divergence(pred_dist, true_dist) / np.log(num_classes)  # Normalize KL divergence by log(num_classes) 
    reverse_kl = kl_divergence(true_dist, pred_dist) / np.log(num_classes)  # Normalize reverse KL divergence by log(num_classes)
    
    balanced_pred_dist = pred_dist  / true_dist
    balanced_pred_dist = balanced_pred_dist / np.sum(balanced_pred_dist)  # Normalize to get a valid distribution
    pred_entropy = -np.sum(balanced_pred_dist * np.log(balanced_pred_dist + 1e-12))  # Add small constant to avoid log(0)
    pred_entropy_normalized = pred_entropy / np.log(num_classes)
    
    pred_entropy_orig = -np.sum(pred_dist * np.log(pred_dist + 1e-12)) / np.log(num_classes)

    details = {}
    if return_entropy:
        all_entropies = torch.cat(all_entropies, dim=0).numpy()
        details["entropy"] = all_entropies
    if return_topk:
        probs_sorted = np.argsort(all_probs, axis=1)[:, ::-1]
        topk = np.argwhere(probs_sorted == all_labels[:, None])[:, 1] + 1  # +1 for 1-based rank
        details["topk"] = topk
    if collect_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        details["embeddings"] = all_embeddings
        details["predictions"] = predictions
        
    # For binary: use probability for positive class
    # For multi-class: use probabilities with multi_class='ovr' or 'ovo'

    # distinct_labels = np.unique(all_labels)
    # assert np.array_equal(distinct_labels, np.arange(len(distinct_labels))), "Labels must be contiguous integers starting from 0"
    # all_probs = all_probs[:, distinct_labels]  # Ensure columns correspond to actual labels
    try:
        if all_probs.shape[1] == 2:
            # Binary classification - use probability of positive class
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multi-class - use all class probabilities
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', labels=np.arange(all_probs.shape[-1]))
    except ValueError:
        roc_auc = float('nan')  # Handle cases where ROC AUC cannot be computed
        
    balanced_acc = balanced_accuracy_score(all_labels, np.argmax(all_probs, axis=1))

    results = {
        "accuracy": correct / total,
        "entropy": ent / total,
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_acc,
        "pred_entropy": pred_entropy,
        "pred_entropy_normalized": pred_entropy_normalized,
        "pred_entropy_orig": pred_entropy_orig,
        "jensenshannon": jensenshannon_divergence,
        "kl_divergence": kl,
        "reverse_kl_divergence": reverse_kl,
        "tvd": tvd,
    }
    return results, details

def collect_grad_norm(network, optimizer=None):
    total_norm = 0.0
    if optimizer is not None:
        params = []
        for group in optimizer.param_groups:
            params.extend(group['params'])
    else:
        params = network.parameters()

    for p in params:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def run_adaptation_loop(
    network, 
    adapt_dataloader, 
    eval_dataloader, 
    device, 
    log_interval=-1, 
    num_epochs=1, 
    sample=None, 
    collect_grads=False, 
    collect_logits=False, 
    title=None, 
    worker_id=None, 
    collect_entropy=False, 
    collect_labels=False, 
    collect_outputs=False, 
    collect_embeddings=False,
    log_strategy="uniform",
    ):
    
    results = defaultdict(list)
    results["details"] = defaultdict(list)
    if log_interval <= 0:
        log_interval = len(adapt_dataloader)  # log only at the beginning and end
    network.eval()
    batch = 0
    
    all_labels = []
    all_outputs = []
    
    def is_log_step(batch):
        if batch == 0 or batch == len(adapt_dataloader) - 1:
            return True
        if log_strategy == "uniform":
            return (batch % log_interval) == 0
        elif log_strategy == "logarithmic":
            log = 0
            idx = 1
            while batch > log:
                log = np.round(log_interval ** (idx))
                idx += 1
            return batch == log
            
        raise ValueError(f"Unknown log strategy: {log_strategy}")
        
    for epoch in range(num_epochs):
        # print("Epoch", epoch)
        epoch_postfix = f"Epoch {epoch+1}/{num_epochs}" if num_epochs > 1 else ""
        description = f"{title} {epoch_postfix}".strip() if title else epoch_postfix
        for x, y in tqdm(adapt_dataloader, desc=description, position=worker_id, ):
            x = x.to(device)
            y = y.to(device)
            
            # Evaluation logging
            if is_log_step(batch):
                with torch.no_grad():
                    if eval_dataloader is not None:
                        test_results, details = evaluate_algorithm(network.model, eval_dataloader, device, sample=sample, collect_embeddings=collect_embeddings, return_entropy=collect_entropy)
                        for key, value in details.items():
                            results["details"][key].append(value)
                        for key, value in test_results.items():
                            results[key].append(value)
                    results["batch"].append(batch)

            # Main adaptation step
            outputs = network(x, adapt=True)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                outputs, num_filtered_1, num_filtered_2 = outputs
                # results["filter_log_1"] += num_filtered_1
                # results["filter_log_2"] += num_filtered_2
            
            # Collect metrics
            with torch.no_grad():
                if collect_grads:
                    optimizer = getattr(network, 'optimizer', None)
                    grad_norm = collect_grad_norm(network.model, optimizer=optimizer)
                    results["grad_norm"].append(grad_norm)
                if collect_entropy:
                    entropy = softmax_entropy(outputs)
                    results["batch_entropy"].append(entropy.mean().item())
                if collect_logits:
                    logits_norm = torch.norm(outputs, p=2, dim=1).mean().item()
                    results["logits_norm"].append(logits_norm)
                    
                all_labels.extend(y.cpu().numpy().tolist())
                preds = outputs.argmax(dim=1)
                all_outputs.extend(preds.cpu().numpy().tolist())
                
                if is_log_step(batch):
                    tta_balanced_accuracy = balanced_accuracy_score(all_labels, all_outputs) if all_outputs else 0.0
                    tta_accuracy = accuracy_score(all_labels, all_outputs) if all_outputs else 0.0
                    results["tta_accuracy"].append(tta_accuracy)
                    results["tta_balanced_accuracy"].append(tta_balanced_accuracy)
                    results["tta_accuracy"].append(tta_accuracy)
                    tqdm.write(f"Batch {batch}: TTA Accuracy={tta_accuracy:.4f}, TTA Balanced Accuracy={tta_balanced_accuracy:.4f}")
                    # print(f"Batch {batch}: TTA Accuracy={tta_accuracy:.4f}, TTA Balanced Accuracy={tta_balanced_accuracy:.4f}")
            batch += 1 
    if collect_labels:
        results["labels"] = all_labels
    if collect_outputs:
        results["outputs"] = all_outputs
    
    return results

def build_adaptation_algorithm(base_algorithm, adapt_algorithm_name, adapt_hparams, dataset, num_domains, device):
    adapt_algorithm_class = all_adapt_algorithms.get_algorithm_class(
        adapt_algorithm_name)
    adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
        num_domains, adapt_hparams, base_algorithm
    )
    # Ensure adapted algorithm is on the correct device
    adapted_algorithm.to(device)
    adapt_hparams['cached_loader'] = False
    adapted_algorithm.reset()
    return adapted_algorithm


def run_adaptation_group(input_dir, adapt_algorithm_name, num_epochs, log_interval, tta_batch_size, adapt_hparams, test_envs, seed, device, args=None, collect_grads=False, collect_logits=False, collect_entropy=False, temporal_dirichlet=None, imbalance_ratio=None, static_dirichlet=None, title=None, worker_id=None, collect_labels=False, collect_outputs=False, eval_hold_out=False):
    """
    Perform adaptation on specified test environments.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the pre-trained model
    adapt_algorithm_name : str
        Name of the adaptation algorithm
    num_epochs : int
        Number of adaptation epochs
    log_interval : int
        Logging interval
    tta_batch_size : int
        Batch size for adaptation
    adapt_hparams : dict
        Adaptation hyperparameters
    test_envs : list
        List of environment indices to adapt on (subset of args.test_envs)
    device : torch.device
        Device to run the adaptation on
    args : Namespace or None
        If input_dir is None, default ImageNet pre-trained model is used. args must be provided to determine dataset and other parameters 
    collect_grads : bool
        Whether to collect gradient norms
    collect_logits : bool
        Whether to collect logits norms
    collect_outputs : bool
        Whether to collect predicted class outputs
    temporal_dirichlet : float, optional
        Concentration parameter for temporally-correlated Dirichlet imbalance
    static_dirichlet : float, optional
        Concentration parameter for static Dirichlet imbalance
    imbalance_ratio : float, optional
        Imbalance ratio for SAR label shift
    worker_id : int or None
    Returns
    -------
    dict
        Adaptation results
    """

    group_start_time = time.perf_counter()

    if input_dir is None:
        assert args is not None, "Args must be provided if input_dir is None"
    else:
        assert os.path.exists(os.path.join(input_dir, 'done')), input_dir
        args, _ = load_run_record(input_dir)
    hparams = resolve_run_hparams(args)

    set_seed(seed)

    dataset = load_dataset(args, hparams)

    # Apply static Dirichlet imbalance to each full environment before any splitting,
    # so adapt and eval splits share the same class distribution.
    if static_dirichlet is not None:
        source_envs = [
            StaticDirichletImbalanceDataset(env, static_dirichlet, seed=seed)
            for env in dataset
        ]
    else:
        source_envs = list(dataset)

    in_splits, out_splits, full_splits = [], [], []
    for env_i, env in enumerate(source_envs):
        name = dataset.ENVIRONMENTS[env_i]
        out, in_ = misc.split_dataset(env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        in_splits.append((f"{name}_in", in_))
        out_splits.append((f"{name}_out", out))
        full_splits.append((f"{name}_full", env))

    if not eval_hold_out:
        t_adapt_names, t_adapt_lengths, t_adapt_dataloader = build_dataloader(
            test_envs, full_splits, tta_batch_size, dataset.N_WORKERS,
            temporal_dirichlet=temporal_dirichlet,
            imbalance_ratio=imbalance_ratio,
            seed=seed,
            dataset_name=args.dataset,
        )
        t_eval_dataloader = None
    else:
        t_adapt_names, t_adapt_lengths, t_adapt_dataloader = build_dataloader(
            test_envs, in_splits, tta_batch_size, dataset.N_WORKERS,
            temporal_dirichlet=temporal_dirichlet,
            imbalance_ratio=imbalance_ratio,
            seed=seed,
            dataset_name=args.dataset,
        )
        t_eval_names, t_eval_lengths, t_eval_dataloader = build_dataloader(test_envs, out_splits, tta_batch_size, dataset.N_WORKERS)

    base_algorithm = get_base_algorithm(args, dataset, hparams, device, input_dir=input_dir)

    assert adapt_hparams is not None, "Adaptation hyperparameters must be provided."

    if hparams is not None:
        hparams.update(adapt_hparams)
        adapt_hparams = hparams
    
    num_domains = len(dataset) - len(args.test_envs)
    adapt_algorithm = build_adaptation_algorithm(base_algorithm, adapt_algorithm_name, adapt_hparams, dataset, num_domains, device)

    log_interval = log_interval // tta_batch_size
    try:
        results = run_adaptation_loop(
            adapt_algorithm,
            t_adapt_dataloader,
            t_eval_dataloader,
            device,
            log_interval=log_interval,
            num_epochs=num_epochs,
            sample=None,
            collect_grads=collect_grads,
            collect_logits=collect_logits,
            collect_entropy=collect_entropy,
            collect_labels=collect_labels,
            collect_outputs=collect_outputs,
            title=title,
            worker_id=worker_id
        )
    finally:
        for dataloader in [t_adapt_dataloader, t_eval_dataloader]:
            if hasattr(dataloader, "close"):
                dataloader.close()

    results["num_epochs"] = num_epochs
    results["log_interval"] = log_interval
    results['tta_batch_size'] = tta_batch_size
    run_duration_seconds = time.perf_counter() - group_start_time
    results["run_duration_seconds"] = run_duration_seconds
    return results

def _weighted_metric(env_entries, splits, metric, entropy_scaling=1.0):
    values = []
    lengths = []
    for env_entry in env_entries:
        for split in splits:
            split_metrics = env_entry.get(split, {})
            if metric not in split_metrics:
                continue
            value = split_metrics[metric]
            if metric == 'entropy':
                value = value * entropy_scaling
            values.append(value)
            lengths.append(split_metrics.get('len', 1.0))
    if not values:
        return float('nan')
    return float(np.average(values, weights=lengths))

METRIC_DISPLAY_NAMES = {
    'accuracy': 'Accuracy',
    'entropy': 'Entropy',
    'roc_auc': 'ROC AUC',
    'balanced_accuracy': 'Balanced Accuracy',
}

def plot_adaptation_results(results, output_file, collect_filter_counts=False, collect_grads=False, collect_logits=False, collect_entropy=False, metrics=None, collect_labels=False, collect_outputs=False, display_legend=True, cmap=None):
    if metrics is None:
        metrics = ['balanced_accuracy', 'entropy']

    sns.set_theme()
    if cmap is not None:
        cmap_obj = plt.get_cmap(cmap)

    import pandas as pd
    df = pd.DataFrame(results)

    dataset_varies = df['dataset'].nunique() > 1
    train_envs_varies = df['train_envs'].map(tuple).nunique() > 1
    
    def create_title(entry, dataset_varies, train_envs_varies):
        title = ""
        if dataset_varies:
            title += entry['dataset'] + "\n"
        if train_envs_varies and entry["train_envs"][0] != -1:
            title += ','.join(map(str, entry['train_env_names']))
            title += " $\rightarrow$ "
        title += ','.join(map(str, entry['test_env_names']))
        return title


    df['title'] = df.apply(lambda row: create_title(row, dataset_varies, train_envs_varies), axis=1)

    groups = df.groupby('title')

    cols = len(groups)
    collect_grad_entropy_scatter = collect_grads and collect_entropy
    rows = len(metrics) + int(collect_filter_counts) + int(collect_grads) + int(collect_logits) + int(collect_entropy) + int(collect_grad_entropy_scatter) + int(collect_labels) + int(collect_outputs)

    fig, axes = plt.subplots(rows, cols, sharey='row', figsize=(cols * 4, rows * 4))

    for col, (title, group_df) in enumerate(groups):
        group_results = group_df.to_dict(orient='records')
        first_res = group_results[0]
        group_metadata = first_res.get('metadata', {})
        num_classes = group_metadata.get('num_classes')
        entropy_scaling = 1.0 / np.log(num_classes) if num_classes is not None and num_classes > 1 else 1.0
        eval_hold_out = group_metadata.get('eval_hold_out', True)

        if rows == 1 and cols == 1:
            my_axes = [axes]
        elif rows == 1:
            my_axes = [axes[col]]
        elif cols == 1:
            my_axes = axes
        else:
            my_axes = axes[:, col]

        my_axes[0].set_title(title)

        if collect_grad_entropy_scatter:
            my_axes[-2].set_xlabel("Adaptation Steps")
            my_axes[-1].set_xlabel("Gradient Norm")
        else:
            my_axes[-1].set_xlabel("Adaptation Steps")

        if col == 0:
            for m_idx, metric in enumerate(metrics):
                my_axes[m_idx].set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title()))

        group_base_results = first_res.get('base_results')
        if group_base_results:
            train_base = group_base_results.get('train', [])
            test_base = group_base_results.get('test', [])

            for m_idx, metric in enumerate(metrics):
                source_train = _weighted_metric(train_base, ['in'], metric, entropy_scaling)
                source_eval = _weighted_metric(train_base, ['out'], metric, entropy_scaling)
                if eval_hold_out:
                    no_adapt = _weighted_metric(test_base, ['out'], metric, entropy_scaling)
                else:
                    no_adapt = _weighted_metric(test_base, ['in', 'out'], metric, entropy_scaling)

                if np.isfinite(source_train):
                    my_axes[m_idx].axhline(source_train, label='Source Train Results', **plot_config.get('Source Train Results', {}))
                if np.isfinite(source_eval):
                    my_axes[m_idx].axhline(source_eval, label='Source Eval Results', **plot_config.get('Source Eval Results', {}))
                if np.isfinite(no_adapt):
                    my_axes[m_idx].axhline(no_adapt, label='No Adaptation', **plot_config.get('No Adaptation', {}))

        for res in group_results:
            adapt_result = res['result']

            batches = adapt_result.get('batch', [])

            steps = [batch * adapt_result['tta_batch_size'] for batch in batches]
            interleave = max(1, len(steps) // 20)
            steps = steps[::interleave]

            algo_label = res['label']
            plot_kwargs = plot_config.get(algo_label, {}).copy()
            if cmap is not None and 'threshold' in res:
                plot_kwargs['color'] = cmap_obj(res['threshold'])

            for m_idx, metric in enumerate(metrics):
                metric_values = adapt_result.get(metric, [])
                if not metric_values:
                    continue
                if metric == 'entropy':
                    metric_values = [v * entropy_scaling for v in metric_values]
                my_axes[m_idx].plot(steps, metric_values[::interleave], label=algo_label, **plot_kwargs)

            row_offset = len(metrics)
            if collect_grads:
                grads = adapt_result['grad_norm']
                steps = adapt_result['tta_batch_size'] * np.arange(len(grads))
                my_axes[row_offset].plot(steps, grads, label=algo_label, **plot_kwargs)
                if col == 0:
                    my_axes[row_offset].set_ylabel('Gradient Norm')
                row_offset += 1
            if collect_logits:
                logits = adapt_result['logits_norm']
                steps = adapt_result['tta_batch_size'] * np.arange(len(logits))
                my_axes[row_offset].plot(steps, logits, label=algo_label, **plot_kwargs)
                if col == 0:
                    my_axes[row_offset].set_ylabel('Logits Norm')
                row_offset += 1
            if collect_entropy:
                ent = [e * entropy_scaling for e in adapt_result['batch_entropy']]
                steps = adapt_result['tta_batch_size'] * np.arange(len(ent))
                my_axes[row_offset].plot(steps, ent, label=algo_label, **plot_kwargs)
                if col == 0:
                    my_axes[row_offset].set_ylabel('Adapt Entropy')
                row_offset += 1
            if collect_grad_entropy_scatter:
                grads = np.asarray(adapt_result['grad_norm'])
                ent = np.asarray([e * entropy_scaling for e in adapt_result['batch_entropy']])
                assert len(grads) == len(ent), 'Gradient norms and entropies must have the same length for scatter plot'
                my_axes[row_offset].scatter(
                    grads,
                    ent,
                    label=algo_label,
                    alpha=0.7,
                    **plot_kwargs,
                )
                if col == 0:
                    my_axes[row_offset].set_ylabel('Adapt Entropy')
                row_offset += 1
            if collect_labels:
                labels = adapt_result['labels']
                steps = np.arange(len(labels))
                my_axes[row_offset].scatter(
                    steps,
                    labels,
                    label=algo_label,
                    alpha=0.5,
                    s=1,
                    **plot_kwargs,
                )
                if col == 0:
                    my_axes[row_offset].set_ylabel('Labels')
                row_offset += 1
            if collect_outputs:
                outputs = adapt_result['outputs']
                steps = np.arange(len(outputs))
                my_axes[row_offset].scatter(
                    steps,
                    outputs,
                    label=algo_label,
                    alpha=0.5,
                    s=1,
                    **plot_kwargs,
                )
                if col == 0:
                    my_axes[row_offset].set_ylabel('Outputs')
                row_offset += 1
            if collect_filter_counts:
                raise NotImplementedError('Plotting SAR filter counts not implemented in this context.')

    if display_legend:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            frameon=False,
        )

    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def _parse_train_envs(train_envs):
    if isinstance(train_envs, int):
        return [[train_envs]]
    if isinstance(train_envs, str):
        return [[int(i) for i in train_envs.split(',')]]
    if not train_envs:
        return []
    if isinstance(train_envs[0], list):
        return train_envs
    return [train_envs]


def resolve_test_env_groups(test_envs, available_test_envs, mixed_envs):
    if test_envs is None or test_envs == 'all':
        parsed_test_envs = [env for env in available_test_envs]
    elif isinstance(test_envs, int):
        parsed_test_envs = [test_envs]
    elif isinstance(test_envs, str):
        parsed_test_envs = [int(i) for i in test_envs.split(',')]
    else:
        parsed_test_envs = test_envs

    if not isinstance(parsed_test_envs[0], list):
        parsed_test_envs = [parsed_test_envs] if mixed_envs else [[env] for env in parsed_test_envs]

    for sublist in parsed_test_envs:
        assert set(sublist).issubset(available_test_envs), \
            f"Test environments {sublist} must be subset of environments not in train_envs {available_test_envs}"

    return parsed_test_envs


def build_input_dir(dataset, model, available_test_envs, pretraining_seed=0):
    return (
        f"output/models/data_{dataset}_model_{model}_test_env_{'_'.join(map(str, sorted(available_test_envs)))}"
        f"_seed_{pretraining_seed}"
    )


def _build_imagenetc_args(dataset, backbone, num_envs):
    '''Build arguments for ImageNet-C dataset as it does not follow the same pre-training and evaluation structure as other datasets in DomainBed.'''
    return Namespace(**{
        "algorithm": "ERM",
        "dataset": dataset,
        "holdout_fraction": 0.2,
        "hparams": {"backbone": backbone, "use_image_net_pretrained": True},
        "data_dir": "data",
        "test_envs": list(range(num_envs)),
        "seed": 0,
        "hparams_seed": 0,
        "trial_seed": 0,
    })


def get_default_adaptation_lr(model, batch_size, algorithm_name=None):
    """Return default adaptation learning rate based on model and batch size."""
    model_lower = model.lower()
    if "resnet" in model_lower:
        default_lr = (0.00025 / 64) * batch_size * 2 if batch_size < 32 else 0.00025
    elif "vit" in model_lower:
        default_lr = (0.001 / 64) * batch_size
    else:
        default_lr = (0.00025 / 64) * batch_size * 2 if batch_size < 32 else 0.00025

    if algorithm_name is not None and "SAR" in algorithm_name and batch_size == 1:
        default_lr *= 2
    return default_lr


def collect_base_results_for_group(base_results_dict, environments, env_group):
    """Collect per-environment base metrics for a specific environment group."""
    grouped = []

    for env_idx in env_group:
        if env_idx < 0 or env_idx >= len(environments):
            # Sentinel values (e.g. -1) or out-of-range indices have no base split metrics.
            continue
        env_name = environments[env_idx]
        in_prefix = f"{env_name}_in_"
        out_prefix = f"{env_name}_out_"

        grouped.append({
            "in": {},
            "out": {}
        })
        for key, value in base_results_dict.items():
            if key.startswith(in_prefix):
                metric_name = key[len(in_prefix):]
                grouped[-1]["in"][metric_name] = value
            if key.startswith(out_prefix):
                metric_name = key[len(out_prefix):]
                grouped[-1]["out"][metric_name] = value

    return grouped


def run_adaptation_study(model, dataset, train_envs, test_envs=None, adapt_algorithms_with_hparams=None,
                        num_epochs=1, log_interval=-1, batch_size=32,
                        mixed_envs=False, use_temporal_dirichlet_imbalance=False, temporal_dirichlet=4.0,
                        use_label_shift=False, imbalance_ratio=10.0,
                        use_static_dirichlet_imbalance=False, static_dirichlet=1.0,
                        collect_grads=False, collect_logits=False, collect_outputs=False, collect_entropy=False, worker_id=None,
                        seed=0, collect_labels=False, eval_hold_out=False, pretraining_seed=0):
    """
    Run adaptation study for specified algorithms and environments.
    """
    active_shifts = sum([use_temporal_dirichlet_imbalance, use_label_shift, use_static_dirichlet_imbalance])
    if active_shifts > 1:
        raise ValueError("At most one of use_temporal_dirichlet_imbalance, use_label_shift, use_static_dirichlet_imbalance may be active.")

    worker_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

    dataset_class = resolve_dataset_class(dataset)
    environments = dataset_class.ENVIRONMENTS
    num_envs = len(environments)

    train_env_groups = _parse_train_envs(train_envs)
    all_env_indices = set(range(num_envs))

    print(f"{worker_prefix}Environments: {environments}")
    print(f"{worker_prefix}Training environment groups: {train_env_groups}")
    print(f"{worker_prefix}Algorithms to run:")
    for algo_config in adapt_algorithms_with_hparams:
        print(f"{worker_prefix}  {algo_config['name']}: {algo_config['hparams']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(device)
    else:
        gpu_device_name = "cpu"

    temporal_dirichlet_val = temporal_dirichlet if use_temporal_dirichlet_imbalance else None
    imbalance_ratio_val = imbalance_ratio if use_label_shift else None
    static_dirichlet_val = static_dirichlet if use_static_dirichlet_imbalance else None

    timestamp = datetime.datetime.now().isoformat()
    shared_metadata = {
        'num_classes': None,
        'num_epochs': num_epochs,
        'log_interval': log_interval,
        'gpu_device_name': gpu_device_name,
        'mixed_envs': mixed_envs,
        'use_temporal_dirichlet_imbalance': use_temporal_dirichlet_imbalance,
        'temporal_dirichlet': temporal_dirichlet if use_temporal_dirichlet_imbalance else None,
        'use_label_shift': use_label_shift,
        'imbalance_ratio': imbalance_ratio if use_label_shift else None,
        'use_static_dirichlet_imbalance': use_static_dirichlet_imbalance,
        'static_dirichlet': static_dirichlet if use_static_dirichlet_imbalance else None,
        'eval_hold_out': eval_hold_out,
        'collect_outputs': collect_outputs,
        'seed': seed,
        'collect_labels': collect_labels,
        'collect_entropy': collect_entropy,
        'timestamp': timestamp,
    }

    results = []
    for train_idx, train_env_group in enumerate(train_env_groups):
        train_env_set = set(train_env_group)
        available_test_envs = sorted(list(all_env_indices - train_env_set))
        resolved_test_env_groups = resolve_test_env_groups(test_envs, available_test_envs, mixed_envs)

        input_dir = build_input_dir(dataset, model, available_test_envs, pretraining_seed=pretraining_seed)
        args = None
        if dataset in ['ImageNetC']:
            input_dir = None
            args = _build_imagenetc_args(dataset, model, num_envs)

        print(f"{worker_prefix}Train group {train_idx+1}/{len(train_env_groups)}: {sorted(train_env_group)}")
        print(f"{worker_prefix}Test environment groups: {resolved_test_env_groups}")
        print(f"{worker_prefix}Input directory: {input_dir}")

        required_env_indices = sorted(
            ({e for e in train_env_group if 0 <= e < num_envs}) |
            ({e for group in resolved_test_env_groups for e in group if 0 <= e < num_envs})
        )
        print(f"{worker_prefix}Evaluating base model on envs {required_env_indices}...")
        base_results = eval_base_model(input_dir, device, use_cached=True, args=args,
                                        env_indices=required_env_indices)
        print(f"{worker_prefix}Base model evaluation complete.")

        train_env_names = [environments[e] if e >= 0 else "Train Env" for e in train_env_group]
        base_results_train = collect_base_results_for_group(base_results, environments, train_env_group)

        for test_idx, test_env_group in enumerate(resolved_test_env_groups):
            test_env_names = [environments[e] for e in test_env_group]
            group_base_results = {
                "train": base_results_train,
                "test": collect_base_results_for_group(base_results, environments, test_env_group)
            }
            for adapt_idx, algo_config in enumerate(adapt_algorithms_with_hparams):
                algo_name = algo_config['name']
                algo_hparams = algo_config['hparams'].copy()
                if algo_hparams.get('lr') is None:
                    algo_hparams['lr'] = get_default_adaptation_lr(model, batch_size, algorithm_name=algo_name)

                title = (
                    f"{dataset}({','.join(map(str, train_env_group))}) - "
                    f"Train {train_idx+1}/{len(train_env_groups)}, "
                    f"Test {test_idx+1}/{len(resolved_test_env_groups)}, "
                    f"Algo {adapt_idx+1}/{len(adapt_algorithms_with_hparams)}"
                )

                adaptation_result = run_adaptation_group(
                    input_dir,
                    algo_name,
                    num_epochs,
                    log_interval,
                    batch_size,
                    adapt_hparams=algo_hparams,
                    test_envs=test_env_group,
                    seed=seed,
                    device=device,
                    args=args,
                    collect_grads=collect_grads,
                    collect_logits=collect_logits,
                    collect_outputs=collect_outputs,
                    collect_entropy=collect_entropy,
                    temporal_dirichlet=temporal_dirichlet_val,
                    imbalance_ratio=imbalance_ratio_val,
                    static_dirichlet=static_dirichlet_val,
                    title=title,
                    worker_id=worker_id,
                    collect_labels=collect_labels,
                    eval_hold_out=eval_hold_out,
                )

                my_metadata = dict(shared_metadata)
                my_metadata['run_duration_seconds'] = adaptation_result.get('run_duration_seconds')

                results.append({
                    'algorithm': algo_name,
                    'hparams': algo_hparams,
                    'label': create_algorithm_label(algo_name, algo_hparams),
                    'dataset': dataset,
                    'model': model,
                    'test_envs': test_env_group,
                    'test_env_names': test_env_names,
                    'train_envs': train_env_group,
                    'train_env_names': train_env_names,
                    'batch_size': batch_size,
                    'base_results': group_base_results,
                    'metadata': my_metadata,
                    'result': adaptation_result,
                })

    return results


def create_algorithm_label(algo_name, hparams):
    """Create a readable label for an algorithm based on its hyperparameters."""
    # Start with algorithm name
    label = algo_name
    
    # Add key hyperparameters to label
    hparam_strs = []
    for key, value in sorted(hparams.items()):
        if key == 'lr':
            hparam_strs.append(f"lr={value:.0e}")
        elif key in ['margin_e0', 'deyo_margin']:
            hparam_strs.append(f"{key}={value:.2f}")
        else:
            hparam_strs.append(f"{key}={value}")
    
    if hparam_strs:
        label += f" ({', '.join(hparam_strs)})"
    
    return label


if __name__ == "__main__":
    # read cmdline args
    parser = argparse.ArgumentParser(description='Visualize Unsupervised Adaptation')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_envs', type=str, required=True,
                       help='Training environment indices (e.g., "0,1"). Determines which pre-trained model to load.')
    parser.add_argument('--test_envs', type=str, required=False, default='all',
                       help='Test environments: "all", comma-separated indices, or 2D array format')
    parser.add_argument('--adapt_algorithms', type=str, default='all')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--collect_grads', action='store_true', default=False)
    parser.add_argument('--collect_logits', action='store_true', default=False)
    parser.add_argument('--collect_labels', action='store_true', default=False)
    parser.add_argument('--collect_outputs', action='store_true', default=False)
    parser.add_argument('--collect_entropy', action='store_true', default=False)
    parser.add_argument('--mixed_envs', action='store_true', default=False)
    parser.add_argument('--eval_hold_out', action='store_true', default=False,
                       help='Evaluate adaptation on a holdout split instead of using everything for adaptation.')
    parser.add_argument('--use_temporal_dirichlet_imbalance', action='store_true', default=False,
                       help='Apply temporally-correlated Dirichlet class imbalance to the adaptation data.')
    parser.add_argument('--temporal_dirichlet', type=float, default=4.0,
                       help='Concentration parameter for temporal Dirichlet imbalance (default: 4.0).')
    parser.add_argument('--use_label_shift', action='store_true', default=False,
                       help='Apply SAR-style label shift to the adaptation data.')
    parser.add_argument('--imbalance_ratio', type=float, default=10.0,
                       help='Imbalance ratio for SAR label shift (default: 10.0).')
    parser.add_argument('--use_static_dirichlet_imbalance', action='store_true', default=False,
                       help='Apply static Dirichlet class imbalance to the adaptation data (no temporal correlation).')
    parser.add_argument('--static_dirichlet', type=float, default=1.0,
                       help='Concentration parameter for static Dirichlet imbalance (default: 1.0). Small values produce severe imbalance.')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate for adaptation. If not set, computed based on model and batch size.')
    parser.add_argument('--hparams', type=str, default=None,
                       help='JSON object with extra adaptation hyperparameters to merge into each algorithm config.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for adaptation runs.')
    parser.add_argument('--pretraining_seed', type=int, default=0,
                       help='Seed used in the pre-trained model -> directory suffix (_seed_<seed>).')
    adapt_args = parser.parse_args()

    # Parse algorithm specifications
    if adapt_args.adapt_algorithms == 'all':
        adapt_algorithm_specs = ["SAR_NO_SAM", "SAR", "COME", "TentFull", "COME_NO_REG", "DeYO_OCC", "DeYO_PATCH", "DeYO_PIXEL"]
    else:
        adapt_algorithm_specs = adapt_args.adapt_algorithms.split(',')

    # Parse algorithm specifications to extract algorithm name and lr_factor
    # Format: "AlgorithmName" or "AlgorithmName_0.5lr"
    lr_pattern = re.compile(r'^(.+?)_([0-9]*\.?[0-9]+)lr$')
    
    parsed_algorithms = []
    for spec in adapt_algorithm_specs:
        match = lr_pattern.match(spec)
        if match:
            algorithm_name = match.group(1)
            try:
                lr_factor = float(match.group(2))
            except ValueError:
                print(f"Warning: Could not parse lr_factor from '{spec}', using 1.0")
                lr_factor = 1.0
        else:
            algorithm_name = spec
            lr_factor = 1.0
        parsed_algorithms.append((algorithm_name, lr_factor))

    print("Parsed Adaptation Algorithms and LR Factors:")
    for alg_name, lr_factor in parsed_algorithms:
        print(f"  Algorithm = {alg_name}, LR Factor = {lr_factor}")

    default_lr = adapt_args.lr

    extra_adapt_hparams = {}
    if adapt_args.hparams is not None:
        try:
            extra_adapt_hparams = json.loads(adapt_args.hparams)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON passed to --hparams: {e}") from e
        if not isinstance(extra_adapt_hparams, dict):
            raise ValueError("--adapt_hparams_json must decode to a JSON object (dictionary).")

    # Build algorithm configurations with hyperparameters
    adapt_algorithms_with_hparams = []
    for algo_name, lr_factor in parsed_algorithms:
        if default_lr is None:
            base_lr = get_default_adaptation_lr(
                adapt_args.model,
                adapt_args.batch_size,
                algorithm_name=algo_name
            )
        else:
            base_lr = default_lr
        my_lr = base_lr * lr_factor

        algo_hparams = {'lr': my_lr}
        algo_hparams.update(extra_adapt_hparams)
        adapt_algorithms_with_hparams.append({
            'name': algo_name,
            'hparams': algo_hparams
        })

    # Run the adaptation study
    study_arguments = vars(adapt_args)
    study_arguments.pop('adapt_algorithms')  # already processed
    study_arguments.pop('lr')  # already processed
    study_arguments.pop('hparams')  # already processed
    
    results = run_adaptation_study(
        adapt_algorithms_with_hparams=adapt_algorithms_with_hparams,
        **study_arguments
    )

    if not results:
        raise RuntimeError("run_adaptation_study returned no results.")

    first_result = results[0]
    metadata = first_result['metadata']
    
    # Create output directory
    output_dir = f"output/adaptation/{first_result['dataset']}_{first_result['model']}/bs_{first_result['batch_size']}_epochs={metadata['num_epochs']}"
    if first_result.get('train_envs'):
        output_dir += f"_train_{'_'.join(map(str, first_result['train_envs']))}"
    if metadata['mixed_envs']:
        output_dir += "_mixed_envs"
    if metadata['use_temporal_dirichlet_imbalance']:
        output_dir += f"_temporal_dirichlet_{metadata['temporal_dirichlet']}"
    if metadata['use_label_shift']:
        output_dir += f"_label_imbalance_{metadata['imbalance_ratio']}"
    if metadata['use_static_dirichlet_imbalance']:
        output_dir += f"_static_dirichlet_{metadata['static_dirichlet']}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    timestamp = metadata['timestamp']

    # Save results to output dir
    print(results)
    with open(os.path.join(output_dir, f"adaptation_results_{timestamp}.json"), "w") as f:
        # Convert tensors to lists
        for result_entry in results:
            result = result_entry['result']
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()
        json.dump(results, f, indent=4)
        
    plot_adaptation_results(
        results,
        os.path.join(output_dir, f"adaptation_results_{timestamp}.pdf"),
        collect_filter_counts=False,
        collect_grads=adapt_args.collect_grads,
        collect_logits=adapt_args.collect_logits,
        collect_entropy=adapt_args.collect_entropy,
        collect_labels=adapt_args.collect_labels,
        collect_outputs=adapt_args.collect_outputs,
    )


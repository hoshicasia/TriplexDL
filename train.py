#!/usr/bin/env python3
import logging
import warnings
from collections import Counter, defaultdict

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, open_dict
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.datasets.collate import collate_fn
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, set_worker_seed

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _metrics_at_threshold_np(probs, labels, threshold):
    """Compute accuracy/precision/recall/F1 for binary probs at a threshold."""
    preds = (probs >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _find_best_threshold_np(probs, labels):
    """Find threshold in [0.01, 0.99] maximizing F1."""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        f1 = _metrics_at_threshold_np(probs, labels, float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def _resolve_dataset_paths(dataset_cfg):
    """Resolve dataset paths to absolute paths for Hydra run directories."""
    for key in ("pos_fasta_path", "neg_fasta_path", "bed_dir"):
        if key in dataset_cfg and dataset_cfg[key] is not None:
            dataset_cfg[key] = to_absolute_path(str(dataset_cfg[key]))
    return dataset_cfg


def _balance_split(indices, all_labels, target_ratio, split_name, seed):
    """
    Balance a split to achieve target neg:pos ratio using undersampling.
    """
    np.random.seed(seed)

    split_labels = [all_labels[i] for i in indices]
    pos_indices = [idx for idx, label in zip(indices, split_labels) if label == 1]
    neg_indices = [idx for idx, label in zip(indices, split_labels) if label == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            f"{split_name}: Cannot balance - only one class present (pos={n_pos}, neg={n_neg})"
        )
        return indices

    current_ratio = n_neg / n_pos
    logger.info(
        f"{split_name} before balancing: Pos={n_pos}, Neg={n_neg}, Ratio={current_ratio:.2f}:1"
    )

    if current_ratio > target_ratio:
        target_neg = int(n_pos * target_ratio)
        target_neg = min(target_neg, n_neg)  # Can't sample more than available
        neg_indices_sampled = np.random.choice(
            neg_indices, size=target_neg, replace=False
        ).tolist()
        balanced_indices = pos_indices + neg_indices_sampled
    else:
        logger.info(
            f"{split_name}: Ratio {current_ratio:.2f}:1 is better than target {target_ratio}:1, keeping all samples"
        )
        balanced_indices = (
            indices.tolist() if isinstance(indices, np.ndarray) else indices
        )

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    balanced_labels = [all_labels[i] for i in balanced_indices]
    n_pos_new = sum(balanced_labels)
    n_neg_new = len(balanced_labels) - n_pos_new
    new_ratio = n_neg_new / n_pos_new if n_pos_new > 0 else 0
    logger.info(
        f"{split_name} after balancing: Pos={n_pos_new}, Neg={n_neg_new}, Ratio={new_ratio:.2f}:1"
    )

    return balanced_indices


def _quantile_bin(values, n_quantiles: int = 4):
    """Quantize a numeric vector into quantile bins [0..n_quantiles-1]."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.array([], dtype=np.int32)
    if n_quantiles <= 1:
        return np.zeros(values.shape[0], dtype=np.int32)

    probs = np.linspace(0, 1, n_quantiles + 1)[1:-1]
    edges = np.quantile(values, probs)
    edges = np.unique(edges)
    if edges.size == 0:
        return np.zeros(values.shape[0], dtype=np.int32)

    return np.searchsorted(edges, values, side="right").astype(np.int32)


def _build_safe_strata(strata, class_labels, min_count=3):
    """Replace rare strata with class-only to make stratify stable."""
    counts = Counter(strata)
    safe = []
    for s, c in zip(strata, class_labels):
        if counts[s] >= min_count:
            safe.append(s)
        else:
            safe.append(f"class_{int(c)}")
    return safe


def _genomic_bin_representative_split(full_dataset, labels, config, use_validation):
    """Representative group split by genomic bins."""
    bin_size = config.get("bin_size", 2000000)
    test_frac = config.get("test_frac", 0.15)
    val_frac = config.get("val_frac", 0.15)

    n_quantiles = int(config.get("rep_split_n_quantiles", 4))
    min_stratum_count = int(config.get("rep_split_min_stratum_count", 3))

    logger.info(
        "Using representative genomic bin split: "
        f"bin_size={bin_size / 1e6:.0f}Mb, test={test_frac:.0%}, val={val_frac:.0%}, "
        f"quantiles={n_quantiles}"
    )

    bin_to_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        item = full_dataset.data[idx]
        chrom, start, end = item[0], int(item[1]), int(item[2])
        mid = (start + end) // 2
        bin_idx = mid // bin_size
        bid = f"{chrom}_{bin_idx}"
        bin_to_indices[bid].append(idx)

    unique_bins = sorted(bin_to_indices.keys())

    bin_labels = []
    bin_pos_rate = []
    bin_mean_len = []
    bin_mean_gc = []

    for b in unique_bins:
        idxs = bin_to_indices[b]
        y = np.array([labels[i] for i in idxs], dtype=np.float32)
        lengths = np.array(
            [int(full_dataset.data[i][2]) - int(full_dataset.data[i][1]) for i in idxs],
            dtype=np.float32,
        )

        gc_vals = []
        for i in idxs:
            seq = str(full_dataset.data[i][4])
            seq_len = max(len(seq), 1)
            gc = (seq.count("G") + seq.count("C")) / seq_len
            gc_vals.append(gc)
        gc_vals = np.array(gc_vals, dtype=np.float32)

        bin_labels.append(1 if y.sum() > 0 else 0)
        bin_pos_rate.append(float(y.mean()))
        bin_mean_len.append(float(lengths.mean()))
        bin_mean_gc.append(float(gc_vals.mean()))

    bin_labels = np.array(bin_labels, dtype=np.int32)
    pos_q = _quantile_bin(bin_pos_rate, n_quantiles=n_quantiles)
    len_q = _quantile_bin(bin_mean_len, n_quantiles=n_quantiles)
    gc_q = _quantile_bin(bin_mean_gc, n_quantiles=n_quantiles)

    strata = [
        f"c{int(c)}_p{int(p)}_l{int(length_bin)}_g{int(g)}"
        for c, p, length_bin, g in zip(bin_labels, pos_q, len_q, gc_q)
    ]
    safe_strata = _build_safe_strata(strata, bin_labels, min_count=min_stratum_count)

    logger.info(
        f"  Total bins: {len(unique_bins)} "
        f"(pos_bins={int(bin_labels.sum())}, neg_bins={len(bin_labels) - int(bin_labels.sum())})"
    )
    logger.info(
        f"  Unique strata: {len(set(strata))}, safe strata: {len(set(safe_strata))}"
    )

    def _choose_stratify_labels(indices, detailed_labels, class_labels, stage_name):
        idx_list = list(indices)
        if len(idx_list) < 2:
            logger.warning(
                f"  {stage_name}: too few bins for stratification, using random split"
            )
            return None

        det = [detailed_labels[i] for i in idx_list]
        det_counts = Counter(det)
        if len(det_counts) >= 2 and min(det_counts.values()) >= 2:
            return det

        cls = [int(class_labels[i]) for i in idx_list]
        cls_counts = Counter(cls)
        if len(cls_counts) >= 2 and min(cls_counts.values()) >= 2:
            logger.info(f"  {stage_name}: fallback to class-only stratification")
            return cls

        logger.warning(
            f"  {stage_name}: no valid stratification labels, using random split"
        )
        return None

    bin_indices = np.arange(len(unique_bins))
    if use_validation and val_frac > 0:
        strat_first = _choose_stratify_labels(
            indices=bin_indices,
            detailed_labels=safe_strata,
            class_labels=bin_labels,
            stage_name="Repr.bin. split stage 1",
        )
        train_bins, temp_bins = train_test_split(
            bin_indices,
            test_size=test_frac + val_frac,
            random_state=config.seed,
            stratify=strat_first,
        )
        strat_second = _choose_stratify_labels(
            indices=temp_bins,
            detailed_labels=safe_strata,
            class_labels=bin_labels,
            stage_name="Repr.bin. split stage 2",
        )
        relative_val = val_frac / (test_frac + val_frac)
        val_bins, test_bins = train_test_split(
            temp_bins,
            test_size=1 - relative_val,
            random_state=config.seed,
            stratify=strat_second,
        )
    else:
        strat_single = _choose_stratify_labels(
            indices=bin_indices,
            detailed_labels=safe_strata,
            class_labels=bin_labels,
            stage_name="Repr.bin. split single stage",
        )
        train_bins, test_bins = train_test_split(
            bin_indices,
            test_size=test_frac,
            random_state=config.seed,
            stratify=strat_single,
        )
        val_bins = np.array([], dtype=int)

    train_set, val_set = set(train_bins), set(val_bins)
    train_indices, val_indices, test_indices = [], [], []
    for bi, bname in enumerate(unique_bins):
        idxs = bin_to_indices[bname]
        if bi in train_set:
            train_indices.extend(idxs)
        elif bi in val_set:
            val_indices.extend(idxs)
        else:
            test_indices.extend(idxs)

    for split_name, split_bins in [
        ("Train", train_bins),
        ("Val", val_bins),
        ("Test", test_bins),
    ]:
        n_pos_bins = int(sum(1 for i in split_bins if bin_labels[i] == 1))
        n_neg_bins = int(len(split_bins) - n_pos_bins)
        logger.info(
            f"  {split_name}: {len(split_bins)} bins (pos={n_pos_bins}, neg={n_neg_bins})"
        )

    def _split_stats(indices):
        if len(indices) == 0:
            return {
                "n": 0,
                "pos_rate": 0.0,
                "mean_len": 0.0,
                "mean_gc": 0.0,
                "len_vals": np.array([], dtype=np.float32),
                "gc_vals": np.array([], dtype=np.float32),
            }
        y = np.array([labels[i] for i in indices], dtype=np.float32)
        lens = np.array(
            [
                int(full_dataset.data[i][2]) - int(full_dataset.data[i][1])
                for i in indices
            ],
            dtype=np.float32,
        )
        gcs = []
        for i in indices:
            seq = str(full_dataset.data[i][4])
            seq_len = max(len(seq), 1)
            gcs.append((seq.count("G") + seq.count("C")) / seq_len)
        gcs = np.array(gcs, dtype=np.float32)
        return {
            "n": int(len(indices)),
            "pos_rate": float(y.mean()),
            "mean_len": float(lens.mean()),
            "mean_gc": float(gcs.mean()),
            "len_vals": lens,
            "gc_vals": gcs,
        }

    def _hist_prob(vals, bins=20, vmin=None, vmax=None):
        if vals.size == 0:
            return np.ones(bins, dtype=np.float64) / bins
        if vmin is None:
            vmin = float(vals.min())
        if vmax is None:
            vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        hist, _ = np.histogram(vals, bins=bins, range=(vmin, vmax))
        hist = hist.astype(np.float64) + 1e-8
        return hist / hist.sum()

    def _js_divergence(p, q):
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return float(0.5 * (kl_pm + kl_qm))

    train_stats = _split_stats(train_indices)
    val_stats = _split_stats(val_indices)
    test_stats = _split_stats(test_indices)

    len_min = float(
        min(
            train_stats["len_vals"].min() if train_stats["len_vals"].size else 0.0,
            val_stats["len_vals"].min() if val_stats["len_vals"].size else 0.0,
            test_stats["len_vals"].min() if test_stats["len_vals"].size else 0.0,
        )
    )
    len_max = float(
        max(
            train_stats["len_vals"].max() if train_stats["len_vals"].size else 1.0,
            val_stats["len_vals"].max() if val_stats["len_vals"].size else 1.0,
            test_stats["len_vals"].max() if test_stats["len_vals"].size else 1.0,
        )
    )
    gc_min, gc_max = 0.0, 1.0

    p_len_train = _hist_prob(
        train_stats["len_vals"], bins=20, vmin=len_min, vmax=len_max
    )
    p_gc_train = _hist_prob(train_stats["gc_vals"], bins=20, vmin=gc_min, vmax=gc_max)

    if val_stats["n"] > 0:
        p_len_val = _hist_prob(
            val_stats["len_vals"], bins=20, vmin=len_min, vmax=len_max
        )
        p_gc_val = _hist_prob(val_stats["gc_vals"], bins=20, vmin=gc_min, vmax=gc_max)
        logger.info(
            f"    JS(train||val): len={_js_divergence(p_len_train, p_len_val):.4f}, "
            f"GC={_js_divergence(p_gc_train, p_gc_val):.4f}"
        )

    p_len_test = _hist_prob(test_stats["len_vals"], bins=20, vmin=len_min, vmax=len_max)
    p_gc_test = _hist_prob(test_stats["gc_vals"], bins=20, vmin=gc_min, vmax=gc_max)
    logger.info(
        f"    JS(train||test): len={_js_divergence(p_len_train, p_len_test):.4f}, "
        f"GC={_js_divergence(p_gc_train, p_gc_test):.4f}"
    )

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def _build_kfold_index_groups(config, full_dataset, chromosomes, labels, k_fold):
    """Build k-fold groups as sample-index lists, respecting split_method."""
    split_method = config.get("split_method", "chromosome")

    if split_method in {"genomic_bin", "genomic_bin_representative"}:
        bin_size = int(config.get("bin_size", 2_000_000))
        logger.info(
            f"Building k-fold groups by genomic bins (split_method={split_method}, "
            f"bin_size={bin_size / 1e6:.0f}Mb)"
        )

        bin_to_indices = defaultdict(list)
        for idx in range(len(full_dataset)):
            item = full_dataset.data[idx]
            chrom, start, end = item[0], int(item[1]), int(item[2])
            mid = (start + end) // 2
            bin_idx = mid // bin_size
            bid = f"{chrom}_{bin_idx}"
            bin_to_indices[bid].append(idx)

        unique_bins = sorted(bin_to_indices.keys())
        X = np.arange(len(unique_bins))

        bin_labels = []
        for b in unique_bins:
            y = [labels[i] for i in bin_to_indices[b]]
            bin_labels.append(1 if any(v == 1 for v in y) else 0)
        bin_labels = np.array(bin_labels, dtype=np.int32)

        strat_labels = None
        if split_method == "genomic_bin_representative":
            n_quantiles = int(config.get("rep_split_n_quantiles", 4))
            min_stratum_count = int(config.get("rep_split_min_stratum_count", 3))

            bin_pos_rate = []
            bin_mean_len = []
            bin_mean_gc = []
            for b in unique_bins:
                idxs = bin_to_indices[b]
                y = np.array([labels[i] for i in idxs], dtype=np.float32)
                lengths = np.array(
                    [
                        int(full_dataset.data[i][2]) - int(full_dataset.data[i][1])
                        for i in idxs
                    ],
                    dtype=np.float32,
                )
                gc_vals = []
                for i in idxs:
                    seq = str(full_dataset.data[i][4])
                    seq_len = max(len(seq), 1)
                    gc_vals.append((seq.count("G") + seq.count("C")) / seq_len)
                gc_vals = np.array(gc_vals, dtype=np.float32)

                bin_pos_rate.append(float(y.mean()))
                bin_mean_len.append(float(lengths.mean()))
                bin_mean_gc.append(float(gc_vals.mean()))

            pos_q = _quantile_bin(bin_pos_rate, n_quantiles=n_quantiles)
            len_q = _quantile_bin(bin_mean_len, n_quantiles=n_quantiles)
            gc_q = _quantile_bin(bin_mean_gc, n_quantiles=n_quantiles)

            detailed = [
                f"c{int(c)}_p{int(p)}_l{int(length_bin)}_g{int(g)}"
                for c, p, length_bin, g in zip(bin_labels, pos_q, len_q, gc_q)
            ]
            safe = _build_safe_strata(detailed, bin_labels, min_count=min_stratum_count)
            safe_counts = Counter(safe)

            if len(safe_counts) >= 2 and min(safe_counts.values()) >= k_fold:
                strat_labels = np.array(safe)
                logger.info(
                    f"Using binrepr strata for k-fold bin split "
                    f"(unique={len(safe_counts)})"
                )
            else:
                class_counts = Counter(bin_labels.tolist())
                if len(class_counts) >= 2 and min(class_counts.values()) >= k_fold:
                    strat_labels = bin_labels
                    logger.info("Binrepr strata too sparse")
                else:
                    logger.warning("No valid stratification labels for bin k-fold")
        else:
            class_counts = Counter(bin_labels.tolist())
            if len(class_counts) >= 2 and min(class_counts.values()) >= k_fold:
                strat_labels = bin_labels
            else:
                logger.warning("Class labels too sparse for stratified bin k-fold")

        if strat_labels is not None:
            splitter = StratifiedKFold(
                n_splits=k_fold, shuffle=True, random_state=config.seed
            )
            split_iter = splitter.split(X, strat_labels)
        else:
            splitter = KFold(n_splits=k_fold, shuffle=True, random_state=config.seed)
            split_iter = splitter.split(X)

        groups = []
        for _, test_bin_ids in split_iter:
            fold_indices = []
            for bi in test_bin_ids:
                fold_indices.extend(bin_to_indices[unique_bins[bi]])
            groups.append(fold_indices)

        return groups, "genomic_bin"

    logger.info("Building k-fold groups by chromosome")
    unique_chroms = sorted(set(chromosomes))
    chrom_groups = [[] for _ in range(k_fold)]
    for i, chrom in enumerate(unique_chroms):
        chrom_groups[i % k_fold].append(chrom)

    groups = []
    for group_chroms in chrom_groups:
        idxs = [i for i, c in enumerate(chromosomes) if c in group_chroms]
        groups.append(idxs)

    return groups, "chromosome"


def _build_fold(
    config,
    full_dataset,
    chromosomes,
    labels,
    train_indices,
    val_indices,
    test_indices,
    device,
    fold_name="",
    initial_best_threshold=0.5,
):
    balance_method = config.get("balance_method", "downsample")
    train_ratio = config.get("target_class_ratio", 5.0)
    eval_ratio = config.get("eval_class_ratio", None)
    use_validation = config.trainer.get("use_validation", True)

    resample_info = None

    _train_labels_pre = [labels[i] for i in train_indices]
    _pos_pre = [
        int(idx) for idx, lbl in zip(train_indices, _train_labels_pre) if lbl == 1
    ]
    _neg_pre = [
        int(idx) for idx, lbl in zip(train_indices, _train_labels_pre) if lbl == 0
    ]
    hard_neg_info = {
        "pos_indices": _pos_pre,
        "neg_indices": _neg_pre,
        "full_dataset": full_dataset,
        "target_ratio": train_ratio,
        "seed": config.seed,
    }
    logger.info(
        f"{fold_name} Hard-neg pool: {len(_pos_pre)} pos, {len(_neg_pre)} neg available"
    )

    if balance_method == "resample":
        train_labels_arr = np.array([labels[i] for i in train_indices])
        pos_mask = train_labels_arr == 1
        pos_indices = np.array(train_indices)[pos_mask].tolist()
        neg_indices = np.array(train_indices)[~pos_mask].tolist()
        logger.info(
            f"{fold_name} Train: epoch-wise negative resampling "
            f"(pos={len(pos_indices)}, neg_pool={len(neg_indices)})"
        )
        resample_info = {
            "pos_indices": pos_indices,
            "neg_indices": neg_indices,
            "full_dataset": full_dataset,
            "seed": config.seed,
        }
        train_indices = _balance_split(
            train_indices, labels, 1.0, f"{fold_name} Train (initial)", config.seed
        )
    elif balance_method == "oversample":
        logger.info(
            f"{fold_name} Train: using WeightedRandomSampler (oversample positives)"
        )
    else:
        train_indices = _balance_split(
            train_indices, labels, train_ratio, f"{fold_name} Train", config.seed
        )

    if eval_ratio is not None and use_validation and len(val_indices) > 0:
        val_indices = _balance_split(
            val_indices, labels, eval_ratio, f"{fold_name} Val", config.seed + 1
        )
    if eval_ratio is not None:
        test_indices = _balance_split(
            test_indices, labels, eval_ratio, f"{fold_name} Test", config.seed + 2
        )

    logger.info(f"\n{fold_name} Chromosome distribution by split:")
    for split_name, split_idx in [
        ("Train", train_indices),
        ("Val", val_indices),
        ("Test", test_indices),
    ]:
        Counter([chromosomes[i] for i in split_idx])
        split_labels = Counter([labels[i] for i in split_idx])
        logger.info(
            f"  {split_name}: {len(split_idx)} samples  "
            f"Pos={split_labels.get(1, 0)} Neg={split_labels.get(0, 0)}"
        )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = (
        torch.utils.data.Subset(full_dataset, val_indices)
        if use_validation and len(val_indices) > 0
        else None
    )
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    if balance_method == "oversample":
        train_labels = np.array([labels[i] for i in train_indices])
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        weight_pos = len(train_labels) / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = len(train_labels) / (2.0 * n_neg) if n_neg > 0 else 1.0
        sample_weights = torch.tensor(
            [weight_pos if label == 1 else weight_neg for label in train_labels],
            dtype=torch.double,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_labels), replacement=True
        )
        logger.info(
            f"  WeightedRandomSampler: n_pos={int(n_pos)}, n_neg={int(n_neg)}, "
            f"weight_pos={weight_pos:.3f}, weight_neg={weight_neg:.3f}"
        )
        train_dl = instantiate(
            config.dataloader,
            dataset=train_dataset,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=set_worker_seed,
        )
    else:
        train_dl = instantiate(
            config.dataloader,
            dataset=train_dataset,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            worker_init_fn=set_worker_seed,
        )
    val_dl = None
    if val_dataset is not None:
        val_dl = instantiate(
            config.dataloader,
            dataset=val_dataset,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            worker_init_fn=set_worker_seed,
        )
    test_dl = instantiate(
        config.dataloader,
        dataset=test_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        worker_init_fn=set_worker_seed,
    )
    dataloaders = {"train": train_dl, "val": val_dl, "test": test_dl}

    model = instantiate(config.model).to(device)
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    for metric_list in metrics.values():
        for m in metric_list:
            m.to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    scheduler_target = config.lr_scheduler.get("_target_", "")
    if "OneCycleLR" in scheduler_target:
        steps_per_epoch = len(train_dl)
        sched_cfg = OmegaConf.to_container(config.lr_scheduler, resolve=True)
        for bad_key in [
            "mode",
            "factor",
            "patience",
            "min_lr",
            "T_0",
            "T_mult",
            "eta_min",
        ]:
            sched_cfg.pop(bad_key, None)
        lr_scheduler = instantiate(
            sched_cfg,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            epochs=config.trainer.n_epochs,
        )
    else:
        sched_cfg = OmegaConf.to_container(config.lr_scheduler, resolve=True)
        sched_cfg = {k: v for k, v in sched_cfg.items() if v is not None}
        lr_scheduler = instantiate(sched_cfg, optimizer=optimizer)

    project_config = OmegaConf.to_container(config, resolve=True)
    base_run_name = config.writer.get("run_name") or "experiment"
    fold_run_name = (
        f"{base_run_name}_{fold_name.replace(' ', '_')}" if fold_name else base_run_name
    )
    writer = instantiate(
        config.writer,
        logger=logger,
        project_config=project_config,
        run_name=fold_run_name,
        _recursive_=False,
    )

    data_indices = {"train": train_indices, "val": val_indices, "test": test_indices}

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        writer=writer,
        start_epoch=1,
        best_val_metric=0,
        best_threshold=float(initial_best_threshold),
        data_indices=data_indices,
        resample_info=resample_info,
        hard_neg_info=hard_neg_info,
    )
    trainer.train()

    result = {
        "best_val_f1": trainer.best_val_metric,
        "best_threshold": trainer.best_threshold,
    }

    val_preds = None
    if trainer.val_dataloader is not None:
        val_preds = trainer._collect_predictions(
            trainer.val_dataloader, "ValFinal", epoch=0
        )
    test_preds = trainer._collect_predictions(
        trainer.test_dataloader, "TestFinal", epoch=0
    )

    if val_preds is not None:
        if "seq_probs" in val_preds and "seq_labels" in val_preds:
            result["val_probs"] = val_preds["seq_probs"].cpu().numpy()
            result["val_labels"] = val_preds["seq_labels"].cpu().numpy()
        else:
            result["val_probs"] = val_preds["nuc_probs"].cpu().numpy()
            result["val_labels"] = val_preds["nuc_labels"].cpu().numpy()

    if "seq_probs" in test_preds and "seq_labels" in test_preds:
        result["test_probs"] = test_preds["seq_probs"].cpu().numpy()
        result["test_labels"] = test_preds["seq_labels"].cpu().numpy()
    else:
        result["test_probs"] = test_preds["nuc_probs"].cpu().numpy()
        result["test_labels"] = test_preds["nuc_labels"].cpu().numpy()

    if trainer.test_metrics is not None:
        result["test_f1"] = trainer.test_metrics.get("f1", float("nan"))
        result["test_auc"] = trainer.test_metrics.get("auc", float("nan"))
        result["test_ap"] = trainer.test_metrics.get("avg_precision", float("nan"))
        result["test_nuc_auc"] = trainer.test_metrics.get("nuc_auc", float("nan"))
    return result


@hydra.main(
    version_base=None, config_path="src/configs", config_name="grummit_triplexnet"
)
def main(config):
    set_random_seed(config.seed)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    logger.info(f"Using device: {device}")

    logger.info("Loading dataset:")
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(config.datasets.train, resolve=True)
    )
    dataset_cfg = _resolve_dataset_paths(dataset_cfg)
    full_dataset = instantiate(dataset_cfg)

    dataset_feature_dim = getattr(full_dataset, "feature_dim", None)
    if dataset_feature_dim is not None:
        model_feature_dim = config.model.get("n_omics_features", None)
        if model_feature_dim != dataset_feature_dim:
            with open_dict(config.model):
                config.model.n_omics_features = int(dataset_feature_dim)

    dataset_kmer_feature_dim = getattr(full_dataset, "kmer_feature_dim", 0)
    model_kmer_feature_dim = config.model.get("n_kmer_features", 0)
    if model_kmer_feature_dim != dataset_kmer_feature_dim:
        with open_dict(config.model):
            config.model.n_kmer_features = int(dataset_kmer_feature_dim)

    chromosomes = []
    labels = []
    for idx in range(len(full_dataset)):
        item = full_dataset.data[idx]
        chrom = item[0]
        label = item[5]
        chromosomes.append(chrom)
        labels.append(label)

    logger.info("Dataset distribution by chromosome:")
    chrom_counts = Counter(chromosomes)
    for chrom, count in sorted(chrom_counts.items()):
        logger.info(f"  {chrom}: {count} samples")

    logger.info("Dataset distribution by label:")
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        logger.info(f"  Label {label}: {count} samples")

    logger.info("\nLabel distribution by chromosome:")
    for chrom in sorted(set(chromosomes)):
        chrom_labels = [labels[i] for i, c in enumerate(chromosomes) if c == chrom]
        pos = sum(chrom_labels)
        neg = len(chrom_labels) - pos
        pct_pos = 100 * pos / len(chrom_labels) if chrom_labels else 0
        logger.info(
            f"  {chrom}: Pos={pos}, Neg={neg}, Total={len(chrom_labels)}, %Pos={pct_pos:.1f}%"
        )

    k_fold = config.get("k_fold", 0)
    if k_fold and k_fold > 1:
        kfold_final_cfg = config.get("k_fold_final_stage", {})
        if isinstance(kfold_final_cfg, bool):
            final_stage_enabled = bool(kfold_final_cfg)
            blind_group_index = 0
            final_stage_use_global_threshold = True
            final_stage_disable_validation = True
        else:
            final_stage_enabled = bool(kfold_final_cfg.get("enabled", False))
            blind_group_index = int(kfold_final_cfg.get("blind_group_index", 0))
            final_stage_use_global_threshold = bool(
                kfold_final_cfg.get("use_global_threshold", True)
            )
            final_stage_disable_validation = bool(
                kfold_final_cfg.get("disable_validation", True)
            )

        logger.info(f"K-fold CV: {k_fold} folds")

        index_groups, grouping_mode = _build_kfold_index_groups(
            config=config,
            full_dataset=full_dataset,
            chromosomes=chromosomes,
            labels=labels,
            k_fold=int(k_fold),
        )

        blind_test_indices = []
        cv_groups = index_groups
        if final_stage_enabled:
            blind_group_index = blind_group_index % len(index_groups)
            blind_test_indices = index_groups[blind_group_index]
            cv_groups = [
                g for i, g in enumerate(index_groups) if i != blind_group_index
            ]

            blind_test_chroms = sorted(set(chromosomes[i] for i in blind_test_indices))

            logger.info("\nFinal stage enabled:")
            logger.info(f"  Blind test group index: {blind_group_index}")
            logger.info(f"  Blind test chromosomes: {blind_test_chroms}")
            logger.info(
                f"  CV groups used for threshold/model selection: {len(cv_groups)}"
            )

        n_cv_folds = len(cv_groups)
        fold_results = []
        for fold_idx in range(n_cv_folds):
            logger.info(f"Fold number {fold_idx + 1}/{n_cv_folds}")
            test_idx = np.array(cv_groups[fold_idx], dtype=int)
            val_idx = np.array(cv_groups[(fold_idx + 1) % n_cv_folds], dtype=int)
            train_idx = []
            for j in range(n_cv_folds):
                if j != fold_idx and j != (fold_idx + 1) % n_cv_folds:
                    train_idx.extend(cv_groups[j])
            train_idx = np.array(train_idx, dtype=int)

            logger.info(
                f"  Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
            )

            set_random_seed(config.seed + fold_idx)

            result = _build_fold(
                config,
                full_dataset,
                chromosomes,
                labels,
                train_idx,
                val_idx,
                test_idx,
                device,
                fold_name=f"Fold {fold_idx + 1}",
            )
            fold_results.append(result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()

        logger.info(f"K-fold summary ({n_cv_folds} folds)")
        val_f1s = [r["best_val_f1"] for r in fold_results]
        test_f1s = [r.get("test_f1", float("nan")) for r in fold_results]
        test_aucs = [r.get("test_auc", float("nan")) for r in fold_results]
        test_nuc_aucs = [r.get("test_nuc_auc", float("nan")) for r in fold_results]
        for i, r in enumerate(fold_results):
            logger.info(
                f"  Fold {i + 1}: "
                f"val_F1={r['best_val_f1']:.4f}  "
                f"test_F1={r.get('test_f1', float('nan')):.4f}  "
                f"test_seqAUC={r.get('test_auc', float('nan')):.4f}  "
                f"test_nucAUC={r.get('test_nuc_auc', float('nan')):.4f}  "
                f"thr={r['best_threshold']:.3f}"
            )
        import math

        valid_val = [v for v in val_f1s if not math.isnan(v)]
        valid_test = [v for v in test_f1s if not math.isnan(v)]
        valid_auc = [v for v in test_aucs if not math.isnan(v)]
        valid_nuc_auc = [v for v in test_nuc_aucs if not math.isnan(v)]
        if valid_val:
            logger.info(
                f"  Mean val  F1:       {np.mean(valid_val):.4f} ± {np.std(valid_val):.4f}"
            )
        if valid_test:
            logger.info(
                f"  Mean test F1 (seq): {np.mean(valid_test):.4f} ± {np.std(valid_test):.4f}"
            )
        if valid_auc:
            logger.info(
                f"  Mean test AUC(seq): {np.mean(valid_auc):.4f} ± {np.std(valid_auc):.4f}"
            )
        if valid_nuc_auc:
            logger.info(
                f"  Mean test AUC(nuc): {np.mean(valid_nuc_auc):.4f} ± {np.std(valid_nuc_auc):.4f}"
            )

        use_global_kfold_threshold = config.trainer.get(
            "use_global_kfold_threshold", True
        )
        global_thr = None
        if use_global_kfold_threshold:
            pooled_val_probs = []
            pooled_val_labels = []
            for r in fold_results:
                if "val_probs" in r and "val_labels" in r:
                    pooled_val_probs.append(r["val_probs"])
                    pooled_val_labels.append(r["val_labels"])

            if pooled_val_probs and pooled_val_labels:
                pooled_val_probs = np.concatenate(pooled_val_probs)
                pooled_val_labels = np.concatenate(pooled_val_labels)
                global_thr, pooled_val_best_f1 = _find_best_threshold_np(
                    pooled_val_probs, pooled_val_labels
                )
                logger.info(
                    f"\nGlobal threshold from pooled OOF val: {global_thr:.3f} "
                    f"(pooled val F1={pooled_val_best_f1:.4f})"
                )

                global_test_f1s = []
                global_test_precs = []
                global_test_recs = []
                global_test_accs = []

                logger.info("Per-fold test metrics with global threshold:")
                for i, r in enumerate(fold_results):
                    if "test_probs" not in r or "test_labels" not in r:
                        continue
                    metrics_global = _metrics_at_threshold_np(
                        r["test_probs"], r["test_labels"], global_thr
                    )
                    global_test_f1s.append(metrics_global["f1"])
                    global_test_precs.append(metrics_global["precision"])
                    global_test_recs.append(metrics_global["recall"])
                    global_test_accs.append(metrics_global["accuracy"])

                    logger.info(
                        f"  Fold {i + 1}: "
                        f"test_F1={metrics_global['f1']:.4f} "
                        f"P={metrics_global['precision']:.4f} "
                        f"R={metrics_global['recall']:.4f} "
                        f"Acc={metrics_global['accuracy']:.4f}"
                    )

                if global_test_f1s:
                    logger.info(
                        f"  Mean test F1 (global thr): {np.mean(global_test_f1s):.4f} ± {np.std(global_test_f1s):.4f}"
                    )
                    logger.info(
                        f"  Mean test P  (global thr): {np.mean(global_test_precs):.4f} ± {np.std(global_test_precs):.4f}"
                    )
                    logger.info(
                        f"  Mean test R  (global thr): {np.mean(global_test_recs):.4f} ± {np.std(global_test_recs):.4f}"
                    )
                    logger.info(
                        f"  Mean test Acc(global thr): {np.mean(global_test_accs):.4f} ± {np.std(global_test_accs):.4f}"
                    )

        if final_stage_enabled:
            logger.info("Last stage: blind test")
            train_indices_final = [idx for group in cv_groups for idx in group]
            test_indices_final = list(blind_test_indices)
            val_indices_final = np.array([], dtype=int)

            final_train_chroms = sorted(
                set(chromosomes[i] for i in train_indices_final)
            )
            final_test_chroms = sorted(set(chromosomes[i] for i in test_indices_final))

            logger.info(f"Final non-blind train chromosomes: {final_train_chroms}")
            logger.info(f"Final blind test chromosomes: {final_test_chroms}")
            logger.info(
                f"Final split sizes: train={len(train_indices_final)}, "
                f"test={len(test_indices_final)}"
            )

            final_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
            if final_stage_disable_validation:
                final_cfg.trainer.use_validation = False
                final_cfg.trainer.threshold_tuning_during_training = False

            init_final_threshold = 0.5
            if final_stage_use_global_threshold and global_thr is not None:
                init_final_threshold = float(global_thr)
                logger.info(
                    f"Using pooled OOF global threshold for final blind test: {init_final_threshold:.3f}"
                )
            elif final_stage_use_global_threshold:
                logger.warning(
                    "Requested global threshold for final stage, but it was unavailable. "
                    "Falling back to threshold=0.5"
                )

            final_result = _build_fold(
                final_cfg,
                full_dataset,
                chromosomes,
                labels,
                np.array(train_indices_final),
                val_indices_final,
                np.array(test_indices_final),
                device,
                fold_name="FinalBlind",
                initial_best_threshold=init_final_threshold,
            )

            logger.info("\nFinal blind-test metrics:")
            logger.info(
                f"  F1={final_result.get('test_f1', float('nan')):.4f}  "
                f"AUC(seq)={final_result.get('test_auc', float('nan')):.4f}  "
                f"AUC(nuc)={final_result.get('test_nuc_auc', float('nan')):.4f}  "
                f"threshold={final_result.get('best_threshold', init_final_threshold):.3f}"
            )
        return

    saved_indices = None
    if config.trainer.get("resume_from") is not None:
        checkpoint_path = config.trainer.resume_from
        temp_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    saved_indices = temp_checkpoint.get("data_indices")

    if saved_indices is not None:
        train_indices = saved_indices["train"]
        val_indices = saved_indices["val"]
        test_indices = saved_indices["test"]
        logger.info("Using saved data split indices from checkpoint")
    else:
        split_method = config.get("split_method", "chromosome")
        use_validation = config.trainer.get("use_validation", True)

        if split_method == "genomic_bin_representative":
            (
                train_indices,
                val_indices,
                test_indices,
            ) = _genomic_bin_representative_split(
                full_dataset=full_dataset,
                labels=labels,
                config=config,
                use_validation=use_validation,
            )

        elif split_method == "genomic_bin":
            bin_size = config.get("bin_size", 2_000_000)
            test_frac = config.get("test_frac", 0.15)
            val_frac = config.get("val_frac", 0.15)

            logger.info(
                f"Using genomic bin split: bin_size={bin_size / 1e6:.0f}Mb, "
                f"test={test_frac:.0%}, val={val_frac:.0%}"
            )

            bin_ids = []
            coords = []
            for idx in range(len(full_dataset)):
                item = full_dataset.data[idx]
                chrom, start, end = item[0], item[1], item[2]
                mid = (start + end) // 2
                bin_idx = mid // bin_size
                bin_ids.append(f"{chrom}_{bin_idx}")
                coords.append((chrom, start, end))

            from collections import defaultdict

            bin_to_indices = defaultdict(list)
            bin_to_labels = defaultdict(list)
            for idx, bid in enumerate(bin_ids):
                bin_to_indices[bid].append(idx)
                bin_to_labels[bid].append(labels[idx])

            unique_bins = sorted(bin_to_indices.keys())
            bin_labels = [
                1 if any(label == 1 for label in bin_to_labels[b]) else 0
                for b in unique_bins
            ]

            logger.info(
                f"  Total bins: {len(unique_bins)} "
                f"(pos_bins={sum(bin_labels)}, neg_bins={len(bin_labels) - sum(bin_labels)})"
            )

            bin_indices = np.arange(len(unique_bins))
            if use_validation and val_frac > 0:
                train_bins, temp_bins = train_test_split(
                    bin_indices,
                    test_size=test_frac + val_frac,
                    random_state=config.seed,
                    stratify=bin_labels,
                )
                temp_labels = [bin_labels[i] for i in temp_bins]
                relative_val = val_frac / (test_frac + val_frac)
                val_bins, test_bins = train_test_split(
                    temp_bins,
                    test_size=1 - relative_val,
                    random_state=config.seed,
                    stratify=temp_labels,
                )
            else:
                train_bins, test_bins = train_test_split(
                    bin_indices,
                    test_size=test_frac,
                    random_state=config.seed,
                    stratify=bin_labels,
                )
                val_bins = np.array([], dtype=int)

            train_set = set(train_bins)
            val_set = set(val_bins)

            train_indices, val_indices, test_indices = [], [], []
            for bi, bname in enumerate(unique_bins):
                idxs = bin_to_indices[bname]
                if bi in train_set:
                    train_indices.extend(idxs)
                elif bi in val_set:
                    val_indices.extend(idxs)
                else:
                    test_indices.extend(idxs)

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)

            for split_name, split_bins in [
                ("Train", train_bins),
                ("Val", val_bins),
                ("Test", test_bins),
            ]:
                n_pos_bins = sum(1 for i in split_bins if bin_labels[i] == 1)
                n_neg_bins = len(split_bins) - n_pos_bins
                logger.info(
                    f"  {split_name}: {len(split_bins)} bins "
                    f"(pos={n_pos_bins}, neg={n_neg_bins})"
                )

            split_pairs = [(test_indices, "Test", train_indices, "Train")]
            if use_validation and len(val_indices) > 0:
                split_pairs.insert(0, (val_indices, "Val", train_indices, "Train"))
            for split_a, name_a, split_b, name_b in split_pairs:
                min_dist = float("inf")
                for ia in split_a[:100]:
                    ca, sa, ea = coords[ia]
                    for ib in split_b:
                        cb, sb, eb = coords[ib]
                        if ca == cb:
                            d = max(0, max(sa, sb) - min(ea, eb))
                            min_dist = min(min_dist, d)
                logger.info(
                    f"  Min distance {name_a}↔{name_b}: {min_dist / 1000:.0f}kb "
                    f"(bin_size={bin_size / 1e6:.0f}Mb)"
                )

        elif split_method == "chromosome" or config.get("chromosome_split", False):
            logger.info("Using chromosome-based split")
            unique_chroms = sorted(set(chromosomes))
            logger.info(f"Unique chromosomes: {unique_chroms}")

            test_chroms = config.get("test_chromosomes", ["chr1", "chr2"])
            val_chroms = config.get("val_chromosomes", ["chr3", "chr4"])

            train_indices, val_indices, test_indices = [], [], []
            for idx, chrom in enumerate(chromosomes):
                if chrom in test_chroms:
                    test_indices.append(idx)
                elif use_validation and chrom in val_chroms:
                    val_indices.append(idx)
                else:
                    train_indices.append(idx)

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)
            logger.info(f"Chromosome split - Test: {test_chroms}, Val: {val_chroms}")
        else:
            logger.warning("Using random stratified split")
            test_frac = config.get("test_frac", 0.15)
            val_frac = config.get("val_frac", 0.15)
            indices = np.arange(len(full_dataset))
            if use_validation and val_frac > 0:
                train_indices, temp_indices = train_test_split(
                    indices,
                    test_size=test_frac + val_frac,
                    random_state=config.seed,
                    stratify=labels,
                )
                temp_labels = [labels[i] for i in temp_indices]
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=test_frac / (test_frac + val_frac),
                    random_state=config.seed,
                    stratify=temp_labels,
                )
            else:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_frac,
                    random_state=config.seed,
                    stratify=labels,
                )
                val_indices = np.array([], dtype=int)

    _build_fold(
        config,
        full_dataset,
        chromosomes,
        labels,
        np.array(train_indices)
        if not isinstance(train_indices, np.ndarray)
        else train_indices,
        np.array(val_indices)
        if not isinstance(val_indices, np.ndarray)
        else val_indices,
        np.array(test_indices)
        if not isinstance(test_indices, np.ndarray)
        else test_indices,
        device,
        fold_name="Single",
    )


if __name__ == "__main__":
    main()

"""Post-processing utilities for nucleotide-level predictions."""

from typing import List, Tuple

import numpy as np
import torch


def predictions_to_intervals(
    predictions: torch.Tensor,
    threshold: float = 0.25,
    min_length: int = 10,
    max_gap: int = 20,
) -> List[Tuple[int, int]]:
    """
    Convert nucleotide-level predictions to intervals.

    Args:
        predictions: [seq_len] probability predictions
        threshold: classification threshold
        min_length: minimum interval length in bp (default 10)
        max_gap: maximum gap between intervals to merge (default 20)

    Returns:
        List of (start, end) tuples representing predicted intervals
    """
    binary_pred = (predictions >= threshold).cpu().numpy().astype(int)

    intervals = []
    start = None

    for i, val in enumerate(binary_pred):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            intervals.append((start, i))
            start = None

    if start is not None:
        intervals.append((start, len(binary_pred)))

    intervals = [(s, e) for s, e in intervals if (e - s) >= min_length]

    if len(intervals) > 0:
        merged = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start - current_end <= max_gap:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        intervals = merged

    return intervals


def batch_predictions_to_intervals(
    predictions: torch.Tensor,
    threshold: float = 0.25,
    min_length: int = 10,
    max_gap: int = 20,
) -> List[List[Tuple[int, int]]]:
    """
    Convert batch of nucleotide-level predictions to intervals.

    Args:
        predictions: [batch, seq_len] probability predictions
        threshold: classification threshold
        min_length: minimum interval length in bp
        max_gap: maximum gap between intervals to merge

    Returns:
        List of interval lists, one per batch item
    """
    batch_intervals = []

    for pred in predictions:
        intervals = predictions_to_intervals(pred, threshold, min_length, max_gap)
        batch_intervals.append(intervals)

    return batch_intervals


def optimize_threshold(
    predictions: torch.Tensor, labels: torch.Tensor, thresholds: np.ndarray = None
) -> Tuple[float, float]:
    """
    Find threshold by maximizing F1-score.

    Args:
        predictions: [n_samples, seq_len] predictions
        labels: [n_samples, seq_len] ground truth
        thresholds: array of thresholds to test (default: 0.1 to 0.9 step 0.05)

    Returns:
        best_threshold: optimal threshold value
        best_f1: F1-score at optimal threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    predictions_flat = predictions.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        pred_binary = (predictions_flat >= threshold).astype(int)

        tp = np.sum((pred_binary == 1) & (labels_flat == 1))
        fp = np.sum((pred_binary == 1) & (labels_flat == 0))
        fn = np.sum((pred_binary == 0) & (labels_flat == 1))

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def intervals_to_bed(
    intervals: List[Tuple[int, int]],
    chrom: str,
    region_start: int,
    score: float = 1000,
    strand: str = "+",
) -> List[str]:
    """
    Convert intervals to BED format lines.

    Args:
        intervals: list of (start, end) tuples (relative to region)
        chrom: chromosome name
        region_start: genomic start position of the region
        score: BED score (default 1000)
        strand: strand (default '+')

    Returns:
        List of BED format strings
    """
    bed_lines = []

    for start, end in intervals:
        genomic_start = region_start + start
        genomic_end = region_start + end

        bed_line = f"{chrom}\t{genomic_start}\t{genomic_end}\tTPX\t{score}\t{strand}"
        bed_lines.append(bed_line)

    return bed_lines


def predictions_to_bedgraph(
    predictions: torch.Tensor,
    chrom: str,
    region_start: int,
    region_end: int = None,
    bin_size: int = 1,
) -> List[str]:
    """
    Convert nucleotide probabilities to bedGraph lines.

    Args:
        predictions: [seq_len] probability predictions
        chrom: chromosome name
        region_start: genomic start position of the region
        region_end: genomic end position of the region (optional)
        bin_size: bin size in bp for averaging probabilities (default 1)

    Returns:
        List of bedGraph lines: chrom start end value
    """
    probs = predictions.detach().cpu().numpy().astype(np.float32)
    valid_len = probs.shape[0]

    if region_end is not None:
        valid_len = min(valid_len, max(int(region_end) - int(region_start), 0))

    if valid_len <= 0:
        return []

    probs = probs[:valid_len]
    bin_size = max(1, int(bin_size))

    lines = []
    for i in range(0, valid_len, bin_size):
        j = min(i + bin_size, valid_len)
        value = float(probs[i:j].mean())
        lines.append(f"{chrom}\t{region_start + i}\t{region_start + j}\t{value:.6f}")

    return lines


def predictions_to_tsv_rows(
    predictions: torch.Tensor,
    chrom: str,
    region_start: int,
    region_end: int = None,
    bin_size: int = 1,
) -> List[str]:
    """
    Convert nucleotide probabilities to TSV rows.

    Args:
        predictions: [seq_len] probability predictions
        chrom: chromosome name
        region_start: genomic start position of the region
        region_end: genomic end position of the region (optional)
        bin_size: bin size in bp for averaging probabilities (default 1)

    Returns:
        List of TSV rows: chrom\tstart\tend\tprobability
    """
    probs = predictions.detach().cpu().numpy().astype(np.float32)
    valid_len = probs.shape[0]

    if region_end is not None:
        valid_len = min(valid_len, max(int(region_end) - int(region_start), 0))

    if valid_len <= 0:
        return []

    probs = probs[:valid_len]
    bin_size = max(1, int(bin_size))

    rows = []
    for i in range(0, valid_len, bin_size):
        j = min(i + bin_size, valid_len)
        value = float(probs[i:j].mean())
        rows.append(f"{chrom}\t{region_start + i}\t{region_start + j}\t{value:.6f}")

    return rows

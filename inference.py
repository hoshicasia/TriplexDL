#!/usr/bin/env python3
"""
Inference script for nucleotide-level predictions with post-processing.
Outputs BED file with predicted triplex intervals.
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_random_seed
from src.utils.postprocess import (
    batch_predictions_to_intervals,
    intervals_to_bed,
    optimize_threshold,
    predictions_to_bedgraph,
    predictions_to_tsv_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="grummit_triplexnet"
)
def main(config):
    """
    Run inference and generate BED file with predictions.

    Args:
        config: Hydra configuration
    """
    set_random_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = instantiate(config.model).to(device)

    checkpoint_path = Path(config.get("checkpoint_path", "saved/model_best.pth"))
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Using randomly initialized model")

    model.eval()

    logger.info("Loading dataset...")
    dataset = instantiate(config.datasets.train)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader.num_workers,
    )

    all_predictions = []
    all_labels = []
    all_metadata = []

    logger.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            sequence = batch["sequence"].to(device)
            omics = batch["omics_features"].to(device)
            labels = batch["label"]

            outputs = model(sequence, omics)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

            all_predictions.append(probs.cpu())
            all_labels.append(labels)

            for idx in range(len(sequence)):
                data_idx = len(all_metadata)
                if data_idx < len(dataset.data):
                    chrom, start, end, strand, seq, label = dataset.data[data_idx]
                    all_metadata.append((chrom, start, end))

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logger.info(f"Collected {len(all_predictions)} predictions")

    logger.info("Optimizing threshold...")
    best_threshold, best_f1 = optimize_threshold(all_predictions, all_labels)
    logger.info(f"Best threshold: {best_threshold:.3f}, F1: {best_f1:.4f}")

    threshold = config.get("prediction_threshold", best_threshold)
    logger.info(f"Using threshold: {threshold:.3f}")

    logger.info("Converting predictions to intervals...")
    min_length = config.get("min_interval_length", 10)
    max_gap = config.get("max_merge_gap", 20)

    all_intervals = batch_predictions_to_intervals(
        all_predictions, threshold=threshold, min_length=min_length, max_gap=max_gap
    )

    output_path = Path(config.get("output_bed", "predictions.bed"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing predictions to: {output_path}")

    total_intervals = 0
    with open(output_path, "w") as f:
        f.write("# Triplex nucleotide-level predictions\n")
        f.write(f"# Threshold: {threshold:.3f}\n")
        f.write(f"# Min length: {min_length} bp\n")
        f.write(f"# Max gap: {max_gap} bp\n")

        for idx, intervals in enumerate(all_intervals):
            if idx < len(all_metadata) and len(intervals) > 0:
                chrom, region_start, region_end = all_metadata[idx]

                bed_lines = intervals_to_bed(
                    intervals, chrom=chrom, region_start=region_start
                )

                for line in bed_lines:
                    f.write(line + "\n")
                    total_intervals += 1

    output_nucleotide_bedgraph = config.get("output_nucleotide_bedgraph", None)
    output_nucleotide_tsv = config.get("output_nucleotide_tsv", None)
    nucleotide_track_bin_size = int(config.get("nucleotide_track_bin_size", 1))

    if output_nucleotide_bedgraph:
        bedgraph_path = Path(output_nucleotide_bedgraph)
        bedgraph_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing nucleotide probability bedGraph: {bedgraph_path}")

        n_lines = 0
        with open(bedgraph_path, "w") as f_bg:
            f_bg.write("track type=bedGraph name=Triplex_probabilities\n")
            for idx, probs in enumerate(all_predictions):
                if idx >= len(all_metadata):
                    break
                chrom, region_start, region_end = all_metadata[idx]
                lines = predictions_to_bedgraph(
                    predictions=probs,
                    chrom=chrom,
                    region_start=region_start,
                    region_end=region_end,
                    bin_size=nucleotide_track_bin_size,
                )
                for line in lines:
                    f_bg.write(line + "\n")
                n_lines += len(lines)
        logger.info(f"Written {n_lines} bedGraph records")

    if output_nucleotide_tsv:
        tsv_path = Path(output_nucleotide_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing nucleotide probability TSV: {tsv_path}")

        n_rows = 0
        with open(tsv_path, "w") as f_tsv:
            f_tsv.write("chrom\tstart\tend\tprobability\n")
            for idx, probs in enumerate(all_predictions):
                if idx >= len(all_metadata):
                    break
                chrom, region_start, region_end = all_metadata[idx]
                rows = predictions_to_tsv_rows(
                    predictions=probs,
                    chrom=chrom,
                    region_start=region_start,
                    region_end=region_end,
                    bin_size=nucleotide_track_bin_size,
                )
                for row in rows:
                    f_tsv.write(row + "\n")
                n_rows += len(rows)
        logger.info(f"Written {n_rows} TSV rows")

    logger.info(f"Written {total_intervals} intervals to {output_path}")

    logger.info("\nPrediction statistics:")
    logger.info(f"  Total samples: {len(all_predictions)}")
    logger.info(f"  Samples with predictions: {sum(len(i) > 0 for i in all_intervals)}")
    logger.info(f"  Total intervals: {total_intervals}")
    logger.info(
        f"  Avg intervals per positive sample: {total_intervals / max(1, sum(len(i) > 0 for i in all_intervals)):.2f}"
    )


if __name__ == "__main__":
    main()

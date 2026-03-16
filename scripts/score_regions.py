import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.triplex import TriplexDataset
from src.model.triplexnet import TriplexNet

logger = logging.getLogger("score_regions")


class TriplexInferenceDataset(TriplexDataset):
    def __init__(
        self,
        fasta_path: str,
        bed_dir: str,
        max_seq_len: int = 1000,
        positional_omics: bool = False,
        nucleotide_level: bool = True,
        omics_feature_mode: str = "coverage_score",
        score_transform: str = "log1p",
        selected_bed_features: List[str] | None = None,
        kmer_max_k: int = 0,
        kmer_window_count: int = 0,
    ):
        self.name = "inference"
        self.max_seq_len = max_seq_len
        self.positional_omics = positional_omics
        self.nucleotide_level = nucleotide_level
        self.rc_augment = False
        self.nuc_mask_prob = 0.0
        self.coord_shift_max = 0
        self.kmer_max_k = kmer_max_k
        self.kmer_window_count = kmer_window_count

        if kmer_max_k > 0 and kmer_window_count > 0:
            self._kmer_feature_dim = sum(4**k for k in range(1, kmer_max_k + 1))
            self._nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        else:
            self._kmer_feature_dim = 0

        self.selected_bed_features = (
            set(selected_bed_features) if selected_bed_features else None
        )
        self.omics_feature_mode = omics_feature_mode
        self.score_transform = score_transform

        seq_data = self._parse_fasta_with_coords(Path(fasta_path))

        bed_dir = Path(bed_dir)
        self.bed_data, self.bed_names, self.bed_score_scales = self._load_bed_files(
            bed_dir
        )
        self.base_feature_dim = len(self.bed_names)
        self.kmer_feature_dim = self._kmer_feature_dim
        self.feature_dim = self._infer_feature_dim()
        logger.info(f"Loaded {len(self.bed_names)} omics features")

        self.data = []
        for chrom, start, end, strand, seq in seq_data:
            self.data.append((chrom, start, end, strand, seq, 0))

        logger.info(f"Total inference samples: {len(self.data)}")

    def __getitem__(self, idx):
        chrom, start, end, strand, seq, label = self.data[idx]

        sequence = self._one_hot_encode(seq)
        omics_features = self._extract_omics_features(chrom, start, end)
        kmer_features = None
        if self._kmer_feature_dim > 0:
            kmer_features = self._compute_local_kmer_features(seq)

        actual_seq_len = min(len(seq), self.max_seq_len)

        if self.nucleotide_level:
            labels = torch.zeros(self.max_seq_len, dtype=torch.float32)
            mask = torch.zeros(self.max_seq_len, dtype=torch.float32)
            mask[:actual_seq_len] = 1.0
        else:
            labels = torch.LongTensor([label])[0]
            mask = None

        result = {
            "sequence": torch.FloatTensor(sequence),
            "omics_features": torch.FloatTensor(omics_features),
            "label": labels,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
        }

        if kmer_features is not None:
            result["kmer_features"] = torch.FloatTensor(kmer_features)

        if mask is not None:
            result["mask"] = mask

        return result


def inference_collate(items: List[dict]):
    batch = {
        "sequence": torch.stack([i["sequence"] for i in items]),
        "omics_features": torch.stack([i["omics_features"] for i in items]),
    }

    labels = [i["label"] for i in items]
    if isinstance(labels[0], torch.Tensor) and labels[0].dim() > 0:
        batch["label"] = torch.stack(labels)
    else:
        batch["label"] = torch.tensor(labels)

    if "kmer_features" in items[0]:
        batch["kmer_features"] = torch.stack([i["kmer_features"] for i in items])

    if "mask" in items[0]:
        batch["mask"] = torch.stack([i["mask"] for i in items])

    batch["coords"] = [(i["chrom"], i["start"], i["end"], i["strand"]) for i in items]
    return batch


def _to_plain(obj):
    if obj is None:
        return {}
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _clean_model_cfg(model_cfg: dict) -> dict:
    return {k: v for k, v in model_cfg.items() if not k.startswith("_")}


def _aggregate_sequence_probs(
    nuc_probs: torch.Tensor, mask: torch.Tensor | None, topk_ratio: float = 0.2
) -> torch.Tensor:
    if mask is not None and nuc_probs.shape == mask.shape:
        valid_mask = mask.bool()
        seq_lengths = valid_mask.sum(dim=1).long().clamp(min=1)
        k_vals = (seq_lengths.float() * topk_ratio).long().clamp(min=1)
        seq_prob_list = []
        for i in range(nuc_probs.size(0)):
            probs = nuc_probs[i][valid_mask[i]]
            k = int(k_vals[i].item())
            topk_probs = probs.topk(k).values if probs.numel() else probs
            seq_prob_list.append(
                topk_probs.mean()
                if topk_probs.numel()
                else torch.tensor(0.0, device=nuc_probs.device)
            )
        return torch.stack(seq_prob_list)

    k = max(1, int(nuc_probs.size(1) * topk_ratio))
    return nuc_probs.topk(k, dim=1).values.mean(dim=1)


def parse_args():
    p = argparse.ArgumentParser(description="Score regions with a trained TriplexNet")
    p.add_argument(
        "--fasta", required=True, help="FASTA of regions to score (coords in header)"
    )
    p.add_argument(
        "--bed-dir",
        required=True,
        help="Directory with BED features (same as training)",
    )
    p.add_argument("--checkpoint", required=True, help="Path to model_best.pth")
    p.add_argument("--out", default="predictions.tsv", help="Output TSV path")
    p.add_argument(
        "--nuc-out",
        default=None,
        help="Optional bedGraph path for per-base probabilities",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold; defaults to checkpoint best_threshold",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max_seq_len (defaults to training config)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model_cfg = _clean_model_cfg(_to_plain(getattr(cfg, "model", {})))
    loss_cfg = _to_plain(getattr(cfg, "loss_function", {}))
    ds_cfg = _to_plain(getattr(getattr(cfg, "datasets", {}), "train", {}))

    max_seq_len = args.max_seq_len or int(ds_cfg.get("max_seq_len", 1500))
    positional_omics = bool(ds_cfg.get("positional_omics", False))
    omics_mode = ds_cfg.get("omics_feature_mode", "coverage_score")
    score_transform = ds_cfg.get("score_transform", "log1p")
    selected_beds = ds_cfg.get("selected_bed_features")
    kmer_max_k = int(ds_cfg.get("kmer_max_k", 0))
    kmer_window_count = int(ds_cfg.get("kmer_window_count", 0))
    nucleotide_level = bool(model_cfg.get("nucleotide_level", True))

    dataset = TriplexInferenceDataset(
        fasta_path=args.fasta,
        bed_dir=args.bed_dir,
        max_seq_len=max_seq_len,
        positional_omics=positional_omics,
        nucleotide_level=nucleotide_level,
        omics_feature_mode=omics_mode,
        score_transform=score_transform,
        selected_bed_features=selected_beds,
        kmer_max_k=kmer_max_k,
        kmer_window_count=kmer_window_count,
    )

    model_cfg["n_omics_features"] = int(dataset.feature_dim)
    model_cfg["n_kmer_features"] = int(dataset.kmer_feature_dim)
    model = TriplexNet(**model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(ckpt.get("best_threshold", 0.5))
    )
    topk_ratio = float(loss_cfg.get("top_k_ratio", 0.2))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=inference_collate,
    )

    region_records = []
    nuc_records: List[Tuple[str, int, int, float]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**batch)
            logits = outputs["logits"]
            mask = batch.get("mask")
            nuc_probs = torch.sigmoid(logits)

            if nuc_probs.dim() == 2:
                seq_probs = _aggregate_sequence_probs(nuc_probs, mask, topk_ratio)
            else:
                seq_probs = nuc_probs.view(nuc_probs.size(0), -1).mean(dim=1)

            seq_preds = (seq_probs >= threshold).long()

            for i, (chrom, start, end, strand) in enumerate(batch["coords"]):
                prob = float(seq_probs[i].detach().cpu())
                pred = int(seq_preds[i].detach().cpu())
                region_records.append((chrom, int(start), int(end), strand, prob, pred))

                if args.nuc_out:
                    if mask is not None and mask.dim() == 2:
                        valid_len = int(mask[i].sum().cpu().item())
                    else:
                        valid_len = nuc_probs.size(1)
                    probs = nuc_probs[i].detach().cpu().numpy()[:valid_len]
                    for offset, p in enumerate(probs):
                        nuc_records.append(
                            (
                                chrom,
                                int(start) + offset,
                                int(start) + offset + 1,
                                float(p),
                            )
                        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("chrom\tstart\tend\tstrand\tseq_prob\tseq_pred\tthreshold\n")
        for chrom, start, end, strand, prob, pred in region_records:
            f.write(
                f"{chrom}\t{start}\t{end}\t{strand}\t{prob:.6f}\t{pred}\t{threshold:.3f}\n"
            )
    logger.info(f"Wrote region predictions to {out_path}")

    if args.nuc_out and nuc_records:
        nuc_path = Path(args.nuc_out)
        nuc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nuc_path, "w") as f:
            for chrom, start, end, p in nuc_records:
                f.write(f"{chrom}\t{start}\t{end}\t{p:.6f}\n")
        logger.info(f"Wrote nucleotide probabilities to {nuc_path}")


if __name__ == "__main__":
    main()

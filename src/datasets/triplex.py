import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TriplexDataset(Dataset):
    _COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

    def __init__(
        self,
        pos_fasta_path: str,
        neg_fasta_path: str,
        bed_dir: str,
        max_seq_len: int = 1000,
        limit: int = None,
        name: str = "train",
        positional_omics: bool = False,
        nucleotide_level: bool = False,
        rc_augment: bool = False,
        selected_bed_features: List[str] = None,
        omics_feature_mode: str = "coverage",
        score_transform: str = "log1p",
    ):
        """
        Args:
            pos_fasta_path (str): path to positive samples FASTA file.
            neg_fasta_path (str): path to negative samples FASTA file.
            bed_dir (str): directory with BED files for omics features.
            max_seq_len (int): maximum sequence length.
            limit (int | None): limit number of samples (for debugging).
            name (str): dataset partition name.
            positional_omics (bool): if True, use position-specific omics features.
            nucleotide_level (bool): if True, return nucleotide-level labels [seq_len].
            rc_augment (bool): if True, randomly return reverse complement of sequence
                               (50% chance per sample). Only apply during training.
            selected_bed_features (list[str] | None): optional BED stem whitelist.
            omics_feature_mode (str): one of {"coverage", "score_mean", "coverage_score"}.
                - coverage: overlap fraction per track
                - score_mean: overlap-weighted normalized BED score per track
                - coverage_score: concatenate coverage + score_mean per track
            score_transform (str): one of {"none", "log1p"} for BED score normalization.
        """
        self.name = name
        self.max_seq_len = max_seq_len
        self.positional_omics = positional_omics
        self.nucleotide_level = nucleotide_level
        self.rc_augment = rc_augment
        self.selected_bed_features = (
            set(selected_bed_features) if selected_bed_features else None
        )
        self.omics_feature_mode = omics_feature_mode
        self.score_transform = score_transform

        valid_modes = {"coverage", "score_mean", "coverage_score"}
        if self.omics_feature_mode not in valid_modes:
            raise ValueError(
                f"omics_feature_mode must be one of {sorted(valid_modes)}, got {self.omics_feature_mode!r}"
            )
        valid_transforms = {"none", "log1p"}
        if self.score_transform not in valid_transforms:
            raise ValueError(
                f"score_transform must be one of {sorted(valid_transforms)}, got {self.score_transform!r}"
            )

        pos_data = self._parse_fasta_with_coords(Path(pos_fasta_path))
        neg_data = self._parse_fasta_with_coords(Path(neg_fasta_path))

        bed_dir = Path(bed_dir)
        self.bed_data, self.bed_names, self.bed_score_scales = self._load_bed_files(
            bed_dir
        )
        self.base_feature_dim = len(self.bed_names)
        self.feature_dim = self._infer_feature_dim()
        logger.info(f"  Loaded {len(self.bed_names)} omics features")
        logger.info(
            f"  Omics mode={self.omics_feature_mode}, score_transform={self.score_transform}, "
            f"feature_dim={self.feature_dim}"
        )

        self.data = []
        for chrom, start, end, strand, seq in pos_data:
            self.data.append((chrom, start, end, strand, seq, 1))
        for chrom, start, end, strand, seq in neg_data:
            self.data.append((chrom, start, end, strand, seq, 0))

        if limit is not None:
            self.data = self.data[:limit]

        logger.info(f"Total {name} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chrom, start, end, strand, seq, label = self.data[idx]

        if self.rc_augment and torch.rand(1).item() < 0.5:
            seq = seq.translate(self._COMPLEMENT)[::-1]

        sequence = self._one_hot_encode(seq)
        omics_features = self._extract_omics_features(chrom, start, end)

        actual_seq_len = min(len(seq), self.max_seq_len)

        if self.nucleotide_level:
            labels = torch.zeros(self.max_seq_len, dtype=torch.float32)
            if label == 1:
                labels[:actual_seq_len] = 1.0

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
        }

        if mask is not None:
            result["mask"] = mask

        return result

    @staticmethod
    def _parse_fasta_with_coords(
        fasta_file: Path,
    ) -> List[Tuple[str, int, int, str, str]]:
        """Parse FASTA file and extract coordinates from header"""
        sequences = []
        with open(fasta_file, "r") as f:
            header = None
            seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if header and seq:
                        sequences.append((header, "".join(seq)))
                    header = line[1:]
                    seq = []
                else:
                    seq.append(line.upper())
            if header and seq:
                sequences.append((header, "".join(seq)))

        parsed_data = []
        for header, seq in sequences:
            parts = header.split(":")
            if len(parts) >= 5:
                chrom = parts[2]
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[5] if len(parts) > 5 else "+"
                parsed_data.append((chrom, start, end, strand, seq))

        return parsed_data

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode DNA sequence.
        A, C, G, T -> one-hot vectors
        N -> [0.25, 0.25, 0.25, 0.25]
        """
        nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
        encoded = np.zeros((self.max_seq_len, 4), dtype=np.float32)

        for i, nuc in enumerate(sequence[: self.max_seq_len]):
            if nuc in nucleotide_dict:
                encoded[i, nucleotide_dict[nuc]] = 1.0
            elif nuc == "N":
                encoded[i, :] = 0.25

        return encoded

    def _load_bed_files(self, bed_dir: Path):
        bed_files = sorted(list(bed_dir.glob("*.bed")))
        if self.selected_bed_features is not None:
            bed_files = [
                bf for bf in bed_files if bf.stem in self.selected_bed_features
            ]
            missing = sorted(self.selected_bed_features - {bf.stem for bf in bed_files})
            if missing:
                logger.warning(
                    f"Requested BED features not found and will be ignored: {missing}"
                )

        bed_data = {}
        bed_score_scales = {}

        for bed_file in tqdm(bed_files, desc="Loading BED files"):
            intervals = {}
            raw_scores = []
            with open(bed_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    score = 1.0
                    if len(parts) >= 5:
                        try:
                            score = float(parts[4])
                        except ValueError:
                            score = 1.0

                    if chrom not in intervals:
                        intervals[chrom] = []
                    intervals[chrom].append((start, end, score))
                    if score > 0:
                        raw_scores.append(score)

            for chrom in intervals:
                intervals[chrom].sort()

            bed_data[bed_file.stem] = intervals
            if raw_scores:
                scale = float(np.percentile(np.array(raw_scores, dtype=np.float32), 95))
                bed_score_scales[bed_file.stem] = max(scale, 1.0)
            else:
                bed_score_scales[bed_file.stem] = 1.0

        return bed_data, [bf.stem for bf in bed_files], bed_score_scales

    def _infer_feature_dim(self) -> int:
        if self.positional_omics:
            if self.omics_feature_mode == "coverage_score":
                return self.max_seq_len * self.base_feature_dim * 2
            return self.max_seq_len * self.base_feature_dim
        if self.omics_feature_mode == "coverage_score":
            return self.base_feature_dim * 2
        return self.base_feature_dim

    def _transform_score(self, score: float, scale: float) -> float:
        score = max(float(score), 0.0)
        scale = max(float(scale), 1e-8)
        if self.score_transform == "log1p":
            return float(np.log1p(score) / np.log1p(scale))
        return float(score / scale)

    def _extract_omics_features(self, chrom: str, start: int, end: int) -> np.ndarray:
        if self.positional_omics:
            return self._extract_positional_omics(chrom, start, end)
        else:
            return self._extract_global_omics(chrom, start, end)

    def _extract_global_omics(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Extract global omics features from BED overlaps.

        Modes:
            coverage       -> overlap fraction per BED file
            score_mean     -> overlap-weighted normalized BED score per BED file
            coverage_score -> concatenate both for each BED file
        """
        coverage_features = []
        score_features = []
        region_len = end - start

        for bed_name in self.bed_names:
            intervals = self.bed_data[bed_name]
            score_scale = self.bed_score_scales.get(bed_name, 1.0)

            if chrom not in intervals:
                coverage_features.append(0.0)
                score_features.append(0.0)
                continue

            total_overlap = 0
            weighted_score_sum = 0.0
            for int_start, int_end, raw_score in intervals[chrom]:
                if int_end < start:
                    continue
                if int_start > end:
                    break

                overlap = min(end, int_end) - max(start, int_start)
                if overlap > 0:
                    total_overlap += overlap
                    weighted_score_sum += overlap * self._transform_score(
                        raw_score, score_scale
                    )

            coverage = total_overlap / region_len if region_len > 0 else 0.0
            score_mean = weighted_score_sum / region_len if region_len > 0 else 0.0
            coverage_features.append(coverage)
            score_features.append(score_mean)

        if self.omics_feature_mode == "coverage":
            return np.array(coverage_features, dtype=np.float32)
        if self.omics_feature_mode == "score_mean":
            return np.array(score_features, dtype=np.float32)
        return np.array(coverage_features + score_features, dtype=np.float32)

    def _extract_positional_omics(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Extract position-specific BED features [max_seq_len, n_features]."""
        region_len = min(end - start, self.max_seq_len)
        cov_features = np.zeros(
            (self.max_seq_len, len(self.bed_names)), dtype=np.float32
        )
        score_features = np.zeros(
            (self.max_seq_len, len(self.bed_names)), dtype=np.float32
        )

        for bed_idx, bed_name in enumerate(self.bed_names):
            intervals = self.bed_data[bed_name]
            score_scale = self.bed_score_scales.get(bed_name, 1.0)

            if chrom not in intervals:
                continue

            for int_start, int_end, raw_score in intervals[chrom]:
                if int_end < start:
                    continue
                if int_start > end:
                    break

                overlap_start = max(start, int_start) - start
                overlap_end = min(end, int_end) - start

                if overlap_start < self.max_seq_len and overlap_end > 0:
                    sl = slice(overlap_start, min(overlap_end, self.max_seq_len))
                    cov_features[sl, bed_idx] = 1.0
                    score_features[sl, bed_idx] = np.maximum(
                        score_features[sl, bed_idx],
                        self._transform_score(raw_score, score_scale),
                    )

        cov_features[region_len:, :] = 0.0
        score_features[region_len:, :] = 0.0

        if self.omics_feature_mode == "coverage":
            return cov_features.flatten()
        if self.omics_feature_mode == "score_mean":
            return score_features.flatten()
        return np.concatenate([cov_features, score_features], axis=1).flatten()

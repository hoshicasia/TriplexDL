import logging
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TriplexDataset(Dataset):
    _COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

    @staticmethod
    def _normalize_chrom(chrom: str) -> str:
        """Normalize chromosome names to a single style (chr-prefixed)."""
        c = str(chrom).strip()
        if not c:
            return c
        if not c.startswith("chr"):
            return f"chr{c}"
        return c

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
        coord_shift_max: int = 0,
        nuc_mask_prob: float = 0.0,
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
            coord_shift_max (int): max random shift (±bp) applied to BED lookup coords
                                   for omics augmentation. 0 disables.
            nuc_mask_prob (float): per-position probability of zeroing the one-hot
                                   encoding (nucleotide masking augmentation). 0 disables.
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
        self.coord_shift_max = coord_shift_max
        self.nuc_mask_prob = nuc_mask_prob
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

        self._verify_omics_overlap()

        logger.info(f"Total {name} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chrom, start, end, strand, seq, label = self.data[idx]

        do_rc = self.rc_augment and torch.rand(1).item() < 0.5
        if do_rc:
            seq = seq.translate(self._COMPLEMENT)[::-1]

        if self.coord_shift_max > 0:
            shift = int(
                torch.randint(
                    -self.coord_shift_max, self.coord_shift_max + 1, (1,)
                ).item()
            )
            start = max(0, start + shift)
            end = max(start + 1, end + shift)

        sequence = self._one_hot_encode(seq)

        if self.nuc_mask_prob > 0.0:
            mask_vec = np.random.rand(sequence.shape[0]) < self.nuc_mask_prob
            sequence[mask_vec] = 0.0

        omics_features = self._extract_omics_features(chrom, start, end)

        if do_rc and self.positional_omics:
            actual_len = min(end - start, self.max_seq_len)
            if self.omics_feature_mode == "coverage_score":
                mat = omics_features.reshape(self.max_seq_len, -1)
            else:
                mat = omics_features.reshape(self.max_seq_len, -1)
            mat[:actual_len] = mat[:actual_len][::-1].copy()
            omics_features = mat.flatten()

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
                chrom = TriplexDataset._normalize_chrom(parts[2])
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[5] if len(parts) > 5 else "+"
                parsed_data.append((chrom, start, end, strand, seq))

        return parsed_data

    _NUC_TO_IDX = np.frompyfunc(lambda c: {"A": 0, "C": 1, "G": 2, "T": 3}.get(c, -1), 1, 1)

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """Vectorized one-hot encoding.  A/C/G/T -> one-hot, N -> 0.25."""
        seq = sequence[: self.max_seq_len]
        arr = np.array(list(seq), dtype="U1")
        idx = np.array([{"A": 0, "C": 1, "G": 2, "T": 3}.get(c, -1) for c in arr], dtype=np.int8)
        encoded = np.zeros((self.max_seq_len, 4), dtype=np.float32)
        valid = idx >= 0
        rows = np.arange(len(arr))
        encoded[rows[valid], idx[valid]] = 1.0
        n_mask = ~valid & (arr == "N") if len(arr) > 0 else np.zeros(0, dtype=bool)
        encoded[rows[n_mask]] = 0.25
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
                    chrom = self._normalize_chrom(parts[0])
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

            max_width = 0
            for chrom in intervals:
                intervals[chrom].sort()
                for s, e, _ in intervals[chrom]:
                    max_width = max(max_width, e - s)

            bed_data[bed_file.stem] = intervals
            bed_data[bed_file.stem + "__max_width__"] = max_width
            if raw_scores:
                scale = float(np.percentile(np.array(raw_scores, dtype=np.float32), 95))
                bed_score_scales[bed_file.stem] = max(scale, 1.0)
            else:
                bed_score_scales[bed_file.stem] = 1.0

        return bed_data, [bf.stem for bf in bed_files], bed_score_scales

    def _verify_omics_overlap(self):
        """Sanity-check that omics features are not globally zero due to coordinate mismatch."""
        n_check = min(100, len(self.data))
        if n_check == 0:
            return

        n_nonzero = 0
        sample_chroms = set()
        for i in range(n_check):
            chrom, start, end, _strand, _seq, _label = self.data[i]
            sample_chroms.add(chrom)
            feats = self._extract_omics_features(chrom, start, end)
            if np.any(feats > 0):
                n_nonzero += 1

        bed_chroms = set()
        for bed_name in self.bed_names:
            bed_chroms.update(self.bed_data[bed_name].keys())

        if sample_chroms and bed_chroms and not (sample_chroms & bed_chroms):
            raise ValueError(
                "No chromosome overlap between FASTA headers and BED files after normalization. "
                "This typically causes all omics features to be zero. "
                f"sample_chrom_examples={sorted(list(sample_chroms))[:5]}, "
                f"bed_chrom_examples={sorted(list(bed_chroms))[:5]}"
            )

        if n_nonzero == 0:
            logger.warning(
                "Omics sanity check: all tested samples produced zero omics features "
                "(checked %d samples). Verify BED coordinates and preprocessing.",
                n_check,
            )
        else:
            logger.info(
                "Omics sanity check: %d/%d samples have non-zero omics features.",
                n_nonzero,
                n_check,
            )

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

            chrom_intervals = intervals[chrom]
            max_w = self.bed_data.get(bed_name + "__max_width__", 1_000_000)
            search_start = max(0, start - max_w)
            lo = bisect_left(chrom_intervals, (search_start,))

            total_overlap = 0
            weighted_score_sum = 0.0
            for j in range(lo, len(chrom_intervals)):
                int_start, int_end, raw_score = chrom_intervals[j]
                if int_start > end:
                    break

                overlap = min(end, int_end) - max(start, int_start)
                if overlap > 0:
                    total_overlap += overlap
                    weighted_score_sum += overlap * self._transform_score(
                        raw_score, score_scale
                    )

            coverage = min(total_overlap / region_len, 1.0) if region_len > 0 else 0.0
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

            chrom_intervals = intervals[chrom]
            max_w = self.bed_data.get(bed_name + "__max_width__", 1_000_000)
            search_start = max(0, start - max_w)
            lo = bisect_left(chrom_intervals, (search_start,))

            for j in range(lo, len(chrom_intervals)):
                int_start, int_end, raw_score = chrom_intervals[j]
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

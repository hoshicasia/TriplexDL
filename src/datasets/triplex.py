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
        nuc_mask_prob: float = 0.0,
        coord_shift_max: int = 0,
        kmer_max_k: int = 0,
        kmer_window_count: int = 0,
    ):
        self.name = name
        self.max_seq_len = max_seq_len
        self.positional_omics = positional_omics
        self.nucleotide_level = nucleotide_level
        self.rc_augment = rc_augment
        self.nuc_mask_prob = nuc_mask_prob
        self.coord_shift_max = coord_shift_max
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
        if self.coord_shift_max > 0:
            logger.warning("coord_shift_max > 0, disabling to prevent misalignment")
            self.coord_shift_max = 0
        pos_data = self._parse_fasta_with_coords(Path(pos_fasta_path))
        neg_data = self._parse_fasta_with_coords(Path(neg_fasta_path))

        bed_dir = Path(bed_dir)
        self.bed_data, self.bed_names, self.bed_score_scales = self._load_bed_files(
            bed_dir
        )
        self.base_feature_dim = len(self.bed_names)
        self.kmer_feature_dim = self._kmer_feature_dim
        self.feature_dim = self._infer_feature_dim()
        logger.info(f"  Loaded {len(self.bed_names)} omics features")

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

    def _verify_omics_overlap(self):
        """Check that at least some samples have non-zero omics features."""
        n_check = min(100, len(self.data))
        n_nonzero = 0
        sample_chroms = set()
        bed_chroms = set()
        for i in range(n_check):
            chrom, start, end, strand, seq, label = self.data[i]
            sample_chroms.add(chrom)
            feats = self._extract_omics_features(chrom, start, end)
            if feats.any():
                n_nonzero += 1
        for bed_name in self.bed_names[:1]:
            bed_chroms.update(self.bed_data[bed_name].keys())

    def __getitem__(self, idx):
        chrom, start, end, strand, seq, label = self.data[idx]

        if self.rc_augment and torch.rand(1).item() < 0.5:
            seq = seq.translate(self._COMPLEMENT)[::-1]

        aug_start, aug_end = start, end
        if self.coord_shift_max > 0 and self.rc_augment:
            shift = int(
                torch.randint(
                    -self.coord_shift_max, self.coord_shift_max + 1, (1,)
                ).item()
            )
            aug_start = max(0, start + shift)
            aug_end = max(aug_start + 1, end + shift)

        sequence = self._one_hot_encode(seq)
        omics_features = self._extract_omics_features(chrom, aug_start, aug_end)
        kmer_features = None
        if self._kmer_feature_dim > 0:
            kmer_features = self._compute_local_kmer_features(seq)

        if self.nuc_mask_prob > 0 and self.rc_augment:
            seq_len = min(len(seq), self.max_seq_len)
            mask_rand = torch.rand(seq_len)
            mask_positions = (mask_rand < self.nuc_mask_prob).numpy()
            if mask_positions.any():
                sequence[:seq_len][mask_positions] = 0.25

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

        if kmer_features is not None:
            result["kmer_features"] = torch.FloatTensor(kmer_features)

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
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[5] if len(parts) > 5 else "+"
                parsed_data.append((chrom, start, end, strand, seq))

        return parsed_data

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode DNA sequence.
        A, C, G, T into one-hot vectors
        N are encoded with uniform probability [0.25, 0.25, 0.25, 0.25]
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
                    if not chrom.startswith("chr"):
                        chrom = f"chr{chrom}"
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

    def _compute_local_kmer_features(self, seq: str) -> np.ndarray:
        """Compute per-window normalized k-mer frequencies.

        The sequence is trimmed to ``max_seq_len`` and split into
        ``kmer_window_count`` contiguous windows. For each window, normalized
        k-mer frequencies for k=1..kmer_max_k are computed and concatenated.

        Returns:
            np.ndarray of shape [kmer_window_count, sum(4^k for k in 1..kmer_max_k)].
        """
        seq = seq[: self.max_seq_len]
        features = np.zeros(
            (self.kmer_window_count, self._kmer_feature_dim), dtype=np.float32
        )
        if not seq:
            return features

        boundaries = np.linspace(0, len(seq), self.kmer_window_count + 1, dtype=int)
        for window_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            window_seq = seq[start:end]
            if not window_seq:
                continue
            features[window_idx] = self._compute_window_kmer_vector(window_seq)

        return features

    def _compute_window_kmer_vector(self, seq: str) -> np.ndarray:
        features = np.zeros(self._kmer_feature_dim, dtype=np.float32)
        nuc_idx = np.array([self._nuc_to_idx.get(c, -1) for c in seq], dtype=np.int8)

        offset = 0
        for k in range(1, self.kmer_max_k + 1):
            n_possible = 4**k
            n_kmers = len(seq) - k + 1
            if n_kmers <= 0:
                offset += n_possible
                continue

            kmer_indices = np.zeros(n_kmers, dtype=np.int32)
            valid = np.ones(n_kmers, dtype=bool)
            for j in range(k):
                col = nuc_idx[j : j + n_kmers]
                valid &= col >= 0
                kmer_indices += col.astype(np.int32) * (4 ** (k - 1 - j))

            valid_indices = kmer_indices[valid]
            if len(valid_indices) > 0:
                counts = np.bincount(valid_indices, minlength=n_possible).astype(
                    np.float32
                )
                features[offset : offset + n_possible] = counts / len(valid_indices)

            offset += n_possible

        return features

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
        """Extract global omics features from BED overlaps."""
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

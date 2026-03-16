import torch
import torch.nn.functional as F
from torch import nn


class TriplexMILLoss(nn.Module):
    """
    MIL loss for triplex prediction with configurable sequence aggregation.
    """

    def __init__(
        self,
        top_k_ratio: float = 0.2,
        pos_weight: float = 1.0,
        nuc_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
        aggregation_mode: str = "attention_noisy_or",
        attention_temperature: float = 1.0,
        attention_mix_alpha: float = 0.6,
        attention_entropy_weight: float = 0.0,
        positive_sparsity_weight: float = 0.0,
        negative_sparsity_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        smoothness_positive_only: bool = True,
        nucleotide_level: bool = True,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.pos_weight = pos_weight
        self.nuc_loss_weight = nuc_loss_weight
        self.label_smoothing = label_smoothing
        self.aggregation_mode = aggregation_mode
        self.attention_temperature = max(float(attention_temperature), 1e-3)
        self.attention_mix_alpha = float(attention_mix_alpha)
        self.attention_entropy_weight = float(attention_entropy_weight)
        self.positive_sparsity_weight = float(positive_sparsity_weight)
        self.negative_sparsity_weight = float(negative_sparsity_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.smoothness_positive_only = bool(smoothness_positive_only)

        valid_modes = {"topk", "attention", "noisy_or", "attention_noisy_or"}
        if self.aggregation_mode not in valid_modes:
            raise ValueError(
                f"aggregation_mode must be one of {sorted(valid_modes)}, got {self.aggregation_mode!r}"
            )

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor = None, dim: int = -1):
        if mask is None:
            return torch.softmax(scores, dim=dim)

        mask_bool = mask.bool()
        masked_scores = scores.masked_fill(~mask_bool, float("-inf"))
        weights = torch.softmax(masked_scores, dim=dim)

        all_invalid = (~mask_bool).all(dim=dim, keepdim=True)
        weights = torch.where(all_invalid, torch.zeros_like(weights), weights)
        return weights

    @staticmethod
    def _to_sequence_labels(
        label: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if label.dim() == 1:
            return label.float()
        if mask is not None:
            return ((label * mask).sum(dim=1) > 0).float()
        return (label.sum(dim=1) > 0).float()

    @staticmethod
    def _masked_mean_per_sequence(
        values: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is None or values.shape != mask.shape:
            return values.mean(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (values * mask).sum(dim=1) / denom

    @staticmethod
    def _tv_smoothness(
        probs: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if probs.size(1) < 2:
            return torch.zeros((), device=probs.device, dtype=probs.dtype)

        diffs = torch.abs(probs[:, 1:] - probs[:, :-1])
        if mask is None or probs.shape != mask.shape:
            return diffs.mean()

        pair_mask = (mask[:, 1:] * mask[:, :-1]).float()
        return (diffs * pair_mask).sum() / (pair_mask.sum() + 1e-8)

    def _aggregate_topk(
        self, probs: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, _ = probs.shape
        seq_probs = []

        if mask is not None and probs.shape == mask.shape:
            valid_mask = mask.bool()
            seq_lengths = valid_mask.sum(dim=1).long().clamp(min=1)
            k_vals = (seq_lengths.float() * self.top_k_ratio).long().clamp(min=1)

            masked_probs = probs.clone()
            masked_probs[~valid_mask] = -1.0
            sorted_probs, _ = masked_probs.sort(dim=1, descending=True)
            for i in range(bsz):
                seq_probs.append(sorted_probs[i, : k_vals[i].item()].mean())
            return torch.stack(seq_probs)

        k = max(1, int(probs.size(1) * self.top_k_ratio))
        topk_probs, _ = probs.topk(k, dim=1)
        return topk_probs.mean(dim=1)

    def _aggregate_attention(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        mask: torch.Tensor = None,
        return_entropy: bool = False,
    ):
        attn_scores = logits / self.attention_temperature
        weights = self._masked_softmax(attn_scores, mask=mask, dim=1)
        seq_prob = (weights * probs).sum(dim=1)

        if not return_entropy:
            return seq_prob, None

        entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1)
        if mask is not None and probs.shape == mask.shape:
            valid_count = mask.sum(dim=1).float().clamp(min=2.0)
            entropy = entropy / torch.log(valid_count)
        else:
            entropy = entropy / torch.log(
                torch.tensor(float(probs.size(1)), device=probs.device).clamp(min=2.0)
            )
        return seq_prob, entropy.mean()

    @staticmethod
    def _aggregate_noisy_or(
        probs: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)
        if mask is not None and probs.shape == mask.shape:
            valid_mask = mask.bool()
            probs = torch.where(valid_mask, probs, torch.zeros_like(probs))
        log_not = torch.log1p(-probs)
        sum_log_not = log_not.sum(dim=1)
        seq_prob = 1.0 - torch.exp(sum_log_not)
        return seq_prob.clamp(min=1e-6, max=1.0 - 1e-6)

    def aggregate_sequence_probs_from_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        return_entropy: bool = False,
    ):
        probs = torch.sigmoid(logits)
        return self.aggregate_sequence_probs(
            probs, mask=mask, logits=logits, return_entropy=return_entropy
        )

    def aggregate_sequence_probs(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor = None,
        logits: torch.Tensor = None,
        return_entropy: bool = False,
    ):
        if probs.dim() != 2:
            flat = probs.view(probs.size(0), -1).mean(dim=1)
            return (flat, None) if return_entropy else flat

        mode = self.aggregation_mode
        if mode == "topk":
            seq_prob = self._aggregate_topk(probs, mask=mask)
            return (seq_prob, None) if return_entropy else seq_prob

        if mode == "noisy_or":
            seq_prob = self._aggregate_noisy_or(probs, mask=mask)
            return (seq_prob, None) if return_entropy else seq_prob

        if logits is None:
            logits = torch.logit(probs.clamp(min=1e-6, max=1.0 - 1e-6))

        attn_prob, entropy = self._aggregate_attention(
            logits=logits,
            probs=probs,
            mask=mask,
            return_entropy=return_entropy,
        )

        if mode == "attention":
            return (attn_prob, entropy) if return_entropy else attn_prob

        noisy_prob = self._aggregate_noisy_or(probs, mask=mask)
        alpha = self.attention_mix_alpha
        hybrid_prob = alpha * attn_prob + (1.0 - alpha) * noisy_prob
        hybrid_prob = hybrid_prob.clamp(min=1e-6, max=1.0 - 1e-6)
        return (hybrid_prob, entropy) if return_entropy else hybrid_prob

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor = None,
        **batch,
    ):
        """
        Args:
            logits: [B, L] per-nucleotide logits
            label:  [B, L] per-nucleotide labels (uniform: all 1 or all 0)
            mask:   [B, L] valid position mask (1=valid, 0=padding)
        """
        seq_label_hard = self._to_sequence_labels(label, mask=mask)
        seq_label = seq_label_hard

        if self.label_smoothing > 0:
            seq_label = (
                seq_label * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )

        seq_prob, attn_entropy = self.aggregate_sequence_probs_from_logits(
            logits, mask=mask, return_entropy=True
        )
        seq_logit = torch.logit(seq_prob.clamp(min=1e-6, max=1.0 - 1e-6))
        losses = F.binary_cross_entropy_with_logits(
            seq_logit, seq_label, reduction="none"
        )

        pos_mask = seq_label_hard > 0.5
        sample_weights = torch.where(
            pos_mask,
            torch.full_like(losses, self.pos_weight),
            torch.ones_like(losses),
        )
        mil_loss = (losses * sample_weights).sum() / (sample_weights.sum() + 1e-8)

        total_loss = mil_loss

        if self.attention_entropy_weight > 0 and attn_entropy is not None:
            total_loss = total_loss + self.attention_entropy_weight * attn_entropy

        nuc_probs = torch.sigmoid(logits)
        mean_prob_per_seq = self._masked_mean_per_sequence(nuc_probs, mask=mask)

        if self.positive_sparsity_weight > 0:
            pos_rows = seq_label_hard > 0.5
            if pos_rows.any():
                pos_sparse = mean_prob_per_seq[pos_rows].mean()
                total_loss = total_loss + self.positive_sparsity_weight * pos_sparse

        if self.negative_sparsity_weight > 0:
            neg_rows = seq_label_hard <= 0.5
            if neg_rows.any():
                neg_sparse = mean_prob_per_seq[neg_rows].mean()
                total_loss = total_loss + self.negative_sparsity_weight * neg_sparse

        if self.smoothness_weight > 0:
            if self.smoothness_positive_only:
                pos_rows = seq_label_hard > 0.5
                if pos_rows.any():
                    pos_probs = nuc_probs[pos_rows]
                    pos_mask = mask[pos_rows] if mask is not None else None
                    smooth = self._tv_smoothness(pos_probs, mask=pos_mask)
                    total_loss = total_loss + self.smoothness_weight * smooth
            else:
                smooth = self._tv_smoothness(nuc_probs, mask=mask)
                total_loss = total_loss + self.smoothness_weight * smooth

        if self.nuc_loss_weight > 0:
            smooth_label = label.float()
            if self.label_smoothing > 0:
                smooth_label = (
                    smooth_label * (1.0 - self.label_smoothing)
                    + 0.5 * self.label_smoothing
                )
            nuc_bce = F.binary_cross_entropy_with_logits(
                logits, smooth_label, reduction="none"
            )
            if mask is not None:
                nuc_bce = (nuc_bce * mask).sum() / (mask.sum() + 1e-8)
            else:
                nuc_bce = nuc_bce.mean()
            total_loss = total_loss + self.nuc_loss_weight * nuc_bce

        return {"loss": total_loss}

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
)


class BaseMetric(Metric):
    """
    Base class for metrics.
    Supports both region-level and nucleotide-level predictions.
    """

    def __init__(
        self, name: str = None, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        base_name = name if name is not None else self.__class__.__name__
        prefix = "nuc_" if nucleotide_level else "seq_"
        self.name = prefix + base_name
        self.nucleotide_level = nucleotide_level
        self.metric = None

    def __call__(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor = None,
        seq_logit: torch.Tensor = None,
        **kwargs,
    ):
        if self.nucleotide_level:
            probs = torch.sigmoid(logits.view(-1))
            label_flat = label.view(-1).long()

            if mask is not None:
                mask_flat = mask.view(-1).bool()
                probs = probs[mask_flat]
                label_flat = label_flat[mask_flat]

            return self.update_and_compute(probs, label_flat)
        else:
            # For sequence-level metrics: use aux seq_logit if available,
            # otherwise aggregate per-nucleotide logits via top-k mean.
            if seq_logit is not None:
                probs = torch.sigmoid(seq_logit.view(-1))
            elif logits.dim() == 2:
                # Aggregate per-nucleotide probs to sequence-level
                nuc_probs = torch.sigmoid(logits)
                if mask is not None and logits.shape == mask.shape:
                    masked = nuc_probs.clone()
                    masked[~mask.bool()] = -1.0
                    sorted_probs, _ = masked.sort(dim=1, descending=True)
                    valid_counts = mask.sum(dim=1).long().clamp(min=1)
                    k_vals = (valid_counts.float() * 0.2).long().clamp(min=1)
                    probs = torch.stack(
                        [
                            sorted_probs[i, : k_vals[i].item()].mean()
                            for i in range(logits.size(0))
                        ]
                    )
                else:
                    k = max(1, int(logits.size(1) * 0.2))
                    topk_probs, _ = nuc_probs.topk(k, dim=1)
                    probs = topk_probs.mean(dim=1)
            else:
                probs = torch.sigmoid(logits.squeeze())

            if label.dim() > 1:
                seq_label = (label.sum(dim=1) > 0).long()
            else:
                seq_label = label.long()
            return self.update_and_compute(probs, seq_label)

    def update_and_compute(self, probs, label):
        self.update(probs, label)
        result = self.compute()
        if hasattr(result, "item"):
            return result.item()
        return result

    def reset(self):
        """Reset both outer and inner metric states to prevent memory accumulation."""
        super().reset()
        if self.metric is not None:
            self.metric.reset()

    def to(self, device):
        """Move both outer and inner metric to device."""
        super().to(device)
        if self.metric is not None:
            self.metric.to(device)
        return self


class AccuracyMetric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(
            name="accuracy", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.metric = BinaryAccuracy(threshold=threshold)

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class PrecisionMetric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(
            name="precision", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.metric = BinaryPrecision(threshold=threshold)

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class RecallMetric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(
            name="recall", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.metric = BinaryRecall(threshold=threshold)

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class F1Metric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(name="f1", nucleotide_level=nucleotide_level, *args, **kwargs)
        self.metric = BinaryF1Score(threshold=threshold)

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class MacroF1Metric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(
            name="macro_f1", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.threshold = threshold
        self.add_state("tp_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_neg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_neg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn_neg", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, probs, label):
        preds = (probs >= self.threshold).long()
        label = label.long()

        # Positive class
        self.tp_pos += ((preds == 1) & (label == 1)).sum().float()
        self.fp_pos += ((preds == 1) & (label == 0)).sum().float()
        self.fn_pos += ((preds == 0) & (label == 1)).sum().float()

        # Negative class
        self.tp_neg += ((preds == 0) & (label == 0)).sum().float()
        self.fp_neg += ((preds == 0) & (label == 1)).sum().float()
        self.fn_neg += ((preds == 1) & (label == 0)).sum().float()

    def compute(self):
        eps = 1e-8
        precision_pos = self.tp_pos / (self.tp_pos + self.fp_pos + eps)
        recall_pos = self.tp_pos / (self.tp_pos + self.fn_pos + eps)
        f1_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos + eps)

        precision_neg = self.tp_neg / (self.tp_neg + self.fp_neg + eps)
        recall_neg = self.tp_neg / (self.tp_neg + self.fn_neg + eps)
        f1_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg + eps)

        macro_f1 = (f1_pos + f1_neg) / 2
        return macro_f1.item()


class WeightedF1Metric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(
            name="weighted_f1", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.threshold = threshold
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, probs, label):
        preds = (probs >= self.threshold).long()
        label = label.long()

        self.tp += ((preds == 1) & (label == 1)).sum().float()
        self.fp += ((preds == 1) & (label == 0)).sum().float()
        self.fn += ((preds == 0) & (label == 1)).sum().float()
        self.tn += ((preds == 0) & (label == 0)).sum().float()

    def compute(self):
        eps = 1e-8
        f1_pos = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        f1_neg = (2.0 * self.tn) / (2.0 * self.tn + self.fp + self.fn + eps)

        support_pos = self.tp + self.fn
        support_neg = self.tn + self.fp
        weighted_f1 = (f1_pos * support_pos + f1_neg * support_neg) / (
            support_pos + support_neg + eps
        )
        return weighted_f1


class AUCMetric(BaseMetric):
    def __init__(self, nucleotide_level: bool = False, *args, **kwargs):
        super().__init__(name="auc", nucleotide_level=nucleotide_level, *args, **kwargs)
        self.metric = BinaryAUROC()

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class APMetric(BaseMetric):
    def __init__(self, nucleotide_level: bool = False, *args, **kwargs):
        super().__init__(
            name="avg_precision", nucleotide_level=nucleotide_level, *args, **kwargs
        )
        self.metric = BinaryAveragePrecision()

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()


class MCCMetric(BaseMetric):
    def __init__(
        self, threshold: float = 0.5, nucleotide_level: bool = False, *args, **kwargs
    ):
        super().__init__(name="mcc", nucleotide_level=nucleotide_level, *args, **kwargs)
        self.metric = BinaryMatthewsCorrCoef(threshold=threshold)

    def update(self, probs, label):
        self.metric.update(probs, label)

    def compute(self):
        return self.metric.compute()

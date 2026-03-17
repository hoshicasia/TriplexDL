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
    def __init__(
        self,
        name: str = None,
        nucleotide_level: bool = False,
        use_seq_logit: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name if name is not None else self.__class__.__name__
        self.nucleotide_level = nucleotide_level
        self.use_seq_logit = bool(use_seq_logit)
        self.metric = None

    def __call__(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor = None,
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

        seq_logit = kwargs.get("seq_logit") if self.use_seq_logit else None
        if seq_logit is not None:
            probs = torch.sigmoid(seq_logit.view(-1))
            if label.dim() > 1:
                seq_label = (label.sum(dim=1) > 0).long()
            else:
                seq_label = label.long().view(-1)
            return self.update_and_compute(probs, seq_label)

        if logits.dim() == 2:
            raise ValueError(
                "Sequence-level metric requested but got per-position logits without `seq_logit`. "
                "Enable/emit `seq_logit` from the model or set nucleotide_level=True."
            )

        probs = torch.sigmoid(logits.view(-1))
        seq_label = label.long().view(-1)
        return self.update_and_compute(probs, seq_label)

    def update_and_compute(self, probs, label):
        self.update(probs, label)
        return self.compute().item()

    def reset(self):
        super().reset()
        if self.metric is not None:
            self.metric.reset()

    def to(self, device):
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

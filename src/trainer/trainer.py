import logging

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        writer,
        start_epoch=1,
        best_val_metric=0,
        best_threshold=0.5,
        data_indices=None,
        resample_info=None,
        hard_neg_info=None,
    ):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders.get("val")
        self.test_dataloader = dataloaders.get("test")
        self.writer = writer
        self.start_epoch = start_epoch

        self.resample_info = resample_info

        self.hard_neg_info = hard_neg_info
        self.hard_neg_mining_freq = config.get("hard_neg_mining_freq", 0)
        self.hard_neg_ratio = config.get("hard_neg_ratio", 0.5)

        self.epochs = config.trainer.n_epochs
        self.save_dir = ROOT_PATH / config.trainer.save_dir / writer.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_step = config.trainer.get("log_step", 10)
        self.save_period = config.trainer.get("save_period", 5)

        self.best_val_metric = best_val_metric
        self.sequence_topk_ratio = float(getattr(self.criterion, "top_k_ratio", 0.2))
        self.best_threshold = best_threshold
        self.data_indices = data_indices
        self.threshold_tuning_during_training = config.trainer.get(
            "threshold_tuning_during_training", True
        )

        self.accumulate_grad_batches = config.trainer.get("accumulate_grad_batches", 1)
        if self.accumulate_grad_batches > 1:
            logger.info(
                f"Gradient accumulation: {self.accumulate_grad_batches} steps "
                f"(effective batch size = {config.dataloader.batch_size * self.accumulate_grad_batches})"
            )

        self.early_stop_patience = config.trainer.get("early_stop", 0)
        self.epochs_without_improvement = 0

        warmup_config = config.get("warmup", {})
        self.warmup_enabled = (
            warmup_config.get("enabled", False) if warmup_config else False
        )
        self.warmup_epochs = (
            warmup_config.get("warmup_epochs", 3) if warmup_config else 3
        )
        self.warmup_start_factor = (
            warmup_config.get("warmup_start_factor", 0.01) if warmup_config else 0.01
        )
        self.base_lr = optimizer.param_groups[0]["lr"]

        self.per_batch_scheduler = isinstance(
            lr_scheduler, torch.optim.lr_scheduler.OneCycleLR
        )

        if self.warmup_enabled and not self.per_batch_scheduler:
            warmup_lr = self.base_lr * self.warmup_start_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            logger.info(
                f"Warmup enabled: {self.warmup_epochs} epochs, "
                f"LR {warmup_lr:.6f} → {self.base_lr:.6f}"
            )
        elif self.per_batch_scheduler:
            logger.info("Using per-batch scheduler (OneCycleLR) - warmup is built-in")
            self.warmup_enabled = False

        self.tta_enabled = config.trainer.get("tta", False)

        logger.info(f"Trainer initialized. Saving to: {self.save_dir}")
        if self.tta_enabled:
            logger.info("Test-time augmentation (TTA) enabled: forward + reverse-complement averaging")
        if start_epoch > 1:
            logger.info(f"Resuming training from epoch {start_epoch}")
            logger.info(f"Best threshold: {self.best_threshold:.3f}")

    def _aggregate_sequence_probs(
        self, nuc_probs: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if hasattr(self.criterion, "aggregate_sequence_probs"):
            seq_prob = self.criterion.aggregate_sequence_probs(nuc_probs, mask=mask)
            if isinstance(seq_prob, tuple):
                seq_prob = seq_prob[0]
            return seq_prob

        if nuc_probs.dim() != 2:
            return nuc_probs.view(nuc_probs.size(0), -1).mean(dim=1)

        if mask is not None and nuc_probs.shape == mask.shape:
            valid_mask = mask.bool()
            seq_lengths = valid_mask.sum(dim=1).long().clamp(min=1)

            k_vals = (
                (seq_lengths.float() * self.sequence_topk_ratio).long().clamp(min=1)
            )
            masked = nuc_probs.clone()
            masked[~valid_mask] = -1.0
            sorted_probs, _ = masked.sort(dim=1, descending=True)
            seq_prob_list = []
            for i in range(nuc_probs.size(0)):
                k = k_vals[i].item()
                seq_prob_list.append(sorted_probs[i, :k].mean())
            return torch.stack(seq_prob_list)

        k = max(1, int(nuc_probs.size(1) * self.sequence_topk_ratio))
        topk_probs, _ = nuc_probs.topk(k, dim=1)
        return topk_probs.mean(dim=1)

    def _select_best_threshold(self, probs: torch.Tensor, labels: torch.Tensor):
        thresholds = torch.linspace(0.01, 0.99, 199)
        best_threshold = 0.5
        best_f1 = -1.0
        best_gap = float("inf")
        found_positive_f1 = False

        for t in thresholds:
            _, precision, recall, f1 = self._metrics_at_threshold(
                probs, labels, t.item()
            )
            gap = abs(precision - recall)

            if f1 > 0:
                if (
                    (not found_positive_f1)
                    or (f1 > best_f1)
                    or (abs(f1 - best_f1) < 1e-12 and gap < best_gap)
                ):
                    found_positive_f1 = True
                    best_f1 = f1
                    best_gap = gap
                    best_threshold = t.item()
            elif not found_positive_f1:
                if (gap < best_gap) or (abs(gap - best_gap) < 1e-12 and f1 > best_f1):
                    best_gap = gap
                    best_f1 = f1
                    best_threshold = t.item()

        return best_threshold, best_f1

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.warmup_enabled and epoch <= self.warmup_epochs:
                progress = epoch / self.warmup_epochs
                warmup_lr = self.base_lr * (
                    self.warmup_start_factor
                    + (1.0 - self.warmup_start_factor) * progress
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                logger.info(
                    f"Warmup epoch {epoch}/{self.warmup_epochs}: LR = {warmup_lr:.6f}"
                )

            logger.info("=" * 60)
            logger.info(f"Epoch {epoch}/{self.epochs}")
            logger.info(
                f"Current best val F1: {self.best_val_metric:.4f}, Threshold: {self.best_threshold:.3f}"
            )
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Current learning rate: {current_lr:.6f}")
            logger.info("=" * 60)

            if (
                self.hard_neg_info is not None
                and self.hard_neg_mining_freq > 0
                and epoch >= self.hard_neg_mining_freq
                and epoch % self.hard_neg_mining_freq == 0
            ):
                self._mine_hard_negatives(epoch)

            train_metrics = self._train_epoch(epoch)
            logger.info(f"Train metrics: {train_metrics}")

            if self.writer is not None:
                for metric_name, metric_value in train_metrics.items():
                    self.writer.experiment.log_metric(
                        f"train/{metric_name}", metric_value, step=epoch
                    )

            if self.val_dataloader is not None:
                val_metrics = self._val_epoch(epoch)
                logger.info(f"Val metrics: {val_metrics}")
                if "threshold" in val_metrics:
                    logger.info(
                        f"Optimal threshold for this epoch: {val_metrics['threshold']:.3f}"
                    )

                if self.writer is not None:
                    for metric_name, metric_value in val_metrics.items():
                        self.writer.experiment.log_metric(
                            f"val/{metric_name}", metric_value, step=epoch
                        )

                if val_metrics.get("f1", 0) > self.best_val_metric:
                    self.best_val_metric = val_metrics["f1"]
                    self.best_threshold = val_metrics[
                        "threshold"
                    ]
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(
                        f"New best model! F1: {self.best_val_metric:.4f} "
                        f"@ threshold={self.best_threshold:.3f}"
                    )
                else:
                    self.epochs_without_improvement += 1
                    logger.info(
                        f"No improvement for {self.epochs_without_improvement}/"
                        f"{self.early_stop_patience} epochs"
                    )

                if self.per_batch_scheduler:
                    pass
                elif isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if self.warmup_enabled and epoch <= self.warmup_epochs:
                        logger.info(
                            f"Skipping LR scheduler during warmup (epoch {epoch}/{self.warmup_epochs})"
                        )
                    else:
                        if self.lr_scheduler.mode == "max":
                            scheduler_metric = val_metrics.get("f1", 0)
                            logger.info(
                                f"ReduceLROnPlateau (mode=max): monitoring val_f1={scheduler_metric:.4f}"
                            )
                        else:
                            scheduler_metric = val_metrics["loss"]
                            logger.info(
                                f"ReduceLROnPlateau (mode=min): monitoring val_loss={scheduler_metric:.4f}"
                            )
                        self.lr_scheduler.step(scheduler_metric)
                else:
                    self.lr_scheduler.step()
            else:
                if self.per_batch_scheduler:
                    pass
                elif not isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step()

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=False)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate after epoch {epoch}: {current_lr:.6f}")

            if self.writer is not None:
                self.writer.experiment.log_metric(
                    "learning_rate", current_lr, step=epoch
                )

            if (
                self.early_stop_patience > 0
                and self.epochs_without_improvement >= self.early_stop_patience
            ):
                logger.info(
                    f"Early stopping triggered after {self.early_stop_patience} epochs "
                    f"without improvement. Best val F1: {self.best_val_metric:.4f}"
                )
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()

        if self.val_dataloader is None and self.test_dataloader is not None:
            logger.info(
                "No validation set configured; saving current model as best model for test evaluation"
            )
            self._save_checkpoint(self.epochs, is_best=True)

        self.test_metrics = None
        if self.test_dataloader is not None:
            best_model_path = self.save_dir / "model_best.pth"
            if best_model_path.exists():
                logger.info("=" * 60)
                logger.info(
                    f"Loading best model from {best_model_path} for final test evaluation..."
                )
                checkpoint = torch.load(
                    best_model_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["state_dict"])
                logger.info(
                    f"Best model was from epoch {checkpoint['epoch']} with val F1: {checkpoint['best_val_metric']:.4f}"
                )
            else:
                logger.warning(
                    f"Best model not found at {best_model_path}, using current model for test"
                )

            logger.info("=" * 60)
            logger.info("Running final test evaluation...")
            logger.info("=" * 60)
            test_metrics = self._test_epoch()
            self.test_metrics = test_metrics
            logger.info(f"Test metrics: {test_metrics}")

            if self.writer is not None:
                for metric_name, metric_value in test_metrics.items():
                    self.writer.experiment.log_metric(
                        f"test/{metric_name}", metric_value
                    )

    def _set_augmentation(self, enabled: bool):
        for dl in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            if dl is None:
                continue
            ds = dl.dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            if hasattr(ds, "training"):
                ds.training = enabled

    def _set_force_rc(self, value):
        for dl in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            if dl is None:
                continue
            ds = dl.dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            if hasattr(ds, "_force_rc"):
                ds._force_rc = value

    def _collect_predictions_with_tta(self, dataloader, desc, epoch, collect_nuc=True):
        self._set_force_rc(False)
        fwd = self._collect_predictions(dataloader, f"{desc}[fwd]", epoch, collect_nuc)

        self._set_force_rc(True)
        rc = self._collect_predictions(dataloader, f"{desc}[RC]", epoch, collect_nuc)

        self._set_force_rc(None)

        result = {"avg_loss": (fwd["avg_loss"] + rc["avg_loss"]) / 2}
        if "seq_probs" in fwd and "seq_probs" in rc:
            result["seq_probs"] = (fwd["seq_probs"] + rc["seq_probs"]) / 2
            result["seq_labels"] = fwd["seq_labels"]
        elif "seq_probs" in fwd:
            result["seq_probs"] = fwd["seq_probs"]
            result["seq_labels"] = fwd["seq_labels"]
        if collect_nuc and "nuc_probs" in fwd:
            result["nuc_probs"] = fwd["nuc_probs"]
            result["nuc_labels"] = fwd["nuc_labels"]

        logger.info(f"TTA: averaged forward + RC predictions for {desc}")
        return result

    def _resample_train_dataloader(self, epoch):
        info = self.resample_info
        rng = np.random.RandomState(info["seed"] + epoch)

        n_pos = len(info["pos_indices"])
        n_neg_pool = len(info["neg_indices"])
        n_sample = min(n_pos, n_neg_pool)

        sampled_neg = rng.choice(
            info["neg_indices"], size=n_sample, replace=False
        ).tolist()
        epoch_indices = info["pos_indices"] + sampled_neg
        rng.shuffle(epoch_indices)

        epoch_dataset = torch.utils.data.Subset(info["full_dataset"], epoch_indices)

        from hydra.utils import instantiate

        from src.utils.init_utils import set_worker_seed

        self.train_dataloader = instantiate(
            self.config.dataloader,
            dataset=epoch_dataset,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            worker_init_fn=set_worker_seed,
        )
        logger.info(
            f"  Resampled train: {n_pos} pos + {n_sample} neg "
            f"(from pool of {n_neg_pool})"
        )

    def _mine_hard_negatives(self, epoch):
        from hydra.utils import instantiate as hydra_instantiate

        from src.utils.init_utils import set_worker_seed

        info = self.hard_neg_info
        all_neg_indices = info["neg_indices"]
        pos_indices = info["pos_indices"]
        full_dataset = info["full_dataset"]
        target_ratio = info["target_ratio"]

        logger.info(
            f"  Hard-neg mining @ epoch {epoch}: "
            f"scoring {len(all_neg_indices)} negatives..."
        )

        neg_dataset = torch.utils.data.Subset(full_dataset, all_neg_indices)
        mine_bs = max(self.config.dataloader.batch_size * 4, 16)
        neg_loader = torch.utils.data.DataLoader(
            neg_dataset,
            batch_size=mine_bs,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        self.model.eval()
        self._set_augmentation(False)
        all_scores = []

        with torch.no_grad():
            for batch in neg_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)
                logits = outputs["logits"]
                nuc_probs = torch.sigmoid(logits)
                mask = batch.get("mask")

                if logits.dim() == 2:
                    seq_prob = self._aggregate_sequence_probs(nuc_probs, mask)
                    all_scores.extend(seq_prob.cpu().tolist())
                else:
                    all_scores.extend(
                        nuc_probs.view(nuc_probs.size(0), -1).mean(dim=1).cpu().tolist()
                    )

        all_scores = np.array(all_scores)

        n_pos = len(pos_indices)
        n_neg_target = min(int(n_pos * target_ratio), len(all_neg_indices))
        n_hard = min(int(n_neg_target * self.hard_neg_ratio), len(all_neg_indices))
        n_random = n_neg_target - n_hard

        sorted_by_score = np.argsort(all_scores)[::-1]
        hard_local = sorted_by_score[:n_hard].tolist()
        remaining_pool = sorted_by_score[n_hard:]

        rng = np.random.RandomState(info["seed"] + epoch)
        n_random_actual = min(n_random, len(remaining_pool))
        random_local = rng.choice(
            remaining_pool, size=n_random_actual, replace=False
        ).tolist()

        hard_neg_indices = [all_neg_indices[i] for i in hard_local]
        random_neg_indices = [all_neg_indices[i] for i in random_local]

        epoch_indices = list(pos_indices) + hard_neg_indices + random_neg_indices
        rng.shuffle(epoch_indices)

        epoch_dataset = torch.utils.data.Subset(full_dataset, epoch_indices)
        self.train_dataloader = hydra_instantiate(
            self.config.dataloader,
            dataset=epoch_dataset,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            worker_init_fn=set_worker_seed,
        )

        score_min = all_scores[hard_local[-1]] if hard_local else float("nan")
        score_max = all_scores[hard_local[0]] if hard_local else float("nan")
        logger.info(
            f"  → {n_pos} pos  |  {len(hard_neg_indices)} hard neg "
            f"(score [{score_min:.3f}–{score_max:.3f}])  |  "
            f"{len(random_neg_indices)} random neg"
        )

        if self.writer is not None:
            self.writer.experiment.log_metric(
                "hard_neg/min_score", score_min, step=epoch
            )
            self.writer.experiment.log_metric(
                "hard_neg/max_score", score_max, step=epoch
            )
            self.writer.experiment.log_metric(
                "hard_neg/n_hard", len(hard_neg_indices), step=epoch
            )

    def _train_epoch(self, epoch):
        if self.resample_info is not None:
            self._resample_train_dataloader(epoch)

        self.model.train()
        self._set_augmentation(True)
        total_loss = 0

        for m in self.metrics["train"]:
            m.reset()

        accum = self.accumulate_grad_batches
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = self.model(**batch)

            logits_detached = outputs["logits"].detach()
            batch["logits"] = logits_detached
            if "seq_logit" in outputs:
                batch["seq_logit"] = outputs["seq_logit"].detach()

            losses = self.criterion(
                logits=outputs["logits"],
                label=batch["label"],
                mask=batch.get("mask"),
            )
            loss = losses["loss"]

            if "aux_loss" in outputs:
                loss = loss + outputs["aux_loss"]

            loss = loss / accum

            loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(self.train_dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.per_batch_scheduler:
                    self.lr_scheduler.step()

            loss_value = loss.item() * accum
            total_loss += loss_value

            metric_values = {}
            for m in self.metrics["train"]:
                metric_val = m(**batch)
                metric_values[m.name] = (
                    metric_val.item() if torch.is_tensor(metric_val) else metric_val
                )
            pbar.set_postfix(loss=loss_value, **metric_values)

            del outputs, losses, loss

        avg_loss = total_loss / len(self.train_dataloader)
        metrics_result = {"loss": avg_loss}
        for m in self.metrics["train"]:
            metrics_result[m.name] = m.compute().item()

        return metrics_result

    def _evaluate_on_dataloader(self, dataloader, desc, epoch):
        self.model.eval()
        total_loss = 0

        for m in self.metrics["inference"]:
            m.reset()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch}")
            for batch in pbar:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**batch)
                batch["logits"] = outputs["logits"]
                if "seq_logit" in outputs:
                    batch["seq_logit"] = outputs["seq_logit"]

                losses = self.criterion(**batch)
                total_loss += losses["loss"].item()

                metric_values = {}
                for m in self.metrics["inference"]:
                    metric_val = m(**batch)
                    metric_values[m.name] = (
                        metric_val.item() if torch.is_tensor(metric_val) else metric_val
                    )

                pbar.set_postfix(**metric_values)

                del outputs, losses

        avg_loss = total_loss / len(dataloader)
        metrics_result = {"loss": avg_loss}
        for m in self.metrics["inference"]:
            metrics_result[m.name] = m.compute().item()

        return metrics_result

    def _val_epoch(self, epoch):
        if self.threshold_tuning_during_training:
            return self._val_epoch_with_threshold_tuning(epoch)

        metrics = self._evaluate_on_dataloader(self.val_dataloader, "Val", epoch)
        metrics["threshold"] = float(self.best_threshold)
        return metrics

    def _collect_predictions(self, dataloader, desc, epoch, collect_nuc=True):
        self.model.eval()
        self._set_augmentation(False)
        all_seq_probs = []
        all_seq_labels = []
        all_nuc_probs = []
        all_nuc_labels = []
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch}")
            for batch in pbar:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**batch)
                batch.update(outputs)

                losses = self.criterion(**batch)
                total_loss += losses["loss"].item()
                n_batches += 1

                logits = batch["logits"]
                labels = batch["label"]
                mask = batch.get("mask")

                nuc_probs = torch.sigmoid(logits)

                if collect_nuc:
                    if mask is not None and logits.shape == mask.shape:
                        mask_flat = mask.view(-1).bool()
                        all_nuc_probs.append(nuc_probs.view(-1)[mask_flat].cpu())
                        all_nuc_labels.append(labels.view(-1)[mask_flat].cpu())
                    else:
                        all_nuc_probs.append(nuc_probs.view(-1).cpu())
                        all_nuc_labels.append(labels.view(-1).cpu())

                if "seq_logit" in batch:
                    seq_prob = torch.sigmoid(batch["seq_logit"]).view(-1)
                    all_seq_probs.append(seq_prob.cpu())
                    if labels.dim() > 1:
                        seq_lab = (labels.sum(dim=1) > 0).float()
                    else:
                        seq_lab = labels.float()
                    all_seq_labels.append(seq_lab.cpu())
                elif logits.dim() == 2:
                    seq_prob = self._aggregate_sequence_probs(nuc_probs, mask)
                    all_seq_probs.append(seq_prob.cpu())
                    if labels.dim() > 1:
                        seq_lab = (labels.sum(dim=1) > 0).float()
                    else:
                        seq_lab = labels.float()
                    all_seq_labels.append(seq_lab.cpu())

        if n_batches == 0:
            raise RuntimeError(
                f"{desc} dataloader produced 0 batches. "
                "Check that val/test chromosome names in config match those in the FASTA headers."
            )

        avg_loss = total_loss / max(n_batches, 1)

        result = {"avg_loss": avg_loss}
        if all_nuc_probs:
            result["nuc_probs"] = torch.cat(all_nuc_probs)
            result["nuc_labels"] = torch.cat(all_nuc_labels).long()
        if all_seq_probs:
            result["seq_probs"] = torch.cat(all_seq_probs)
            result["seq_labels"] = torch.cat(all_seq_labels).long()

        return result

    def _metrics_at_threshold(self, all_probs, all_labels, threshold):
        preds = (all_probs >= threshold).long()
        tp = ((preds == 1) & (all_labels == 1)).sum().float()
        fp = ((preds == 1) & (all_labels == 0)).sum().float()
        fn = ((preds == 0) & (all_labels == 1)).sum().float()
        tn = ((preds == 0) & (all_labels == 0)).sum().float()
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return accuracy.item(), precision.item(), recall.item(), f1.item()

    def _val_epoch_with_threshold_tuning(self, epoch):
        from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

        compute_nuc_metrics = bool(
            self.config.trainer.get("compute_nuc_metrics", False)
        )
        if self.tta_enabled:
            preds = self._collect_predictions_with_tta(
                self.val_dataloader, "Val", epoch, collect_nuc=compute_nuc_metrics
            )
        else:
            preds = self._collect_predictions(
                self.val_dataloader, "Val", epoch, collect_nuc=compute_nuc_metrics
            )

        avg_loss = preds["avg_loss"]
        nuc_auc = float("nan")
        nuc_ap = float("nan")
        if compute_nuc_metrics and "nuc_probs" in preds:
            nuc_probs = preds["nuc_probs"]
            nuc_labels = preds["nuc_labels"]
            nuc_auc = BinaryAUROC()(nuc_probs, nuc_labels).item()
            nuc_ap = BinaryAveragePrecision()(nuc_probs, nuc_labels).item()
            logger.info(
                f"Val nucleotide-level:  ROC-AUC={nuc_auc:.4f}  PR-AUC={nuc_ap:.4f}"
            )

        if "seq_probs" in preds:
            seq_probs = preds["seq_probs"]
            seq_labels = preds["seq_labels"]

            seq_auc = BinaryAUROC()(seq_probs, seq_labels).item()
            seq_ap = BinaryAveragePrecision()(seq_probs, seq_labels).item()
            logger.info(
                f"Val sequence-level:   ROC-AUC={seq_auc:.4f}  PR-AUC={seq_ap:.4f}"
            )

            best_threshold, best_f1 = self._select_best_threshold(seq_probs, seq_labels)

            accuracy, precision, recall, f1 = self._metrics_at_threshold(
                seq_probs, seq_labels, best_threshold
            )
            logger.info(
                f"Seq-level threshold (pr_intersection): {best_threshold:.3f}  "
                f"F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}"
            )
        else:
            logger.warning(
                "No seq_logit available, falling back to nucleotide-level evaluation"
            )
            if not compute_nuc_metrics:
                raise RuntimeError(
                    "No `seq_logit`/sequence probs available, and nucleotide metrics are disabled. "
                    "Enable `model` sequence head (`seq_logit`) or set trainer.compute_nuc_metrics=true."
                )

            seq_auc = nuc_auc
            seq_ap = nuc_ap

            nuc_probs = preds["nuc_probs"]
            nuc_labels = preds["nuc_labels"]
            best_threshold, best_f1 = self._select_best_threshold(nuc_probs, nuc_labels)

            accuracy, precision, recall, f1 = self._metrics_at_threshold(
                nuc_probs, nuc_labels, best_threshold
            )
            logger.info(
                f"Nuc-level threshold: {best_threshold:.3f}  "
                f"F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}"
            )

        if self.writer is not None:
            self.writer.experiment.log_metric(
                "val/optimal_threshold", best_threshold, step=epoch
            )
            self.writer.experiment.log_metric("val/seq_roc_auc", seq_auc, step=epoch)
            self.writer.experiment.log_metric("val/seq_pr_auc", seq_ap, step=epoch)
            if compute_nuc_metrics:
                self.writer.experiment.log_metric("val/nuc_roc_auc", nuc_auc, step=epoch)
                self.writer.experiment.log_metric("val/nuc_pr_auc", nuc_ap, step=epoch)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": seq_auc,
            "avg_precision": seq_ap,
            "nuc_auc": nuc_auc,
            "nuc_ap": nuc_ap,
            "threshold": best_threshold,
        }

    def _test_epoch(self):
        from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

        compute_nuc_metrics = bool(
            self.config.trainer.get("compute_nuc_metrics", False)
        )
        if self.tta_enabled:
            preds = self._collect_predictions_with_tta(
                self.test_dataloader, "Test", epoch=0, collect_nuc=compute_nuc_metrics
            )
        else:
            preds = self._collect_predictions(
                self.test_dataloader, "Test", epoch=0, collect_nuc=compute_nuc_metrics
            )

        avg_loss = preds["avg_loss"]
        nuc_auc = float("nan")
        nuc_ap = float("nan")
        best_threshold = float("nan")
        best_f1 = float("nan")

        if compute_nuc_metrics and "nuc_probs" in preds:
            nuc_probs = preds["nuc_probs"]
            nuc_labels = preds["nuc_labels"]
            nuc_auc = BinaryAUROC()(nuc_probs, nuc_labels).item()
            nuc_ap = BinaryAveragePrecision()(nuc_probs, nuc_labels).item()
            logger.info(
                f"Test nucleotide-level:  ROC-AUC={nuc_auc:.4f}  PR-AUC={nuc_ap:.4f}"
            )

        if "seq_probs" in preds:
            seq_probs = preds["seq_probs"]
            seq_labels = preds["seq_labels"]

            seq_auc = BinaryAUROC()(seq_probs, seq_labels).item()
            seq_ap = BinaryAveragePrecision()(seq_probs, seq_labels).item()
            logger.info(
                f"Test sequence-level:   ROC-AUC={seq_auc:.4f}  PR-AUC={seq_ap:.4f}"
            )

            best_threshold, best_f1 = self._select_best_threshold(seq_probs, seq_labels)
            logger.info(
                f"Test seq-level best threshold: {best_threshold:.3f} (F1={best_f1:.4f})"
            )

            accuracy, precision, recall, f1 = self._metrics_at_threshold(
                seq_probs, seq_labels, best_threshold
            )
        else:
            if not compute_nuc_metrics:
                raise RuntimeError(
                    "No `seq_logit`/sequence probs available, and nucleotide metrics are disabled. "
                    "Enable `model` sequence head (`seq_logit`) or set trainer.compute_nuc_metrics=true."
                )
            nuc_probs = preds["nuc_probs"]
            nuc_labels = preds["nuc_labels"]
            seq_auc = nuc_auc
            seq_ap = nuc_ap

            best_threshold, best_f1 = self._select_best_threshold(nuc_probs, nuc_labels)
            logger.info(
                f"Test nuc-level best threshold: {best_threshold:.3f} (F1={best_f1:.4f})"
            )

            accuracy, precision, recall, f1 = self._metrics_at_threshold(
                nuc_probs, nuc_labels, best_threshold
            )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": seq_auc,
            "avg_precision": seq_ap,
            "nuc_auc": nuc_auc,
            "nuc_ap": nuc_ap,
            "threshold": float(best_threshold),
        }

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "best_threshold": self.best_threshold,
            "config": self.config,
            "data_indices": self.data_indices,
        }

        if is_best:
            filename = self.save_dir / "model_best.pth"
        else:
            filename = self.save_dir / f"checkpoint-epoch{epoch}.pth"

        torch.save(state, filename)
        logger.info(f"Checkpoint saved: {filename}")

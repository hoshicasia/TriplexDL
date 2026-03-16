import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.

    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["sequence"] = torch.stack([item["sequence"] for item in dataset_items])
    result_batch["omics_features"] = torch.stack(
        [item["omics_features"] for item in dataset_items]
    )

    labels = [item["label"] for item in dataset_items]
    if isinstance(labels[0], torch.Tensor) and labels[0].dim() > 0:
        result_batch["label"] = torch.stack(labels)
    else:
        result_batch["label"] = torch.tensor(labels)

    if "mask" in dataset_items[0]:
        result_batch["mask"] = torch.stack([item["mask"] for item in dataset_items])

    if "chrom" in dataset_items[0]:
        result_batch["chrom"] = [item["chrom"] for item in dataset_items]

    if "chrom_id" in dataset_items[0]:
        result_batch["chrom_id"] = torch.stack(
            [item["chrom_id"] for item in dataset_items]
        )

    if "sample_weight" in dataset_items[0]:
        result_batch["sample_weight"] = torch.stack(
            [item["sample_weight"] for item in dataset_items]
        )

    return result_batch

# TPX

## Setup

## Installation

```bash
git clone https://github.com/hoshicasia/TriplexNet.git
cd TriplexDL
python3 -m venv .venv
source .venv/bin/activate   
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Project structure

- **Training**: `train.py`
- **Config**: `src/configs/baseline.yaml`
- **Data**: `grummit/triplexDNA_pos.fa`, `grummit/triplexDNA_neg.fa`, `Neural/` (BED files)
- **Checkpoints**: `saved/`

## Quick start

### Single run

```bash
python train.py
```

### Override parameters

```bash
python train.py k_fold=3
python train.py datasets.train.pos_fasta_path=grummit/triplexDNA_pos.fa \
  datasets.train.neg_fasta_path=grummit/triplexDNA_neg.fa \
  datasets.train.bed_dir=Neural
```


## Inference

Script: [scripts/score_regions.py](scripts/score_regions.py)

Requirements:
- FASTA with headers like `>id:any:chr1:12345:12445:+` (coordinates in header)
- BED files for omics features
- Checkpoint `model_best.pth`

```bash
python scripts/score_regions.py \
  --fasta /path/to/candidates.fa \
  --bed-dir /path/to/Neural \
  --checkpoint saved/<run>/model_best.pth \
  --out saved/<run>/inference/preds.tsv \
  --nuc-out saved/<run>/inference/preds.nuc.bedgraph \
  --batch-size 8
```

Use `--threshold` to override the checkpoint threshold. Omit `--nuc-out` for sequence-level only.

## Key parameters

### Splitting

- `split_method`: `random`, `genomic_bin_representative`
- `bin_size`: genomic bin size (bp)
- `k_fold`: number of folds (0 = single split)
- `target_class_ratio`: neg:pos ratio in train

### Dataset and augmentation

- `datasets.train.max_seq_len`: max sequence length
- `datasets.train.rc_augment`: reverse-complement augmentation
- `datasets.train.nuc_mask_prob`: nucleotide masking probability
- `datasets.train.coord_shift_max`: random coordinate shift for omics
- `datasets.train.omics_feature_mode`: `coverage`, `score_mean`, `coverage_score`

### Model and optimization

- `model.n_channels`, `model.n_dilated_blocks`, `model.dropout`, `model.aux_loss_weight`
- `loss_function.top_k_ratio`, `loss_function.pos_weight`, `loss_function.label_smoothing`
- `optimizer.lr`, `optimizer.weight_decay`
- `lr_scheduler.T_max`, `warmup.enabled`, `warmup.warmup_epochs`

### Hard negative mining

- `hard_neg_mining_freq`: epoch interval for mining (0 = disabled)
- `hard_neg_ratio`: fraction of hard negatives to sample

### Logging

In this project, CometML is used for experiment tracking.To use it, you need to set your API key before running:

```bash
export COMET_API_KEY=YOUR_API_KEY
```

- `writer.run_name`, `writer.experiment_name`
- `trainer.save_dir`, `trainer.save_period`, `trainer.early_stop`

## Outputs

- Best checkpoint: `saved/<run_name>/model_best.pth`
- Periodic checkpoints: `saved/<run_name>/checkpoint-epochN.pth`

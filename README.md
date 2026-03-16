# TPX

## Установка и настройка проекта

## Установка

```bash
git clone https://github.com/hoshicasia/TriplexNet.git
cd TPX
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Основные файлы

- Обучение: `train.py`
- Конфиги: `src/configs/`
- Датасет (пример): `grummit/triplexDNA_pos.fa`, `grummit/triplexDNA_neg.fa`, `Neural/`
- Сохранения: `saved/`

## Быстрый старт

### Один запуск обучения

```bash
python train.py -cn=grummit_triplexnet_improved
```

### K-fold кросс-валидация

```bash
python train.py -cn=grummit_triplexnet_improved k_fold=3
```

**Note**: в то время как кросс-валидация помогает получить менее смещенные метрики,
количество данных для обучения уменьшается, что может приводить к худшем обучению

### Как обучить на своих данных:

```bash
python train.py -cn=grummit_triplexnet_improved \
  datasets.train.pos_fasta_path=grummit/triplexDNA_pos.fa \
  datasets.train.neg_fasta_path=grummit/triplexDNA_neg.fa \
  datasets.train.bed_dir=Neural
```

## Инференс: скоринг своих регионов (тестировалось только на маленьких синтетических данных, могут быть баги)

Скрипт для применения обученной модели: [scripts/score_regions.py](scripts/score_regions.py)

Требования к входу:
- FASTA с хедерами вида `>id:any:chr1:12345:12445:+` (координаты обязаны быть внутри хедера)
- Набор омиксов в формате .BED
- Чекпоинт вида `model_best.pth`

Пример запуска:
```bash
python scripts/score_regions.py \
  --fasta /abs/path/your_candidates.fa \
  --bed-dir /abs/path/Neural \
  --checkpoint saved/<run>/model_best.pth \
  --out saved/<run>/inference/preds.tsv \
  --nuc-out saved/<run>/inference/preds.nuc.bedgraph \
  --batch-size 8
```

По умолчанию используется `best_threshold` из чекпоинта, но можно задать его как `--threshold` явно. Если нужно только sequence-level предсказание, опцию `--nuc-out` можно не указывать.

## Важные параметры

### Разбиение и валидация

- `split_method`:  `genomic_bin_representative`, `chromosome` (первый рекомендуемый)
- `k_fold`: число фолдов (`0` отключает CV)
- `bin_size`: размер геномного бина для bin-based split

### Данные и аугментации

- `datasets.train.max_seq_len`: ограничение длины последовательности
- `datasets.train.rc_augment`: reverse-complement аугментация
- `datasets.train.nuc_mask_prob`: вероятность маскирования нуклеотидов
- `datasets.train.coord_shift_max`: случайный сдвиг координат
- `datasets.train.kmer_max_k`: максимальная длина k-мера для локальных оконных признаков
- `datasets.train.kmer_window_count`: число окон для локальных k-мерных признаков

### Модель, лосс и оптимизация

- `model.n_channels`, `model.n_dilated_blocks`, `model.drop_path_rate`, `model.dropout`
- `model.aux_loss_weight`: вес вспомогательного sequence-level лосса
- `loss_function.nuc_loss_weight`, `loss_function.label_smoothing`, `loss_function.attention_entropy_weight`
- `optimizer.lr`, `optimizer.weight_decay`
- `lr_scheduler.T_max`, `lr_scheduler.eta_min`
- `warmup.enabled`, `warmup.warmup_epochs`, `warmup.warmup_start_factor`

### Баланс классов и hard negatives

- `target_class_ratio`: отношение neg:pos при обучении
- `hard_neg_mining_freq`: частота hard-negative mining в эпохах (`0` отключает)
- `hard_neg_ratio`: доля «сложных» негативов

### Логирование и чекпоинты
**Важно**:
В этом проекте используется CometML для трекинга экспериментов.
Для того, чтобы запустить трекинг, необходимо создать аккаунт comet.com
и перед запуском указать свой API:
export COMET_API_KEY=ВАШ_API

- `writer.run_name`, `writer.experiment_name`, `writer._target_`
- `trainer.save_dir`, `trainer.save_period`, `trainer.early_stop`

## Результаты обучения

- Лучший чекпоинт: `saved/<run_name>/model_best.pth`
- Периодические чекпоинты: `saved/<run_name>/checkpoint-epochN.pth`

## Примечания

- Размеры признаков (`n_omics_features`, `n_kmer_features`) автоматически подстраиваются под созданный датасет.
- Для более стабильной оценки лучше использовать `k_fold`, а не одиночный split.

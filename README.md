# Triplex Prediction Model

## Установка и настройка проекта

### 1)

```bash
git clone https://github.com/hoshicasia/TriplexNet.git
cd TPX
```

### 3) Создание окружения и настройка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Как пользоваться

### 1) Подготовка данных

1. Создайте папку с BED-файлами новой ткани (например, `Liver/`, `Heart/`, `Brain/`).
2. Убедитесь, что формат BED стандартный: `chr`, `start`, `end`, `name`, `score`.
3. FASTA с положительными/отрицательными примерами можно оставить текущими или заменить на свои.

Минимальный запуск:

```bash
python train.py -cn=grummit_triplexnet datasets.train.bed_dir=Liver
```

### 2) Как указывать пути к FASTA и BED


Пример с относительными путями:

```bash
python train.py -cn=grummit_triplexnet \
	datasets.train.pos_fasta_path=grummit/triplexDNA_pos.fa \
	datasets.train.neg_fasta_path=grummit/triplexDNA_neg.fa \
	datasets.train.bed_dir=Neural
```

Пример с абсолютными путями:

```bash
python train.py -cn=grummit_triplexnet \
	datasets.train.pos_fasta_path=/data/my_tissue/pos.fa \
	datasets.train.neg_fasta_path=/data/my_tissue/neg.fa \
	datasets.train.bed_dir=/data/my_tissue/bed
```

### 3) Какие split-методы поддерживаются

В `train.py` используются следующие режимы (`split_method`):

- `genomic_bin` — сплит по геномным бинам (рекомендуется как базовый безопасный вариант).
- `genomic_bin_representative` — более репрезентативный split по бинам с дополнительной стратификацией.
- `chromosome` — сплит по хромосомам (`test_chromosomes`, `val_chromosomes`).

Примеры:

```bash
# Controlled genomic bins
python train.py -cn=grummit_triplexnet split_method=genomic_bin_representative

# Chromosome split
python train.py -cn=grummit_triplexnet split_method=chromosome test_chromosomes=[chr1,chr2] val_chromosomes=[chr3,chr4]
```

### 4) Куда сохраняются модели

Чекпойнты сохраняются в папку:

`<ROOT>/saved/<run_name>/`

Где:

- `<ROOT>` — корень репозитория;
- `saved` — значение `trainer.save_dir`;
- `<run_name>` — значение `writer.run_name`.

Основные файлы:

- `model_best.pth` — лучший чекпойнт по валидационной метрике;
- `checkpoint-epochN.pth` — периодические чекпойнты по эпохам.

Пример запуска:

```bash
python train.py -cn=grummit_triplexnet \
  writer.run_name=liver_noisyor_seed42
```

Тогда лучший чекпойнт будет тут:

`saved/liver_noisyor_seed42/model_best.pth`


### 5) Полный пайплайн экспериментов для одной ткани: подбор модели -> кросс-валидация -> финальный blind test

Рекомендуемый порядок:

1. **Грубый отбор конфигов** на 1 split (быстрые прогоны, меньше эпох).
2. **Стабильность по seed** для топ-2/3 конфигов.
3. **k-fold CV** для выбранного конфига и агрега - подбор лучшей модели по валидации.
4. **Финальный этап**: train на всех train+val данных + 1 оценка на test группе.

Команда для шага 4 (уже поддерживается в `train.py`):

```bash
python train.py -cn=grummit_triplexnet \
	k_fold=4 \
	k_fold_final_stage.enabled=true \
	k_fold_final_stage.blind_group_index=0 \
	k_fold_final_stage.use_global_threshold=true \
	k_fold_final_stage.disable_validation=true
```

Что тут происходит:

- оставляем нетронутой test выборку
- на остальных группах делаем CV и калибруем threshold
- затем обучаем финальную модель на всех non-blind данных
- и только после этого делаем один финальный тест на blind группе.

### 6) Инференс
Важно: текущий `inference.py` делает предсказания по датасету, заданному в `datasets.train`.
Чтобы получить полногеномные предсказания, нужно подать на вход FASTA с окнами по всему геному.

Для корректной работы нужно либо создать пустой negative FASTA для запуска датасета (можно 1 короткую последовательность),
либо подготовить отдельный конфиг датасета под inference-only режим.

Запуск инференса:

```bash
python inference.py -cn=grummit_triplexnet \
	checkpoint_path=saved/liver_noisyor_seed42/model_best.pth \
	datasets.train.pos_fasta_path=hg38_windows_1kb.fa \
	datasets.train.neg_fasta_path=grummit/triplexDNA_neg.fa \
	datasets.train.bed_dir=Neural \
	output_bed=predictions/hg38_triplex.bed
```

Если нужен не только BED с интервалами, но и сырой набор понкулеотидных вероятностей,
добавьте один или оба параметра:

```bash
python inference.py -cn=grummit_triplexnet \
	checkpoint_path=saved/liver_noisyor_seed42/model_best.pth \
	datasets.train.pos_fasta_path=hg38_windows_1kb.fa \
	datasets.train.neg_fasta_path=grummit/triplexDNA_neg.fa \
	datasets.train.bed_dir=Neural \
	output_bed=predictions/hg38_triplex.bed \
	output_nucleotide_bedgraph=predictions/hg38_triplex_prob.bedgraph \
	output_nucleotide_tsv=predictions/hg38_triplex_prob.tsv \
	nucleotide_track_bin_size=1
```

Опции:

- `output_nucleotide_bedgraph` — трек для genome browser
- `output_nucleotide_tsv` — табличный формат (`chrom/start/end/probability`);
- `nucleotide_track_bin_size` — усреднение по бинам (например, `10` уменьшит размер файлов).

Выход:

- `predictions/hg38_triplex_prob.bedgraph` — сырые вероятности по позициям (опционально);
- `predictions/hg38_triplex_prob.tsv` — сырые вероятности в TSV (опционально);


### 7) Как обучать

Рекомендуется запускать несколько сидов (`seed=42,43,44...`) и сравнивать средние метрики и разброс.

Пример:

```bash
python train.py -cn=grummit_triplexnet datasets.train.bed_dir=Liver seed=42
python train.py -cn=grummit_triplexnet datasets.train.bed_dir=Liver seed=43
python train.py -cn=grummit_triplexnet datasets.train.bed_dir=Liver seed=44
```

На практике:
- если метрики сильно прыгают от seed к seed, лучше увеличить число запусков и усреднить.

### 8) Трекинг экспериментов (CometML)

В данном проекте используется **CometML** для трекинга экспериментов, с помощью него удобно отслеживать прогресс.

Базовый запуск с CometML:

```bash
export COMET_API_KEY=your_api_key #Для этого требуется создать аккаунт на comet.com и вставить сюда ваш API
python train.py -cn=grummit_triplexnet writer._target_=src.logger.comet_writer.CometMLWriter
```

Если CometML не нужен, можно использовать локальный writer из конфига.

## References
- **PyTorch Template**: [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template)

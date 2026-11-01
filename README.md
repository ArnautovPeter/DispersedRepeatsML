# DispersedRepeatsML

Research code for dispersed DNA repeat discovery and family clustering using
Nucleotide Transformer embeddings, supervised contrastive learning, HDBSCAN,
and a complementary canonical 9-mer/profile search pipeline.

**Report:** [English PDF](output/pdf/repeat_discovery_article/article.pdf) ·
[LaTeX source](output/pdf/repeat_discovery_article/article.tex) ·
[Saved comparison tables](output/pdf/repeat_discovery_article/data/).

The extracted-copy benchmark achieved test pair-F1 **0.602** with canonical
4/5-mer features, **0.311** with frozen NT, and **0.260** with the selected
contrastive projection. The separate 9-mer/profile genome-search benchmark
recovered all 150 synthetic copies through mutation parameter x=1.5, with
locus precision 0.962–1.000 at IoU ≥0.5; it returned no loci at x=2.
These are different evaluation tasks, not a single head-to-head comparison.

Source code, configurations, tests, reports, figures and compact saved metrics
are included. Downloaded genomes, pretrained weights, checkpoints, local
runtime diagnostics and full experiment directories are excluded from Git.
The commands below regenerate local artifacts; see the linked reports for
protocols and limitations.

Работающий небольшой эксперимент для поиска групп повторов через эмбеддинги ДНК.
Подготовка данных, обе стадии обучения и оценка реально выполнены на RTX 3050 Laptop.
Результаты и ограничения: [REPORT.md](REPORT.md).

В текущем пилоте контрастивное обучение **не улучшило** кластеризацию неизвестных
семейств: test pair-F1 0,260 против 0,311 у замороженного NT и 0,602 у k-меров.
Это небольшой вычислительный бюджет, а не окончательная проверка гипотезы.

## Воспроизведение

```bash
uv sync --locked --extra dev
uv run --no-sync python -m src.check_gpu
uv run --no-sync python -m pytest -q
uv run --no-sync python -m src.pipeline --output runs/reproduce --with-dfam --with-scan
```

Команды выполняются из корня проекта. Вторая команда проверяет forward/backward
на GPU. В окружении Codex с ограниченным доступом к устройствам нужен запуск
с разрешённым доступом к `/dev/nvidia*`; в обычном терминале это не требуется.
Системный CUDA Toolkit не нужен: PyTorch 2.6.0+cu124 ставит нужные runtime-библиотеки
в `.venv`; сборка закреплена для драйвера NVIDIA 550.

`src.pipeline` использует [configs/pilot.json](configs/pilot.json). Завершённые
запуски не перезаписываются: укажите новую папку `--output`. Raw-данные кэшируются
с SHA256, дата и версия аннотации сохранены в manifest. Модель закреплена на commit
`81b29e5786726d891dbf929404ef20adca5b36f1`. Нужны сеть для первой загрузки и
несколько ГБ свободного места; последующие запуски используют локальные файлы.

## Что сохранено

- `data/pilot/copies.jsonl` — реальные копии, верхний регистр, координаты 0-based,
  half-open, ориентация, `repName`, `repFamily`, покрытие и split; фон помечен отдельно.
- `data/pilot/split.json`, `audit.json`, `manifest.json` — разбиение и аудит.
- `data/raw/` — согласованные hg38 chr22 FASTA и UCSC rmsk со снимками источников.
- `data/model/` — исходный NT с кодом, весами и pinned revision.
- `data/dfam/` — train-only Dfam метаданные и три seed-выравнивания, не входившие в модель.
- `runs/baseline/best.pt` — выбранный по validation checkpoint всего encoder+head.
  В этом запуске выбрана стадия проекции. `projection_best.pt` и `finetune_best.pt`
  сохраняют лучшие веса каждой стадии отдельно.
- `runs/baseline/comparison.tsv`, `*.assignments.tsv`, `*.errors.json` — сравнение,
  присвоения и конкретные ошибки. `training.json`, `*.validation_grid.json`,
  `preflight.json`, `provenance.json` — параметры выбора и диагностика.
- `runs/baseline/{kmers,frozen,finetuned}.npz` — нормированные векторы с `copy_ids`.
- `runs/context/` — отдельное дообучение со сдвигом полной копии в окнах по 800 bp,
  с фоновыми краями; выбранная модель здесь обучала последние два блока.
- `runs/genome_scan/loci.tsv` — координаты, `family_id`, число окон/локусов,
  статус `unvalidated_candidate` и флаг перекрытий между разными кандидатами.
  `annotation_metrics.json` — отдельная проверка координатного этапа.

Ранние `runs/pilot*` и `runs/scan` — отладочные прогоны. Канонические результаты:
`runs/baseline`, `runs/context`, `runs/genome_scan`. Данные, веса и runs исключены
из Git, но физически сохранены в рабочей папке.

## Отдельные шаги

```bash
uv run --no-sync python -m src.download --chrom chr22
uv run --no-sync python -m src.download --model
uv run --no-sync python -m src.data --output data/pilot
uv run --no-sync python -m src.train --output runs/new_baseline \
  --projection-epochs 2 --finetune-epochs 2 --steps-per-epoch 24
uv run --no-sync python -m src.dfam

uv run --no-sync python -m src.train --output runs/new_context \
  --initial-checkpoint runs/new_baseline/best.pt \
  --projection-epochs 2 --finetune-epochs 2 --steps-per-epoch 24 \
  --flank-probability 1 --window-size 800 --evaluation-splits validation
uv run --no-sync python -m src.scan --checkpoint runs/new_context/best.pt \
  --start 20100000 --end 20200000 --output runs/new_scan
uv run --no-sync python -m src.scan_evaluate --scan runs/new_scan
```

Порог FAISS выбирается по парам validation-копий в окнах, resolution Leiden = 1
зафиксирована заранее. Координатная аннотация читается только в последнем шаге
аудита. Поиск отвергает участок, пересекающий обучающие локусы с запасом на контекст.

Для увеличения числа копий и бюджета есть `configs/expanded.json`:

```bash
uv run --no-sync python -m src.pipeline --config configs/expanded.json \
  --data-output data/expanded --output runs/expanded
```

Этот режим **не запускался**. На chr22 не для всех меток найдётся 128 полных
независимых копий; сохраняются доступные при минимуме 16. Он меняет набор и поэтому
является новым экспериментом. Для следующей подтверждающей проверки следует
зафиксировать новый test до дальнейшего подбора гиперпараметров. Поддержка
объединённого многохромосомного набора в этот первый пилот не включена.

## Источники

[UCSC hg38 FASTA](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz),
[UCSC hg38 RepeatMasker](https://api.genome.ucsc.edu/getData/track?genome=hg38;track=rmsk;chrom=chr22;maxItemsOutput=1000000),
[NT-v2 50M](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species),
[Dfam API](https://dfam.org/help/api),
[совместимые сборки PyTorch](https://pytorch.org/get-started/previous-versions/).

Оформление Python-кода поддерживается Ruff (dev-зависимость). Логические этапы
разделены пустыми строками; длинные вызовы и словари оформляются многострочно.

```bash
uv run --no-sync ruff format src tests
uv run --no-sync ruff format --check src tests
```

## Синтетический тест с управляемыми мутациями

```bash
./scripts/run_synthetic_test.sh
```

Один случайный геном 500 000 bp, 150 копий по 300 bp, пять уровней
`x = 0; 0.5; 1; 1.5; 2`, по `round((x/2) * 300)` замен на копию.
Три метода выполняются последовательно; прогресс выводится в терминал и
`runs/synthetic_<дата-время>_UTC/progress.log`. Итог — `summary.tsv`.
Параметры, определения метрик и команды: [SYNTHETIC_TEST.md](SYNTHETIC_TEST.md).

Следующий протокол с поиском кандидатов, Parasail-выравниванием и графом локусов:
[инструкция v2](SYNTHETIC_TEST_V2.md).

```bash
bash scripts/run_synthetic_test_v2.sh --output runs/synthetic_v2
```

Поиск при сильной дивергенции через тройки копий и итеративный частотный профиль:
[описание метода и проверки](PROFILE_METHOD.md). Запуск полного протокола:

```bash
bash scripts/run_synthetic_profiles.sh --output runs/profile_benchmark_v3
```

Проверка делеций/вставок и двух реальных участков hg38 chr22, с отдельным
экспериментальным gapped-уточнением: [результаты и ограничения](ROBUSTNESS_REPORT.md).
Старые v3-исходники и финальный тест seed=43 сохранены. Новый этап принимает
только уже найденные профили и не заменяет первичную инициализацию:

```bash
.venv/bin/python -m src.refine_profiles \
  --fasta genome.fa --background matched_background.fa \
  --history runs/discovery/profile_history.json --output runs/refined_new
```

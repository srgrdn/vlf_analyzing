# Broadband Data Tools

Набор утилит для разбора и визуализации бинарных записей `Broadband_Data_*.bin`, построения спектрограмм и осциллограмм, детектирования всплесков в выбранной полосе, а также прогона изображений через готовую модель классификации вистлеров.

## Что лежит в папке

- `Broadband_Data_2026.03.25_16.31.00.bin` — пример исходного бинарного файла.
- `plot_broadband_spectrogram.py` — построение спектрограмм по одному каналу `ns` или `we`.
- `plot_broadband_oscillogram.py` — построение осциллограммы в `dBFS`.
- `detect_broadband_bursts.py` — детектор всплесков по огибающей мощности в выбранной полосе.
- `run_whistler_classifier.py` — прогон PNG/JPG изображений через обученную модель `whistler_classify_hq.h5`.

## Формат данных

Текущие скрипты рассчитаны на файлы с заголовком `DTLG`, где:

- в заголовке хранится размер записи `60`;
- в заголовке хранится число записей `200000`;
- основное тело читается как `int16 big-endian`;
- предполагается 2 интерливированных канала: `ns` и `we`;
- по умолчанию используется частота дискретизации `100000 Hz` на канал.

Если у конкретного файла другая частота дискретизации или другое число каналов, это можно переопределить аргументами командной строки.

## Требования

Для спектрограмм, осциллограмм и детектора всплесков:

- Python 3
- `numpy`
- `matplotlib`

Для классификатора изображений дополнительно нужны:

- `tensorflow`
- `pillow`

Классификатор рассчитан на ту же модель, что использует ноутбук:

- `../models/whistler_classify_hq.h5`

## Быстрый старт

### 0. Подготовка датасета

Сгенерировать PNG для ручной разметки и `.npz` для будущего обучения:

```bash
python3 build_burst_dataset.py raw_data --overwrite
```

Скрипт создаёт структуру:

```text
dataset/
  review/
    unlabeled/
    burst/
    no_burst/
  samples/
  metadata.csv
```

Для ручной разметки переносите PNG из `dataset/review/unlabeled` в `dataset/review/burst` или `dataset/review/no_burst`.

### 0b. Localization dataset для 2D модели

Собрать `2`-секундные сегменты со спектрограммами, auto-label временем сфериков и review PNG:

```bash
python3 build_localization_dataset.py raw_data \
  --channel both \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --overwrite
```

Скрипт создаёт структуру:

```text
dataset_localization/
  samples/
  review/
    auto/
      ns/
      we/
    verified/
      ns/
      we/
    corrected/
      ns/
      we/
  metadata.csv
```

В каждом `.npz` хранятся:

- `spec_db`
- `envelope_db`
- `time_axis`
- `freq_axis`
- `event_times_s`
- `event_count`
- `target_heatmap`

В `metadata.csv` сохраняются путь к исходному файлу, канал, границы сегмента, число auto-detected событий и JSON-список времен внутри сегмента.

Собрать уменьшенный stratified subset для ручной проверки:

```bash
python3 build_localization_review_subset.py dataset_localization
```

По умолчанию скрипт создаёт:

```text
dataset_localization/
  review_subset/
    ns/
      empty/
      single/
      few/
      dense/
    we/
      empty/
      single/
      few/
      dense/
    manifest.csv
```

`manifest.csv` хранит `sample_id`, канал, bucket, причину отбора и ссылки на PNG/NPZ, чтобы удобно проводить ручной review по репрезентативной подвыборке.

Создать шаблон ручных правок:

```bash
python3 init_localization_corrections.py dataset_localization/review_subset
```

Скрипт создаст `dataset_localization/review_subset/corrections.csv` с колонками:

- `sample_id`
- `auto_event_count`
- `auto_event_times_s_json`
- `review_status`
- `corrected_event_times_s_json`
- `quality_flag`
- `comment`

Рекомендуемые значения `review_status`:

- `manual_verified` — auto-label принят как есть;
- `manual_corrected` — времена/число событий вручную исправлены;
- `artifact_suspected` — сегмент подозрителен и лучше не использовать для обучения;
- `uncertain` — случай спорный, нужен отдельный разбор.

Проверить прогресс ручной разметки и увидеть следующие PNG для review:

```bash
python3 summarize_localization_review.py dataset_localization
```

Скрипт только читает `manifest.csv` и `corrections.csv`: печатает количество проверенных/ожидающих примеров, распределения по статусам, каналам и bucket, а также список следующих pending-сегментов.

Интерактивно разметить subset в окне `matplotlib`:

```bash
python3 review_localization_interactive.py dataset_localization
```

Горячие клавиши:

- клик мышью по спектрограмме — добавить время события внутри 2-секундного сегмента;
- `v` — принять auto-label как `manual_verified`;
- `enter` или `c` — сохранить выбранные кликами времена как `manual_corrected`;
- `backspace` — удалить последний клик;
- `r` — вернуть auto-label времена;
- `x` — очистить выбранные времена;
- `a` — пометить `artifact_suspected`;
- `u` — пометить `uncertain`;
- `s` — пропустить текущий пример;
- `q` — выйти.

Скрипт сохраняет `corrections.csv` после каждого принятого примера, поэтому review можно останавливать и продолжать позже.

Пересмотреть только строки с конкретным статусом:

```bash
python3 review_localization_interactive.py dataset_localization --review-status manual_corrected
```

После ручного заполнения применить правки к `metadata.csv` и `.npz`:

```bash
python3 apply_localization_corrections.py dataset_localization
```

После применения:

- `metadata.csv` обновляет `event_count`, `event_times_s_json`, `label_source`, `quality_flag`;
- соответствующие `.npz` обновляют `event_times_s`, `event_count`, `target_heatmap`;
- review PNG копируются в `review/verified/<channel>/` или `review/corrected/<channel>/`.

### 1. Спектрограмма

Показать спектрограмму для канала `ns`:

```bash
python3 plot_broadband_spectrogram.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --show
```

Построить последовательные спектрограммы по 2 секунды в полосе `20-30 kHz`:

```bash
python3 plot_broadband_spectrogram.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --cmap jet
```

### 2. Осциллограмма

Построить осциллограмму в `dBFS`:

```bash
python3 plot_broadband_oscillogram.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --mode rms \
  --show
```

### 3. Детектор всплесков

Построить по сегментам совмещённый график `спектрограмма + огибающая + порог + срабатывания`:

```bash
python3 detect_broadband_bursts.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --threshold-mad 8 \
  --min-peak-distance 0.05 \
  --cmap jet \
  --show
```

Полезные параметры детектора:

- `--aggregate mean|sum|max` — как сворачивать спектр по частоте.
- `--threshold-mad` — насколько выше фона должен быть пик.
- `--min-peak-distance` — минимальная дистанция между детекциями в секундах.
- `--mark-span` — ширина подсвеченного окна вокруг детекции.
- `--mark-alpha` — прозрачность подсветки.

Скрипт сохраняет:

- PNG по каждому сегменту;
- CSV со всеми детекциями и временами пиков.

### 4. Классификация изображений

Один файл:

```bash
python run_whistler_classifier.py some_image.png --show
```

Директория с изображениями:

```bash
python run_whistler_classifier.py ./images --recursive
```

В режиме директории скрипт:

- прогоняет все найденные изображения;
- сохраняет индивидуальные визуализации;
- строит summary-график;
- сохраняет `predictions.csv`;
- сохраняет `stats.json`.

## Замечания по окружению

- `plot_broadband_spectrogram.py`, `plot_broadband_oscillogram.py` и `detect_broadband_bursts.py` нормально работают в обычном Linux/WSL Python.
- `run_whistler_classifier.py` требует окружение с TensorFlow. Если в текущем интерпретаторе TensorFlow не установлен, скрипт завершится с подсказкой использовать то же окружение, что и ноутбук.

## Полезные идеи для развития

- добавить автоматическое выделение начала и конца события, а не только центра пика;
- добавить экспорт детекций в формат для последующей ручной разметки;
- добавить пакетный прогон бинарных файлов из директории;
- добавить сравнение каналов `ns/we` на одном графике.

# Broadband Data Tools

Набор утилит для разбора и визуализации бинарных записей `Broadband_Data_*.bin`, построения спектрограмм и осциллограмм, детектирования всплесков в выбранной полосе, а также прогона изображений через готовую модель классификации вистлеров.

## Что лежит в папке

- `Broadband_Data_2026.03.25_16.31.00.bin` — пример исходного бинарного файла.
- `plot_broadband_spectrogram.py` — построение спектрограмм по одному каналу `ns` или `we`.
- `plot_broadband_oscillogram.py` — построение осциллограммы в `dBFS`.
- `detect_broadband_bursts.py` — детектор всплесков по огибающей мощности в выбранной полосе.
- `run_envelope_1d_model.py` — разовый прогон `envelope_1d` модели по сырым минутным `.bin`.
- `monitor_envelope_1d_model.py` — непрерывный model-based monitor для продового режима.
- `envelope_model_inference.py` — общий код инференса `envelope_db -> PyTorch model`.
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

Для `envelope_1d` model pipeline дополнительно нужны:

- `torch`

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

### 5. Прогон `envelope_1d` модели по сырым `.bin`

Если уже есть обученный checkpoint, например `models/envelope_1d_dataset_v2.pt`, можно прогнать его по минутным файлам напрямую, без промежуточной ручной разметки.

Скрипт сам:

- читает `Broadband_Data_*.bin`;
- режет минутный файл на окна по `2 s`;
- строит `envelope_db` в полосе `20-30 kHz`;
- подаёт каждое окно в `envelope_1d` модель;
- пишет CSV по сегментам и по минутным файлам.

Пример:

```bash
python3 run_envelope_1d_model.py raw_vlf_data/raw_data \
  --recursive \
  --model models/envelope_1d_dataset_v2.pt \
  --channel ns \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --burst-threshold 0.5
```

Результаты будут в:

```text
reports/model_envelope_1d/segments.csv
reports/model_envelope_1d/summary.csv
```

`segments.csv` — одна строка на 2-секундный сегмент:

- `segment_start`, `segment_end`;
- `predicted_label`;
- `prob_burst`;
- `peak_value_db`.

`summary.csv` — одна строка на минутный файл:

- `segment_count`;
- `burst_segment_count`;
- `max_prob_burst`;
- `route_decision`.

### 6. Непрерывный model-based monitor

Для продового режима есть отдельный monitor, который следит за директорией с новыми минутными файлами и применяет `envelope_1d` модель автоматически.

Логика:

- ждёт, пока файл перестанет расти;
- прогоняет модель по всем 2-секундным сегментам;
- если хотя бы один сегмент классифицирован как `burst`, переносит файл в `processing1/`;
- если ни одного `burst` нет, по умолчанию удаляет файл;
- если нужен безопасный режим, можно включить `--move-empty`, тогда пустые файлы идут в `empty/`.

Пример:

```bash
python3 monitor_envelope_1d_model.py \
  --watch-dir raw_vlf_data/raw_data \
  --model models/envelope_1d_dataset_v2.pt \
  --channel ns \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --burst-threshold 0.5
```

По умолчанию monitor создаёт рядом с `watch-dir`:

```text
processing1/
failed/
reports/model_envelope_1d/
raw_data/.model_monitor/
```

В `reports/model_envelope_1d/` пишутся:

- `segments.csv` — классификация всех 2-секундных окон;
- `summary.csv` — итог по минутным файлам.

В `.model_monitor/` пишутся:

- `monitor_state.json` — уже обработанные файлы и их статус;
- `monitor_events.jsonl` — журнал работы.

## Замечания по окружению

- `plot_broadband_spectrogram.py`, `plot_broadband_oscillogram.py` и `detect_broadband_bursts.py` нормально работают в обычном Linux/WSL Python.
- `run_envelope_1d_model.py` и `monitor_envelope_1d_model.py` требуют `torch` в том интерпретаторе, из которого запускаются.
- `run_whistler_classifier.py` требует окружение с TensorFlow. Если в текущем интерпретаторе TensorFlow не установлен, скрипт завершится с подсказкой использовать то же окружение, что и ноутбук.

## Полезные идеи для развития

- добавить автоматическое выделение начала и конца события, а не только центра пика;
- добавить экспорт детекций в формат для последующей ручной разметки;
- добавить пакетный прогон бинарных файлов из директории;
- добавить сравнение каналов `ns/we` на одном графике.

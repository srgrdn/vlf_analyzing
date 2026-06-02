# Broadband Data Tools

Набор утилит для разбора бинарных записей `Broadband_Data_*.bin`, построения спектрограмм и осциллограмм, детектирования широкополосных всплесков и автоматического мониторинга директории с сырыми минутными файлами.

Основной сценарий проекта сейчас такой:

- в папку `raw_data/` попадают минутные бинарные файлы;
- `monitor_raw_data.py` постоянно следит за этой папкой;
- как только файл перестает расти, монитор запускает `detect_broadband_bursts.py`;
- если всплесков нет, файл по умолчанию удаляется;
- если всплески есть, файл сохраняется.

## Что лежит в репозитории

- `plot_broadband_spectrogram.py` — построение спектрограмм по одному каналу `ns` или `we`.
- `plot_broadband_oscillogram.py` — построение осциллограммы в `dBFS`.
- `detect_broadband_bursts.py` — детектор всплесков по огибающей мощности в выбранной полосе.
- `monitor_raw_data.py` — постоянный монитор директории, который вызывает детектор для новых файлов.
- `run_whistler_classifier.py` — прогон PNG/JPG изображений через обученную модель `whistler_classify_hq.h5`.
- `Broadband_Data_2026.03.25_16.31.00.bin` — пример исходного бинарного файла.

## Формат данных

Текущие скрипты рассчитаны на файлы с заголовком `DTLG`, где:

- в заголовке хранится размер записи `60`;
- в заголовке хранится число записей `200000`;
- основное тело читается как `int16 big-endian`;
- предполагается 2 интерливированных канала: `ns` и `we`;
- по умолчанию используется частота дискретизации `100000 Hz` на канал.

Если у конкретного файла другая частота дискретизации или другое число каналов, это можно переопределить аргументами командной строки.

## Требования

Для спектрограмм, осциллограмм, детектора и монитора:

- Python 3
- `numpy`
- `matplotlib`

Для классификатора изображений дополнительно нужны:

- `tensorflow`
- `pillow`

Классификатор рассчитан на модель:

- `../models/whistler_classify_hq.h5`

## Быстрый старт

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

Ручной запуск с графиками по сегментам:

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

Запуск на весь минутный файл без графического окна:

```bash
python3 detect_broadband_bursts.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01
```

Автоматический режим для монитора, когда не нужны ни PNG, ни CSV, ни GUI:

```bash
python3 detect_broadband_bursts.py Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01 \
  --no-save \
  --allow-no-output
```

В конце автоматического запуска скрипт печатает строку вида:

```text
TOTAL_DETECTIONS=3
```

И дополнительно печатает машинно-читаемую сводку:

```text
DETECTION_SUMMARY_JSON={"channel": "ns", "total_detections": 3, ...}
```

Именно эту JSON-сводку использует монитор.

Полезные параметры детектора:

- `--aggregate mean|sum|max` — как сворачивать спектр по частоте.
- `--threshold-mad` — насколько выше фона должен быть пик.
- `--min-peak-distance` — минимальная дистанция между детекциями в секундах.
- `--mark-span` — ширина подсвеченного окна вокруг детекции.
- `--mark-alpha` — прозрачность подсветки.
- `--no-save --allow-no-output` — режим для автоматизации без файлового и графического вывода.

Если `--no-save` не указан, детектор сохраняет:

- PNG по каждому сегменту;
- CSV со всеми детекциями и временами пиков.

## Монитор и детектор

### Что делает монитор

`monitor_raw_data.py` — это постоянный Python-процесс, который:

- следит за директорией с минутными файлами;
- ищет файлы по шаблону `Broadband_Data_*.bin`;
- ждет, пока размер файла перестанет меняться;
- запускает `detect_broadband_bursts.py` два раза для каждого файла: по `ns` и по `we`;
- читает JSON-сводку по каждому каналу;
- пишет таблицу по каналам и сводную таблицу по минутным файлам;
- при нуле детекций по обоим каналам по умолчанию удаляет файл;
- если детекции есть, переносит файл в `processing1/` и копирует в `processing2/`;
- при ошибке переносит файл в `failed/`;
- ведет лог и сохраняет состояние.

### Как монитор определяет, что файл готов

Монитор не хватает файл сразу после появления. Он запоминает:

- размер файла;
- время модификации;
- число циклов, в течение которых файл не менялся.

По умолчанию файл считается готовым, если он не менялся `2` цикла подряд, а интервал между циклами `10` секунд. То есть по умолчанию файл должен быть стабильным примерно `20` секунд.

Это настраивается параметрами:

- `--poll-interval`
- `--stable-cycles`

### Что монитор создает на диске

По умолчанию рядом с `watch-dir` он создает:

- `raw_data/.monitor/monitor_state.json` — состояние уже обработанных и наблюдаемых файлов;
- `raw_data/.monitor/monitor.log` — текстовый лог;
- `reports/sferics_per_channel.csv` — одна строка на файл и канал;
- `reports/sferics_summary.csv` — одна строка на минутный файл;
- `processing1/` — непустые файлы для пайплайна сфериков;
- `processing2/` — копии непустых файлов для пайплайна вистлеров;
- `failed/` — файлы, по которым детектор завершился с ошибкой.

Если нужен старый режим с сохранением пустых файлов, включи `--move-empty`: тогда дополнительно будет использоваться каталог `empty/`.

Каталоги `processing1/`, `processing2/`, `failed/` и `reports/` по умолчанию создаются рядом с `raw_data/`, то есть как соседние директории. Каталог `empty/` создается только при `--move-empty`.

### Жизненный цикл файла

Типичный сценарий такой:

1. В `raw_data/` появляется `Broadband_Data_2026.03.25_16.31.00.bin`.
2. Монитор видит файл, но не обрабатывает его, пока он растет.
3. Когда файл стабилен, монитор запускает детектор для `ns` и `we`.
4. Детектор анализирует весь минутный файл в полосе `20-30 кГц`.
5. Если по обоим каналам `TOTAL_DETECTIONS=0`:
   файл по умолчанию удаляется.
6. Если хотя бы по одному каналу детекции есть:
   файл переносится в `processing1/` и копируется в `processing2/`.
7. Если детектор падает:
   файл переносится в `failed/`.

### Почему в автоматическом режиме не нужен `--show`

Параметр `--show` открывает окно `matplotlib`. Для постоянного мониторинга это неудобно:

- будут всплывать окна;
- процесс может ждать GUI;
- это мешает фоновому режиму на Windows.

Поэтому монитор всегда запускает детектор в безоконном режиме:

- `--no-save`
- `--allow-no-output`

## Примеры запуска монитора

### 1. Базовый режим

Монитор:

- анализирует оба канала `ns` и `we`;
- пишет CSV-отчеты в `reports/`;
- удаляет пустые файлы;
- переносит непустые файлы в `processing1/`;
- копирует непустые файлы в `processing2/`.

```bash
python3 monitor_raw_data.py \
  --watch-dir raw_data \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01
```

### 2. Сохранять пустые файлы в `empty/`

```bash
python3 monitor_raw_data.py \
  --watch-dir raw_data \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01 \
  --move-empty
```

Этот режим отключает дефолтное удаление и складывает пустые файлы в `empty/`.

### 3. Поменять директории для пайплайнов сфериков и вистлеров

```bash
python3 monitor_raw_data.py \
  --watch-dir raw_data \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01 \
  --processing1-dir D:/data/processing1 \
  --processing2-dir D:/data/processing2
```

Тогда:

- непустые файлы переедут в `D:/data/processing1`;
- их копии появятся в `D:/data/processing2`;
- пустые по умолчанию будут удаляться;
- ошибки уйдут в `failed/`.

### 4. Медленнее сканировать директорию

```bash
python3 monitor_raw_data.py \
  --watch-dir raw_data \
  --poll-interval 30 \
  --stable-cycles 2 \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01
```

Такой режим дает меньшую фоновую нагрузку: сканирование идет раз в `30` секунд.

### 5. Ждать дольше, пока файл допишется

```bash
python3 monitor_raw_data.py \
  --watch-dir raw_data \
  --poll-interval 10 \
  --stable-cycles 4 \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 60 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --mark-span 0.01
```

Здесь файл должен быть неизменным примерно `40` секунд, прежде чем начнется анализ.

## Пример для Windows 7

Если Python установлен и доступен как `python`, базовая команда будет такой:

```bat
python monitor_raw_data.py --watch-dir raw_data --freq-min 20000 --freq-max 30000 --segment-duration 60 --time-resolution 0.002 --frequency-resolution 50 --threshold-mad 8 --mark-span 0.01
```

Если нужен режим с сохранением пустых файлов в `empty`:

```bat
python monitor_raw_data.py --watch-dir raw_data --freq-min 20000 --freq-max 30000 --segment-duration 60 --time-resolution 0.002 --frequency-resolution 50 --threshold-mad 8 --mark-span 0.01 --move-empty
```

Пример простого `.bat`-файла:

```bat
@echo off
cd /d C:\path\to\vlf_analyzing
python monitor_raw_data.py --watch-dir raw_data --freq-min 20000 --freq-max 30000 --segment-duration 60 --time-resolution 0.002 --frequency-resolution 50 --threshold-mad 8 --mark-span 0.01
pause
```

Если монитор должен работать долго, лучше запускать его в отдельном окне `cmd` или через ярлык на этот `.bat`.

## Что смотреть при отладке

Если монитор не обрабатывает файлы, в первую очередь проверь:

- появляется ли файл в `watch-dir`;
- перестает ли он реально меняться по размеру;
- нет ли его уже в `monitor_state.json`;
- появились ли строки в `reports/sferics_per_channel.csv` и `reports/sferics_summary.csv`;
- не уехал ли он в `failed/`;
- что написано в `raw_data/.monitor/monitor.log`.

Если детектор работает вручную, но монитор отправляет файл в `failed/`, полезно вручную выполнить ту же команду, что запускает монитор, только в консоли.

## Классификация изображений

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

- `plot_broadband_spectrogram.py`, `plot_broadband_oscillogram.py`, `detect_broadband_bursts.py` и `monitor_raw_data.py` работают в обычном Python без TensorFlow.
- `run_whistler_classifier.py` требует окружение с TensorFlow.
- Для автоматического мониторинга лучше не использовать `--show`.

## Полезные идеи для развития

- сохранить для монитора отдельный JSON-отчет по каждому файлу;
- добавить опцию повторной обработки файлов из `failed/`;
- добавить режим пакетного прогона без постоянного цикла;
- добавить экспорт детекций в формат для последующей ручной разметки;
- добавить сравнение каналов `ns/we` на одном графике.

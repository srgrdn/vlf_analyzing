# Experiment Config

Этот файл фиксирует базовые параметры эксперимента для подготовки датасета и обучения моделей:

- `1D CNN` по огибающей;
- `2D CNN` по спектрограмме.

Все дальнейшие скрипты подготовки данных, разметки, обучения и сравнения должны использовать эти параметры как стартовые значения.

## Raw Data

Сырые `.bin` файлы складываются в директорию:

```text
raw_data/
```

Пример файла:

```text
raw_data/Broadband_Data_2026.03.25_16.31.00.bin
```

## Base Visualization Command

Команда, по которой зафиксированы рабочие параметры спектрограммы:

```bash
python plot_broadband_spectrogram.py raw_data/Broadband_Data_2026.03.25_16.31.00.bin \
  --channel ns \
  --show \
  --cmap jet \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --no-save
```

## Signal Parameters

```text
channel = ns
sample_rate = 100000 Hz
channels = 2
segment_duration = 2 s
time_resolution = 0.002 s
frequency_resolution = 50 Hz
freq_min = 20000 Hz
freq_max = 30000 Hz
cmap = jet
```

## Derived Shapes

При текущих параметрах:

```text
segment_duration / time_resolution = 2 / 0.002 = 1000 time steps
```

Для полосы `20000-30000 Hz` с шагом `50 Hz`:

```text
(30000 - 20000) / 50 + 1 = 201 frequency bins
```

Ожидаемые формы обучающих примеров:

```text
1D envelope:       примерно 1000 точек
2D spectrogram:    примерно 201 x 1000
```

Для обучения `2D CNN` матрицу спектрограммы можно будет ресайзить до меньшего размера, например:

```text
128 x 256
224 x 224
```

Исходную матрицу всё равно лучше сохранять в `.npz`, чтобы не терять данные до этапа обучения.

## Dataset Output

Планируемая директория датасета:

```text
dataset/
  samples/
  metadata.csv
```

Каждый `.npz` пример должен содержать:

```text
spec_db
envelope_db
time_axis
freq_axis
label
peak_time
```

Минимальные поля `metadata.csv`:

```csv
sample_id,source_file,channel,segment_start,segment_end,freq_min,freq_max,label,peak_time,split,label_source
```

## Labels

Базовые классы:

```text
burst
no_burst
```

Черновую разметку можно получать пороговым детектором:

```bash
python detect_broadband_bursts.py raw_data \
  --channel ns \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2
```

После автогенерации меток спорные сегменты нужно проверять вручную.

## Current Stage

```text
Stage 0: parameters fixed
Next stage: build_burst_dataset.py
```

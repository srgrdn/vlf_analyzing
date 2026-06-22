# Базовая обработка Broadband/VLF сигналов без нейросети

## 1. Назначение

Этот документ описывает базовый pipeline обработки `Broadband_Data_*.bin` файлов без использования нейросетей:

- чтение бинарного файла;
- извлечение каналов `ns` и `we`;
- построение осциллограммы;
- построение спектрограммы;
- разрезание записи на 2-секундные фрагменты;
- сохранение диагностических PNG.

Эти шаги используются как самостоятельный инструмент анализа и как основа для подготовки данных нейросетевой модели локализации сфериков.

## 2. Формат входного файла

Файлы имеют вид:

```text
Broadband_Data_YYYY.MM.DD_HH.MM.SS.bin
```

Чтение реализовано в:

```text
plot_broadband_spectrogram.py
```

Основные константы:

```python
SIGNATURE = b"DTLG"
RECORD_SIZE_OFFSET = 0x08
SAMPLE_COUNT_OFFSET = 0x0242
DEFAULT_SAMPLE_RATE = 100_000.0
CHANNEL_NAME_TO_INDEX = {"ns": 0, "we": 1}
```

Ожидаемая структура:

- в начале файла есть заголовок с сигнатурой `DTLG`;
- в заголовке хранится размер record;
- в заголовке хранится число records;
- payload после заголовка содержит `int16 big-endian`;
- каналы внутри record интерливированы;
- по умолчанию используется 2 канала:
  - `ns`;
  - `we`;
- частота дискретизации на один канал: `100000 Hz`.

## 3. Чтение metadata

Функция:

```python
read_metadata(path)
```

Она:

1. читает минимально нужную часть заголовка;
2. проверяет сигнатуру `DTLG`;
3. извлекает `record_size`;
4. извлекает `sample_count`;
5. вычисляет размер payload:

```text
payload_bytes = sample_count * record_size * 2
```

`2` здесь означает размер одного `int16` в байтах.

Затем вычисляется смещение начала данных:

```text
data_offset = file_size - payload_bytes
```

Это удобно, потому что даже если заголовок содержит дополнительные поля, начало payload можно найти через размер файла и ожидаемый размер данных.

## 4. Загрузка samples

Функция:

```python
load_samples(path, sample_count, record_size, data_offset)
```

Основная идея:

```python
raw = np.memmap(path, dtype=">i2", mode="r", offset=data_offset, shape=(total_values,))
samples = np.asarray(raw, dtype=np.float32)
```

Где:

- `">i2"` - big-endian signed int16;
- `memmap` позволяет читать большие файлы без ручного копирования всего файла в промежуточные структуры;
- дальше данные переводятся в `float32`, чтобы безопасно выполнять спектральные и амплитудные операции.

## 5. Извлечение канала

Функция:

```python
extract_channel(samples, sample_count, record_size, channels, channel_index)
```

Данные сначала преобразуются в матрицу:

```text
[sample_count, record_size]
```

Затем выбирается каждый `channels`-й элемент с нужным смещением:

```python
rows[:, channel_index::channels]
```

Для двух каналов:

- `channel_index = 0` соответствует `ns`;
- `channel_index = 1` соответствует `we`.

После выбора канал разворачивается в один временной ряд:

```python
channel_samples = rows[:, channel_index::channels].reshape(-1)
```

## 6. Осциллограмма

Скрипт:

```text
plot_broadband_oscillogram.py
```

Он строит не сырую миллионы-точечную волну, а амплитудную огибающую во времени.

Команда:

```bash
python3 plot_broadband_oscillogram.py Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns \
  --output output.png
```

Основная функция:

```python
compute_envelope_db(samples, sample_rate, time_resolution, mode, reference)
```

Сигнал делится на окна длиной:

```text
step = round(time_resolution * sample_rate)
```

По умолчанию:

```text
time_resolution = 0.01 s
sample_rate = 100000 Hz
step = 1000 samples
```

В каждом окне считается уровень:

- `rms` - среднеквадратичный уровень;
- `peak` - максимум абсолютной амплитуды.

Затем уровень переводится в dBFS:

```text
envelope_db = 20 * log10(level / reference)
```

По умолчанию:

```text
reference = 32768
```

То есть шкала показывает амплитуду относительно полного диапазона `int16`.

## 7. Спектрограмма

Скрипт:

```text
plot_broadband_spectrogram.py
```

Команда:

```bash
python3 plot_broadband_spectrogram.py Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns \
  --cmap jet \
  --output output.png
```

Основная функция:

```python
compute_spectrogram(samples, nperseg, hop)
```

Алгоритм:

1. временной ряд режется на перекрывающиеся/последовательные frames;
2. из каждого frame вычитается среднее;
3. применяется окно Ханна;
4. считается `rfft`;
5. берется мощность спектра;
6. мощность нормируется на энергию окна;
7. результат переводится в dB.

Ключевой фрагмент:

```python
centered = frames - frames.mean(axis=1, keepdims=True)
spectrum = np.fft.rfft(centered * window, axis=1)
power = abs(spectrum) ** 2
spec_db = 10 * log10(max(power, 1e-12))
```

Выход:

```text
spec_db: [freq_bins, time_bins]
time_axis
freq_axis
```

## 8. Разрешение по времени и частоте

Параметры задаются через:

```bash
--time-resolution
--frequency-resolution
```

Частотное разрешение управляет длиной FFT:

```text
nperseg = round(sample_rate / frequency_resolution)
```

Например:

```text
sample_rate = 100000 Hz
frequency_resolution = 50 Hz
nperseg = 2000 samples
```

Временное разрешение управляет шагом между спектральными окнами:

```text
hop = round(time_resolution * sample_rate)
```

Например:

```text
time_resolution = 0.002 s
hop = 200 samples
```

## 9. Частотная полоса

Для выбора полосы используется:

```python
crop_frequency_band(spec_db, freq_axis, sample_rate, freq_min, freq_max)
```

В проекте для локализации сфериков обычно используется:

```text
20-30 kHz
```

Команда:

```bash
--freq-min 20000 --freq-max 30000
```

Полный диапазон при `sample_rate = 100000 Hz`:

```text
0-50000 Hz
```

Потому что частота Найквиста равна половине частоты дискретизации.

## 10. Разрезание на 2-секундные сегменты

Функция:

```python
iter_time_segments(samples, sample_rate, segment_duration, nperseg)
```

При:

```text
segment_duration = 2 s
sample_rate = 100000 Hz
```

размер одного сегмента:

```text
segment_samples = 200000 samples
```

Для минутного файла получается:

```text
60 s / 2 s = 30 segments
```

Сегменты не перекрываются:

```text
0-2 s
2-4 s
4-6 s
...
58-60 s
```

## 11. Отрисовка спектрограммы

Функция:

```python
render_spectrogram(...)
```

Визуализация строится через `imshow`:

```python
ax.imshow(
    spec_db,
    origin="lower",
    aspect="auto",
    extent=(time_start, time_end, freq_min, freq_max),
)
```

Для устойчивой картинки используется percentile clipping:

```text
vmin = percentile(spec_db, db_low)
vmax = percentile(spec_db, db_high)
```

По умолчанию:

```text
db_low = 2
db_high = 99.8
```

Это не меняет данные, а только улучшает контраст изображения.

## 12. Что считается признаком сферика на спектрограмме

В выбранной полосе `20-30 kHz` сферик обычно выглядит как короткая вертикальная broadband-структура:

- узкая по времени;
- занимает заметную часть частотной полосы;
- часто выглядит как яркая вертикальная линия;
- может пересекать постоянные горизонтальные помеховые линии.

Горизонтальные линии на спектрограмме обычно означают устойчивые узкополосные компоненты/помехи. Они не являются сфериками сами по себе, потому что не локализованы во времени как короткий broadband-всплеск.

## 13. Команды, использованные для нового файла

Новый файл:

```text
/home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin
```

Артефакты сохранены в:

```text
reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/
```

Структура:

```text
full/
  spectrograms/
  oscillograms/
segments_2s_20_30khz/
  spectrograms/
    ns/
    we/
segments_2s/
  oscillograms/
    ns/
    we/
```

Полные спектрограммы показывают весь файл по времени. Сегментные спектрограммы сохранены в полосе `20-30 kHz`, потому что именно эта полоса используется в localization pipeline.

## 14. Связь с нейросетевым методом

Нейросеть не заменяет базовую обработку. Она работает поверх нее.

Базовый pipeline:

```text
raw bin -> channel samples -> spectrogram
```

Нейросетевой pipeline:

```text
spectrogram -> CNN -> predicted heatmap -> event times
```

То есть чтение binary data, выделение каналов, STFT, dB-преобразование и частотный crop остаются классической обработкой сигнала. Нейросеть появляется только после получения `spec_db`.

## 15. Краткое резюме

Без нейросети проект умеет:

- читать `Broadband_Data_*.bin`;
- проверять заголовок и вычислять начало payload;
- извлекать каналы `ns` и `we`;
- строить амплитудную огибающую в dBFS;
- строить спектрограмму через оконное FFT;
- выбирать частотную полосу;
- резать запись на 2-секундные окна;
- сохранять full и segmented PNG для ручного анализа.

Эта часть является фундаментом для всех последующих detector/model workflows.

# Пороговый детектор широкополосных всплесков

## 1. Назначение

Пороговый детектор - это базовый алгоритм поиска коротких широкополосных всплесков в VLF/Broadband данных без нейросети.

Он используется в проекте в двух ролях:

1. как самостоятельный detector baseline;
2. как источник первичных auto-labels для `dataset_localization`.

Главная идея:

```text
raw signal
  -> spectrogram power в выбранной полосе
  -> временная envelope-кривая мощности
  -> robust threshold
  -> локальные пики выше порога
  -> времена событий
```

Для localization pipeline эти события затем становятся:

```text
event_times_s
event_count
target_heatmap
label_source = threshold_auto
```

## 2. Основные файлы

Самостоятельный детектор:

```text
detect_broadband_bursts.py
```

Использование логики детектора при сборе localization dataset:

```text
build_localization_dataset.py
```

Использование логики детектора для pseudo-labeling:

```text
methods/threshold_pseudo_labeling/prepare_threshold_dataset.py
```

## 3. Отличие от нейросетевой модели

Пороговый детектор не обучается.

Он не имеет весов, epochs, loss function или train/test split. Все его поведение задается параметрами обработки сигнала:

- частотная полоса;
- временное и частотное разрешение спектрограммы;
- способ агрегации мощности по частоте;
- сглаживание;
- множитель MAD-порога;
- минимальное расстояние между пиками.

Нейросетевая модель учится по ручной разметке. Пороговый детектор просто применяет фиксированное правило к каждому сегменту.

## 4. Входные данные

Вход - файл или директория с:

```text
Broadband_Data_*.bin
```

Чтение raw-файлов выполняется теми же низкоуровневыми функциями, что и для спектрограмм:

```text
plot_broadband_spectrogram.py
```

Используются:

- `read_metadata`;
- `load_samples`;
- `extract_channel`;
- `resolve_stft_parameters`;
- `resolve_frequency_band`;
- `iter_time_segments`;
- `crop_frequency_band`.

По умолчанию:

- sample rate: `100000 Hz`;
- channels: `2`;
- channel: `ns`;
- доступные каналы:
  - `ns`;
  - `we`.

## 5. Команда запуска самостоятельного детектора

Пример для одного файла:

```bash
python3 detect_broadband_bursts.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --output reports/threshold_detector/example_ns_burst_detection.png
```

Для директории:

```bash
python3 detect_broadband_bursts.py raw_data \
  --recursive \
  --channel ns \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --threshold-mad 8 \
  --output reports/threshold_detector
```

## 6. STFT и спектральная мощность

Функция:

```python
compute_spectrogram_power(samples, nperseg, hop)
```

Она строит спектрограмму мощности, но в отличие от `plot_broadband_spectrogram.py` сначала возвращает линейную мощность, а не dB.

Алгоритм:

1. временной ряд режется на frames;
2. из каждого frame вычитается среднее;
3. применяется окно Ханна;
4. считается `rfft`;
5. берется квадрат модуля спектра;
6. мощность нормируется на энергию окна.

Ключевая логика:

```python
centered = frames - frames.mean(axis=1, keepdims=True)
spectrum = np.fft.rfft(centered * window, axis=1)
power = abs(spectrum) ** 2
power /= sum(window ** 2)
```

Выход:

```text
power:     [freq_bins, time_bins]
time_axis: centers of STFT frames, samples
freq_axis: normalized rfft frequency bins
```

## 7. Выбор частотной полосы

После построения спектра выбирается частотная полоса:

```python
crop_frequency_band(...)
```

Для текущей задачи сфериков обычно используется:

```text
20-30 kHz
```

Параметры:

```bash
--freq-min 20000
--freq-max 30000
```

Внутри маска строится по частотной оси:

```text
freq_hz = freq_axis * sample_rate
mask = freq_min <= freq_hz <= freq_max
```

## 8. Агрегация мощности по частоте

Сферик - короткое широкополосное событие, поэтому детектор схлопывает выбранную частотную полосу в одну временную кривую.

Функция:

```python
aggregate_band(power, aggregate)
```

Доступные режимы:

- `mean` - средняя мощность по частоте;
- `sum` - суммарная мощность по частоте;
- `max` - максимальная мощность по частоте.

По умолчанию:

```text
aggregate = mean
```

В localization dataset также используется `mean`.

Результат:

```text
band_power[t]
```

Это временная кривая мощности в выбранной полосе.

## 9. Перевод envelope в dB

После агрегации мощность переводится в dB:

```python
series_db = 10 * log10(max(band_power, 1e-12))
```

`1e-12` защищает от логарифма нуля.

В `build_localization_dataset.py` эта кривая называется:

```text
envelope_db
```

Она сохраняется в `.npz` вместе со спектрограммой.

## 10. Сглаживание

Перед порогом envelope можно сгладить скользящим средним.

Функция:

```python
smooth_series(values, smooth_bins)
```

Если:

```text
smooth_bins <= 1
```

сглаживания нет.

Иначе используется uniform kernel:

```python
kernel = ones(smooth_bins) / smooth_bins
series = convolve(values, kernel, mode="same")
```

По умолчанию:

```text
smooth_bins = 3
```

Сглаживание уменьшает одиночные шумовые выбросы, но слишком большое сглаживание может размыть близкие события.

## 11. Robust threshold через MAD

Главный порог строится не через среднее и стандартное отклонение, а через медиану и MAD.

Функция:

```python
robust_threshold(series_db, threshold_mad)
```

Шаги:

```python
median = median(series_db)
mad = median(abs(series_db - median))
robust_sigma = 1.4826 * mad
threshold = median + threshold_mad * robust_sigma
```

Почему MAD:

- медиана устойчивее среднего;
- MAD устойчивее standard deviation;
- короткие всплески меньше портят оценку фонового уровня;
- порог лучше адаптируется к разным уровням шума в разных сегментах.

Если `robust_sigma == 0`, используется обычное стандартное отклонение:

```python
robust_sigma = std(series_db)
```

## 12. Параметр `threshold-mad`

`threshold-mad` задает, насколько выше фонового уровня должна быть envelope, чтобы считаться событием.

Формула:

```text
threshold = median + threshold_mad * robust_sigma
```

Пример:

```text
threshold_mad = 8
```

означает:

```text
порог = медиана + 8 robust_sigma
```

Если `threshold-mad` уменьшить:

- детектор станет чувствительнее;
- найдет больше слабых событий;
- увеличится риск false positives.

Если `threshold-mad` увеличить:

- детектор станет строже;
- ложных срабатываний будет меньше;
- слабые сферики могут быть пропущены.

В `detect_broadband_bursts.py` default:

```text
threshold_mad = 6
```

В `build_localization_dataset.py` для auto-labels default:

```text
threshold_mad = 8
```

То есть для подготовки localization labels используется более строгий порог.

## 13. Поиск локальных пиков

После вычисления threshold ищутся локальные максимумы.

Функция:

```python
pick_peaks(series_db, threshold, min_distance_bins)
```

Кандидат должен удовлетворять условиям:

```text
series[i] >= series[i - 1]
series[i] >  series[i + 1]
series[i] >= threshold
```

То есть точка должна быть локальным максимумом и быть выше robust threshold.

Если кандидатов несколько, они сортируются по высоте от самых сильных к слабым. Затем применяется подавление близких пиков:

```python
if abs(index - previous_selected_index) >= min_distance_bins:
    keep
```

Это похоже на простую non-maximum suppression по времени.

## 14. Минимальное расстояние между событиями

Параметр:

```bash
--min-peak-distance
```

По умолчанию:

```text
0.01 s
```

В bins переводится так:

```python
min_distance_bins = round(min_peak_distance * sample_rate / hop)
```

При:

```text
sample_rate = 100000 Hz
hop = 200 samples
time step = 0.002 s
min_peak_distance = 0.01 s
```

получается:

```text
min_distance_bins = 5
```

То есть два detected peak должны быть разнесены минимум на 5 временных bins, примерно на `10 ms`.

## 15. Сегментация

Детектор может работать:

- по всему файлу;
- по фиксированным сегментам.

Параметр:

```bash
--segment-duration 2
```

режет минутный файл на:

```text
30 сегментов по 2 секунды
```

Сегментация важна для dataset pipeline, потому что обучающие samples имеют фиксированную длительность.

## 16. Output самостоятельного detector CLI

`detect_broadband_bursts.py` сохраняет:

1. PNG-графики детекции;
2. CSV с detected peaks.

PNG состоит из двух панелей:

- верхняя: спектрограмма в выбранной полосе;
- нижняя: band envelope и threshold.

На графике:

- линия threshold показана пунктиром;
- найденные detections отмечены вертикальными полосами;
- на envelope отмечены точки пиков.

CSV содержит:

- `segment_index`;
- `segment_start_s`;
- `segment_stop_s`;
- `peak_time_global_s`;
- `peak_time_local_s`;
- `peak_db`;
- `threshold_db`.

## 17. Использование в `build_localization_dataset.py`

При сборке `dataset_localization` логика такая:

```text
raw bin
  -> channel samples
  -> 2-second segment
  -> power spectrogram
  -> crop 20-30 kHz
  -> envelope_db
  -> robust threshold
  -> pick peaks
  -> event_times_s
  -> target_heatmap
  -> save .npz + review PNG + metadata.csv
```

В коде:

```python
envelope_db = compute_envelope_db(power, aggregate, smooth_bins)
_, _, threshold = robust_threshold(envelope_db, threshold_mad)
peak_indices = pick_peaks(envelope_db, threshold, min_distance_bins)
event_times_s = time_axis_s[peak_indices]
target_heatmap = build_target_heatmap(time_axis_s, event_times_s, sigma_s=heatmap_sigma)
```

Строка metadata получает:

```text
label_source = threshold_auto
quality_flag = unchecked
```

Это означает:

> времена событий получены автоматически и требуют ручной проверки перед использованием как gold labels.

## 18. Target heatmap из threshold labels

Когда пороговый детектор нашел времена:

```text
event_times_s = [0.25, 0.53, 1.10]
```

для каждого события строится Gaussian peak:

```python
gaussian = exp(-0.5 * ((time_axis_s - event_time) / sigma_s) ** 2)
```

Все пики объединяются через максимум:

```python
heatmap = maximum(heatmap, gaussian)
```

По умолчанию:

```text
heatmap_sigma = 0.01 s
```

После ручной проверки `apply_localization_corrections.py` пересобирает `target_heatmap` уже по corrected/manual event times.

## 19. Почему threshold labels не считаются gold

Пороговый детектор прост и полезен, но его ошибки ожидаемы.

Типовые ошибки:

- пропуск слабого сферика;
- ложное срабатывание на шумовой вертикальный артефакт;
- несколько пиков на одном широком событии;
- срабатывание на резкий узкополосный выброс;
- нестабильность на более шумном канале `we`;
- зависимость от выбранного `threshold-mad`;
- зависимость от выбранной частотной полосы.

Поэтому pipeline устроен так:

```text
threshold_auto
  -> review_subset
  -> manual_verified / manual_corrected
  -> supervised training
```

## 20. Связь с review workflow

Пороговый детектор дает первичную гипотезу:

```text
auto_event_times_s_json
```

В interactive reviewer пользователь видит эти времена как белые вертикальные линии.

Дальше возможны варианты:

- `manual_verified` - auto-label принят;
- `manual_corrected` - времена исправлены вручную;
- `artifact_suspected` - сегмент лучше исключить;
- `uncertain` - нужен отдельный разбор.

После применения corrections auto-label перестает быть главным источником истины для reviewed строки.

## 21. Связь с 2D CNN моделью

Пороговый детектор:

- не обучается;
- видит только агрегированную envelope-кривую;
- не использует полный 2D pattern напрямую для принятия решения;
- работает по фиксированному threshold rule.

2D CNN:

- обучается на reviewed labels;
- видит полную спектрограмму `spec_db`;
- может учитывать time-frequency структуру;
- выдает predicted heatmap;
- все равно использует peak picking на выходе, но threshold применяется уже к heatmap модели, а не к band envelope.

Сравнение:

```text
Threshold detector:
spec_db -> band envelope -> MAD threshold -> events

2D CNN:
spec_db -> learned features -> predicted heatmap -> peak threshold -> events
```

## 22. Связь с threshold pseudo-labeling

В директории:

```text
methods/threshold_pseudo_labeling/
```

есть отдельный старый/смежный workflow, где логика detector используется для weak labels:

```text
burst     если найден хотя бы один threshold peak
no_burst  если threshold peaks нет
```

Это уже не localization-задача, а задача binary classification сегмента.

В текущей ветке `feature/2d-sferics-localization` главный фокус другой:

```text
не просто burst/no_burst,
а точные времена событий внутри сегмента
```

## 23. Важные параметры

Для localization dataset наиболее важны:

```text
--freq-min 20000
--freq-max 30000
--segment-duration 2
--time-resolution 0.002
--frequency-resolution 50
--aggregate mean
--threshold-mad 8
--min-peak-distance 0.01
--smooth-bins 3
--heatmap-sigma 0.01
```

Кратко:

- `freq-min/freq-max` - где искать broadband events;
- `segment-duration` - длительность sample;
- `time-resolution` - точность временной сетки;
- `frequency-resolution` - FFT resolution;
- `aggregate` - как схлопывать частотную полосу;
- `threshold-mad` - строгость detector;
- `min-peak-distance` - минимальная дистанция между detections;
- `smooth-bins` - сглаживание envelope;
- `heatmap-sigma` - ширина target пика для обучения CNN.

## 24. Как интерпретировать threshold на графике

На detection plot нижняя панель показывает:

- синяя линия: band envelope в dB;
- красный пунктир: threshold;
- оранжевые точки/области: detections.

Если envelope пересекает threshold, но нет локального максимума, событие не будет выбрано. Детектор выбирает именно локальные пики выше threshold, а не все точки выше порога.

## 25. Преимущества метода

Плюсы:

- простая реализация;
- быстро работает;
- не требует обучающего датасета;
- хорошо подходит для первичной разметки;
- параметры понятны физически;
- дает воспроизводимые auto-labels.

## 26. Ограничения метода

Минусы:

- не различает сложные time-frequency patterns;
- чувствителен к выбору полосы и threshold;
- не учитывает контекст соседних частот так гибко, как CNN;
- может путать артефакты и реальные сферики;
- может хуже работать на более шумном канале;
- не знает, какие ошибки были исправлены человеком, если не дообновить labels вручную.

## 27. Практическая роль в текущем проекте

В текущем localization pipeline пороговый детектор - это не финальная модель, а стартовый weak labeler.

Он нужен, чтобы:

1. быстро получить первичные времена событий;
2. разложить сегменты по bucket:
   - `empty`;
   - `single`;
   - `few`;
   - `dense`;
3. собрать review subset;
4. ускорить ручную разметку;
5. построить первые `target_heatmap`;
6. запустить первый baseline 2D CNN.

После появления manual labels роль detector уменьшается: обучение должно опираться на `manual_verified` и `manual_corrected`, а не на `threshold_auto`.

## 28. Краткое резюме

Пороговый детектор работает так:

```text
1. прочитать raw Broadband файл;
2. извлечь канал ns/we;
3. построить спектрограмму мощности;
4. выбрать полосу 20-30 kHz;
5. агрегировать мощность по частоте;
6. перевести envelope в dB;
7. сгладить envelope;
8. вычислить robust threshold через median + K * MAD;
9. найти локальные пики выше threshold;
10. сохранить времена событий.
```

Это простой и важный baseline. Он не заменяет ручную проверку и не заменяет CNN, но дает удобную первичную структуру данных, с которой начался весь `spec_db -> 2D localization` pipeline.

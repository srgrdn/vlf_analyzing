# Метод 2D CNN локализации сфериков

## 1. Назначение метода

Цель метода - находить времена сфериков внутри широкополосной VLF-записи по спектрограмме.

Вход модели:

- 2D спектрограмма сегмента `spec_db`;
- частотная полоса по умолчанию `20-30 kHz`;
- длительность сегмента по умолчанию `2 s`;
- каналы `ns` и `we` обрабатываются независимо.

Выход модели:

- временная heatmap-кривая той же длины, что и временная ось спектрограммы;
- пики на этой heatmap соответствуют предполагаемым временам сфериков.

Идея метода:

```text
raw Broadband_Data *.bin
  -> канал ns/we
  -> 2-секундный фрагмент
  -> спектрограмма 20-30 kHz
  -> 2D CNN
  -> predicted heatmap по времени
  -> peak picking
  -> список времен сфериков
```

Модель не решает задачу классификации "есть/нет сферик". Она решает задачу локализации: "в какие моменты времени внутри сегмента есть события".

## 2. Исходные данные

Исходные файлы имеют формат `Broadband_Data_*.bin`.

Низкоуровневое чтение выполняется существующими функциями из:

```text
plot_broadband_spectrogram.py
```

Используемые предположения о формате:

- сигнатура файла: `DTLG`;
- данные после заголовка читаются как big-endian `int16`;
- в одном record находятся интерливированные каналы;
- по умолчанию используются 2 канала:
  - `ns`;
  - `we`;
- частота дискретизации на канал: `100000 Hz`.

## 3. Сбор localization dataset

Датасет собирался скриптом:

```bash
python3 build_localization_dataset.py /home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data/raw_data \
  --channel both \
  --recursive \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --overwrite
```

Смысл параметров:

- `--channel both` - собрать оба канала `ns` и `we`;
- `--freq-min 20000`, `--freq-max 30000` - оставить только полосу `20-30 kHz`;
- `--segment-duration 2` - разрезать запись на 2-секундные окна;
- `--time-resolution 0.002` - шаг временной оси спектрограммы около `2 ms`;
- `--frequency-resolution 50` - частотный шаг около `50 Hz`;
- `--recursive` - искать raw-файлы рекурсивно;
- `--overwrite` - пересобрать датасет.

Результат сохраняется в:

```text
dataset_localization/
  metadata.csv
  samples/
  review/
```

В текущем датасете:

- `6000` сегментов всего;
- `3000` сегментов канала `ns`;
- `3000` сегментов канала `we`;
- источник - `100` минутных raw-файлов;
- каждый минутный файл дает 30 сегментов на канал при длительности сегмента `2 s`.

## 4. Содержимое одного `.npz` sample

Каждый сегмент сохраняется как `.npz` в:

```text
dataset_localization/samples/*.npz
```

Основные поля:

- `spec_db` - 2D спектрограмма в dB;
- `envelope_db` - временная огибающая мощности;
- `time_axis` - временная ось внутри сегмента, секунды;
- `freq_axis` - частотная ось, Hz;
- `event_times_s` - времена событий внутри сегмента;
- `event_count` - количество событий;
- `target_heatmap` - целевая временная heatmap для обучения;
- `source_file` - исходный `.bin`;
- `channel` - `ns` или `we`;
- `segment_start`, `segment_end` - границы сегмента в исходном файле.

Главные поля для этой модели:

```text
input  = spec_db
target = target_heatmap
```

## 5. Первичные auto-labels

Изначально времена сфериков были получены автоматически пороговым детектором.

Такие строки в `metadata.csv` имеют:

```text
label_source = threshold_auto
```

Эти auto-labels не считаются полностью надежными. Они нужны для первичной разметки, отбора примеров и ускорения ручной проверки.

## 6. Ручная проверка и corrections workflow

Для ручной проверки был собран stratified review subset:

```bash
python3 build_localization_review_subset.py dataset_localization
python3 init_localization_corrections.py dataset_localization/review_subset
```

Первый subset:

- `160` примеров;
- `80 ns`;
- `80 we`;
- bucket-группы:
  - `empty`;
  - `single`;
  - `few`;
  - `dense`.

Затем была добавлена интерактивная проверка:

```bash
python3 review_localization_interactive.py dataset_localization
```

В интерактивном окне:

- белые линии - auto-detected времена;
- красные линии - времена, выбранные вручную;
- `v` - принять auto-label как корректный;
- `enter` или `c` - сохранить ручные клики как исправленные времена;
- `a` - артефакт;
- `u` - спорный случай.

После первой проверки:

- `160/160` строк проверены;
- `85 manual_corrected`;
- `75 manual_verified`.

Правки применялись командой:

```bash
python3 apply_localization_corrections.py dataset_localization
```

При применении правок обновляются:

- `metadata.csv`;
- соответствующие `.npz`;
- `event_times_s`;
- `event_count`;
- `target_heatmap`;
- review PNG в `review/verified` или `review/corrected`.

## 7. Active review subset v2

После первого baseline-обучения был добавлен active-review отбор:

```bash
python3 build_localization_active_review_subset.py dataset_localization --overwrite
python3 init_localization_corrections.py dataset_localization/review_subset_v2
```

Цель второго subset - добрать более полезные примеры:

- из еще непроверенных строк;
- с приоритетом на `we`;
- с приоритетом на `few` и `dense`;
- с учетом source files, где модель ранее ошибалась;
- с сохранением отрицательных и простых примеров для баланса.

Второй subset:

- `400` примеров;
- `200 ns`;
- `200 we`;
- `35 empty`;
- `167 single`;
- `118 few`;
- `80 dense`.

После ручной проверки и применения:

- `400/400` строк v2 проверены;
- `266 manual_corrected`;
- `134 manual_verified`.

Итоговое состояние reviewed dataset:

- `560` reviewed строк;
- `351 manual_corrected`;
- `209 manual_verified`;
- `5440 threshold_auto`;
- `280 ns`;
- `280 we`.

## 8. Разбиение train/val/test

Разбиение выполнялось скриптом:

```bash
python3 split_localization_reviewed.py dataset_localization
```

Важное правило:

> сегменты из одного `source_file` не должны попадать одновременно в разные split.

Это нужно, чтобы избежать утечки почти одинаковых соседних сегментов между train и test.

Текущий split reviewed-строк:

- `train`: `415`;
- `val`: `70`;
- `test`: `75`.

Строки `threshold_auto` остаются без split и не используются для обучения текущей supervised-модели.

## 9. Построение target heatmap

Модель обучается не напрямую на списке времен, а на временной heatmap.

Например, ручная разметка:

```text
event_times_s = [0.25, 0.53, 1.10]
```

преобразуется в кривую:

```text
target_heatmap(t)
```

На этой кривой около каждого события находится пик. Визуально это похоже на узкий "колокол" в момент сферика.

Такой подход удобен, потому что:

- CNN может выдавать плотную временную карту;
- loss считается по всей временной оси;
- модель учится не только количеству событий, но и их положению;
- близкие события могут быть представлены несколькими пиками.

## 10. Нормализация входа

Перед подачей в модель спектрограмма нормализуется отдельно для каждого sample:

```python
spec = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)
```

Это уменьшает зависимость от абсолютного уровня сигнала и помогает модели фокусироваться на структуре спектрограммы.

## 11. Архитектура модели

Модель определена в:

```text
methods/2d_cnn_localization/train_sferics_localization_2d.py
```

Класс:

```python
SfericsLocalizationCNN
```

Архитектура состоит из двух частей:

1. 2D feature extractor по спектрограмме;
2. temporal head по временной оси.

### 11.1. Вход

Форма входа для PyTorch:

```text
[batch, channels, freq_bins, time_bins]
```

Так как спектрограмма одна, `channels = 1`.

Пример:

```text
[B, 1, F, T]
```

где:

- `B` - batch size;
- `F` - число частотных bins в полосе `20-30 kHz`;
- `T` - число временных bins в 2-секундном сегменте.

### 11.2. Feature extractor

```text
Conv2d(1 -> 16, kernel=3, padding=1)
BatchNorm2d(16)
ReLU
MaxPool2d(kernel=(2, 1))

Conv2d(16 -> 32, kernel=3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(kernel=(2, 1))

Conv2d(32 -> 64, kernel=3, padding=1)
BatchNorm2d(64)
ReLU
```

Особенность pooling:

```text
MaxPool2d(kernel=(2, 1))
```

Он уменьшает размерность по частоте, но не сжимает временную ось. Это важно, потому что задача - точно локализовать событие во времени.

После 2D-блоков получается тензор признаков:

```text
[B, 64, reduced_F, T]
```

Затем модель усредняет признаки по частоте:

```python
temporal = features.mean(dim=2)
```

Получается:

```text
[B, 64, T]
```

То есть для каждого момента времени есть 64 признака, агрегированных по частотной полосе.

### 11.3. Temporal head

```text
Conv1d(64 -> 64, kernel=5, padding=2)
ReLU
Dropout(0.15)
Conv1d(64 -> 1, kernel=1)
Sigmoid
```

Temporal head превращает временную последовательность признаков в одну heatmap:

```text
[B, T]
```

`Sigmoid` ограничивает значения heatmap диапазоном `0..1`.

Важно: это не строго калиброванная вероятность. Значение heatmap лучше понимать как относительную уверенность/похожесть на событие.

## 12. Принцип обучения

Обучение идет по supervised-схеме:

```text
spec_db -> model -> predicted_heatmap
target_heatmap известно из ручной разметки
```

Loss:

```python
MSELoss(predicted_heatmap, target_heatmap)
```

Также считается MAE:

```python
mean(abs(predicted_heatmap - target_heatmap))
```

Оптимизатор:

```text
AdamW
```

Основные параметры:

- `learning_rate = 1e-3`;
- `weight_decay = 1e-4`;
- `batch_size = 16`;
- `seed = 42`;
- CPU/GPU выбирается автоматически.

Сохраняется не последний epoch, а лучшая модель по validation loss:

```text
best_val_loss
```

Checkpoint:

```text
methods/2d_cnn_localization/models/sferics_localization_2d.pt
```

## 13. Роль `peak_threshold`

Модель во время обучения не использует `peak_threshold`.

Во время обучения есть только:

```text
predicted_heatmap vs target_heatmap
```

`peak_threshold` нужен после модели, чтобы превратить heatmap в список событий:

```text
predicted_heatmap
  -> найти локальные пики
  -> оставить пики выше peak_threshold
  -> получить event_times_s
```

Пример:

```text
пики heatmap: 0.92, 0.58, 0.31, 0.12
peak_threshold = 0.35
события: 0.92, 0.58
```

Практически `peak_threshold` можно понимать как порог уверенности модели, но это не калиброванная вероятность.

Если threshold ниже:

- выше recall;
- больше найденных слабых событий;
- больше риск false positives.

Если threshold выше:

- выше precision;
- меньше ложных срабатываний;
- больше риск пропустить слабые сферики.

## 14. Метрики

### 14.1. Heatmap metrics

`heatmap_mse`:

```text
mean((predicted_heatmap - target_heatmap)^2)
```

Показывает среднеквадратичную ошибку формы heatmap.

`heatmap_mae`:

```text
mean(abs(predicted_heatmap - target_heatmap))
```

Показывает среднюю абсолютную ошибку heatmap.

Эти метрики оценивают качество кривой, но не всегда напрямую совпадают с качеством списка событий.

### 14.2. Event-level metrics

После peak picking считаются предсказанные времена событий.

Предсказанное событие считается совпавшим с ручным событием, если расстояние по времени не больше:

```text
match_tolerance = 0.04 s
```

Считаются:

- `event_tp` - правильно найденные события;
- `event_fp` - лишние события;
- `event_fn` - пропущенные события;
- `event_precision`;
- `event_recall`;
- `event_f1`;
- `event_count_mae`.

Формулы:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

`event_count_mae` - средняя ошибка количества событий в сегменте:

```text
mean(abs(predicted_event_count - true_event_count))
```

## 15. Результаты текущего обучения

После расширения датасета до `560` reviewed строк пользователь обучал модель на CPU.

Пример динамики после 15 epochs:

```text
epoch 01: train_loss=0.048852 train_mae=0.157048 val_loss=0.021734 val_mae=0.083636
...
epoch 15: train_loss=0.006591 train_mae=0.027626 val_loss=0.008204 val_mae=0.028671
```

Event-level metrics при `peak_threshold=0.35`, `match_tolerance=0.04`:

Validation:

```text
precision = 0.840
recall    = 0.896
F1        = 0.867
count MAE = 0.843
```

Test:

```text
precision = 0.689
recall    = 0.898
F1        = 0.779
count MAE = 1.640
```

Интерпретация:

- модель хорошо находит большинство событий;
- recall высокий;
- test precision ниже, то есть по метрикам есть лишние пики;
- визуальная проверка нового raw-файла показала, что многие пики heatmap совпадают с видимыми сфериками;
- поэтому следующий важный шаг - независимая human verification на новых raw-файлах.

## 16. Notebook workflow

Основной интерактивный файл:

```text
methods/2d_cnn_localization/train_sferics_localization_2d.ipynb
```

Ключевые секции:

- `Reviewed Split Summary` - проверка состава train/val/test;
- `Inspect One Training Sample` - визуализация одного sample: спектрограмма + target heatmap;
- `Train` - обучение модели;
- `Loss Curves` - графики train/val loss;
- `Peak Threshold Sweep` - перебор threshold и сравнение F1/count MAE;
- `Test Predictions Preview` - таблица худших/интересных test-примеров;
- `Plot Worst Heatmap Examples` - визуальное сравнение target vs prediction;
- `Saved Model Checkpoint` - загрузка сохраненной модели;
- `Run Saved Model On New Raw File` - инференс на новом raw;
- `Plot Raw File Predictions` - визуальная проверка предсказаний на raw.

## 17. CLI для обучения

Запуск обучения из командной строки:

```bash
python3 methods/2d_cnn_localization/train_sferics_localization_2d.py \
  --dataset-dir dataset_localization
```

Основные outputs:

```text
methods/2d_cnn_localization/models/sferics_localization_2d.pt
methods/2d_cnn_localization/reports/sferics_localization_2d_metrics.json
methods/2d_cnn_localization/reports/sferics_localization_2d_val_predictions.csv
methods/2d_cnn_localization/reports/sferics_localization_2d_test_predictions.csv
```

## 18. CLI для инференса на raw-файлах

Для повторяемого запуска без notebook:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both
```

С сохранением диагностических графиков:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both \
  --save-plots \
  --plot-top 6
```

Outputs:

```text
methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_predictions.csv
methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_summary.json
methods/2d_cnn_localization/reports/raw_inference/plots/
```

Пример статистики для одного нового файла:

```text
By channel:
  ns: segments=30 events=82 mean=2.73 max=6
  we: segments=30 events=309 mean=10.30 max=17
```

Здесь:

- `segments` - число 2-секундных сегментов;
- `events` - общее число найденных событий;
- `mean` - среднее число событий на сегмент;
- `max` - максимум событий в одном сегменте.

## 19. Интерактивная верификация модели

Для независимой проверки обученной модели на новых raw-файлах добавлен human-in-the-loop workflow:

```bash
python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both
```

На графике:

- белые линии - события модели;
- красные линии - ручные исправления пользователя;
- нижний график - predicted heatmap.

Горячие клавиши:

- `v` - модель сработала правильно;
- `enter` или `c` - сохранить красные клики как исправленную разметку;
- `f` - false positive, исправленный список событий пустой;
- `m` - missed events, модель пропустила события, сохранить клики как истинные события;
- `backspace` - удалить последний клик;
- `r` - вернуть времена модели;
- `x` - очистить клики;
- `a` - артефакт;
- `u` - спорный случай;
- `s` - пропустить;
- `q` - выйти.

CSV верификации:

```text
methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_raw_verification.csv
```

Смысл статусов:

- `model_verified` - модель правильно нашла события;
- `manual_corrected` - модель частично ошиблась, пользователь исправил времена;
- `false_positive` - модель нашла события, но пользователь считает, что их нет;
- `missed_events` - модель пропустила события, пользователь добавил их;
- `artifact_suspected` - сегмент артефактный;
- `uncertain` - спорный случай.

## 20. Сводка верификации и графики

После ручной проверки:

```bash
python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py \
  --save-plots
```

Outputs:

```text
methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_summary.json
methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_segment_metrics.csv
methods/2d_cnn_localization/reports/raw_verification/plots/
```

Строятся:

- counts по статусам;
- precision/recall/F1 по каналам;
- гистограмма ошибки количества событий;
- scatter plot `predicted_count` vs `corrected_count`.

## 21. Почему выбран такой подход

Преимущества heatmap-регрессии:

- модель сохраняет временное разрешение;
- можно находить несколько событий в одном сегменте;
- можно визуально анализировать уверенность модели;
- архитектура простая и CPU-friendly;
- удобно совмещается с ручной проверкой.

Преимущества 2D CNN:

- модель видит не только огибающую мощности, но и полную time-frequency структуру;
- вертикальные broadband-следы сфериков хорошо представлены на спектрограмме;
- горизонтальные помехи и фоновые линии частично отделимы от коротких вертикальных событий.

## 22. Ограничения текущей версии

Текущая модель является baseline, а не финальной production-системой.

Ограничения:

- обучалась только на `560` reviewed сегментах;
- `peak_threshold` пока подбирается как post-processing параметр;
- heatmap output не является строго калиброванной вероятностью;
- `we` может быть существенно более активным или шумным, чем `ns`;
- test precision ниже validation precision;
- необходимо провести независимую верификацию на новом raw-наборе.

## 23. Лучшие следующие шаги

Рекомендуемая последовательность:

1. Подготовить новый raw-набор, не использованный при сборе train/val/test.
2. Прогнать модель:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py <new_raw_dir> \
  --channel both \
  --save-plots
```

3. Визуально посмотреть top-сегменты.
4. Запустить интерактивную верификацию:

```bash
python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py <new_raw_dir> \
  --channel both
```

5. Собрать статистику:

```bash
python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py \
  --save-plots
```

6. По validation/verification выбрать `peak_threshold`.
7. Добрать новые active-review примеры, если обнаружатся типовые ошибки.
8. Переобучить модель на расширенном reviewed dataset.

## 24. Краткое резюме

Метод строит supervised 2D CNN локализатор сфериков:

```text
spec_db -> temporal heatmap -> event times
```

Датасет был собран из raw Broadband files, автоматически размечен пороговым детектором, затем частично вручную проверен и исправлен. Текущая модель обучается только на `manual_verified` и `manual_corrected` строках. Архитектура сохраняет временную ось, сжимает частотную ось и предсказывает heatmap, пики которой соответствуют временам сфериков.

Ключевая идея: модель должна не просто сказать, что событие есть, а показать, **где именно во времени** оно находится.

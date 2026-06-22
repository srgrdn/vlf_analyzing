# Basic Signal Processing Artifacts

Source file:

```text
/home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin
```

Parsed metadata:

- record size: `60`
- record count: `200000`
- data offset: `818`
- channels: `2`
- per-channel sample rate: `100000 Hz`
- per-channel duration: `60 s`

Generated artifacts:

```text
full/
  spectrograms/       # full-duration spectrograms, 0-50 kHz, ns/we
  oscillograms/       # full-duration RMS dBFS envelopes, ns/we
segments_2s_20_30khz/
  spectrograms/
    ns/               # 30 PNGs, 2-second segments, 20-30 kHz
    we/               # 30 PNGs, 2-second segments, 20-30 kHz
segments_2s/
  oscillograms/
    ns/               # 30 PNGs, 2-second RMS dBFS envelopes
    we/               # 30 PNGs, 2-second RMS dBFS envelopes
```

Commands used for full spectrograms:

```bash
python3 plot_broadband_spectrogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns --cmap jet \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/full/spectrograms/Broadband_Data_2026.06.02_13.51.00_ns_full_spectrogram.png

python3 plot_broadband_spectrogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel we --cmap jet \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/full/spectrograms/Broadband_Data_2026.06.02_13.51.00_we_full_spectrogram.png
```

Commands used for full oscillograms:

```bash
python3 plot_broadband_oscillogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/full/oscillograms/Broadband_Data_2026.06.02_13.51.00_ns_full_oscillogram.png

python3 plot_broadband_oscillogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel we \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/full/oscillograms/Broadband_Data_2026.06.02_13.51.00_we_full_oscillogram.png
```

Commands used for segmented spectrograms:

```bash
python3 plot_broadband_spectrogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel ns \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --cmap jet \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/segments_2s_20_30khz/spectrograms/ns/Broadband_Data_2026.06.02_13.51.00_ns_20_30khz_spectrogram.png

python3 plot_broadband_spectrogram.py /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel we \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --freq-min 20000 \
  --freq-max 30000 \
  --cmap jet \
  --output reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/segments_2s_20_30khz/spectrograms/we/Broadband_Data_2026.06.02_13.51.00_we_20_30khz_spectrogram.png
```

Segmented oscillograms were generated with the same low-level reader and `compute_envelope_db(...)` from `plot_broadband_oscillogram.py`, using:

- segment duration: `2 s`
- envelope mode: `rms`
- envelope time step: `0.01 s`
- reference: `32768`

# Audio sonification for Broadband_Data_2026.03.25_20.11.00.bin

This directory contains WAV sonifications of the VLF band used by the 2D CNN pipeline.

Source file:

```text
/home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data/raw_data_v2/Broadband_Data_2026.03.25_20.11.00.bin
```

Processing:

1. read DTLG binary payload;
2. extract `ns` and `we` channels;
3. band-pass filter `20-30 kHz`;
4. build analytic signal;
5. frequency-shift `20-30 kHz` down to `0-10 kHz`;
6. low-pass filter at `10 kHz`;
7. resample from `100 kHz` to `48 kHz`;
8. normalize to int16 WAV.

Generated files:

- `Broadband_Data_2026.03.25_20.11.00_ns_20-30kHz_shifted_to_0-10kHz.wav`
- `Broadband_Data_2026.03.25_20.11.00_we_20-30kHz_shifted_to_0-10kHz.wav`
- `Broadband_Data_2026.03.25_20.11.00_stereo_ns_we_20-30kHz_shifted_to_0-10kHz.wav`

Stereo file channels:

- left: `ns`
- right: `we`

Note: this is not the original audible recording. It is a sonification: the VLF band was shifted into the human-audible range.

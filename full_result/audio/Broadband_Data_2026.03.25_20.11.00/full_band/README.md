# Full-band audio for Broadband_Data_2026.03.25_20.11.00.bin

Source file:

```text
/home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data/raw_data_v2/Broadband_Data_2026.03.25_20.11.00.bin
```

This directory contains full-band audio exports without selecting the `20-30 kHz` band.

Generated variants:

1. `*_fullband_direct_100kHz.wav`
   - original channel samples saved as WAV;
   - sample rate: `100000 Hz`;
   - duration: `60 s`;
   - preserves the full available band up to Nyquist `50 kHz`;
   - much of the high-frequency content is not audible on normal speakers/headphones.

2. `*_fullband_0-50kHz_to_0-24kHz_48kHz.wav`
   - same samples written at `48000 Hz`;
   - frequency-compressed sonification: original `0-50 kHz` becomes approximately `0-24 kHz`;
   - duration becomes approximately `125 s`;
   - useful for listening to the full-band structure in the audible range.

Stereo files:

- left channel: `ns`;
- right channel: `we`.

Note: the 48 kHz version is a sonification, not a physically time-correct recording.

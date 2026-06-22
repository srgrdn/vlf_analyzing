# Experimental Sferic Arrival-Axis Estimation

This report summarizes an experimental single-station azimuth analysis based on two orthogonal loop antenna channels, `ns` and `we`.

## Summary

- processed events: `296`
- reliable events: `4`
- dominant arrival-axis bin: `50-60 degrees (2 events)`
- mean arrival axis for reliable events: `63.08491674264452`

The result is an **arrival line / bearing axis**, not a source coordinate. Because a single station with two orthogonal loop antennas has a 180-degree ambiguity, every result is represented by two possible azimuth candidates:

```text
azimuth_candidate_1 = arrival_axis_deg
azimuth_candidate_2 = arrival_axis_deg + 180 degrees
```

## Generated Plots

- `azimuth_histogram.png`
- `azimuth_rose.png`
- `ratio_we_ns_histogram.png`
- `pca_linearity_histogram.png`
- `azimuth_by_file_time.png`
- `azimuth_direction_map.png`
- `examples/`

## Interpretation

The peak of the histogram indicates the most common estimated line of arrival for the analyzed events. The opposite direction is physically equivalent in this preliminary single-station estimate because of the 180-degree ambiguity.

Distance to the lightning discharge is **not** estimated by this method.

The map figure uses Irkutsk as the receiver position and draws the dominant bearing axis plus the opposite candidate. It is a geographic schematic of the line of arrival only, not a localization result.

## Quality Flags

Quality counts:

```json
{
  "good": 4,
  "low_snr": 54,
  "low_snr;single_channel_detection": 208,
  "low_snr;weak_linearity": 2,
  "low_snr;weak_linearity;single_channel_detection": 1,
  "weak_linearity": 5,
  "weak_linearity;single_channel_detection": 22
}
```

Events with low energy, low SNR, weak PCA linearity, saturation, or a single-channel detection are retained in the CSV but marked as unreliable via `quality_flag`.

## Method Limitations

- `ns` and `we` channels currently do not have precise amplitude-phase calibration.
- Loop antennas have a figure-eight directional pattern.
- A single station gives a 180-degree ambiguity.
- Absolute amplitude does not determine distance to the source.
- The estimated azimuth should be compared with external lightning-location data, for example Blitzortung or WWLLN.
- Full source localization requires a network of multiple synchronized receivers.

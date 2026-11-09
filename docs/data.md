# Input data

The core model starts at frozen visual features. Prepare a 10 Hz cache with
camera-aligned steering-wheel angles in degrees and speed in metres per second.
Use only data you are authorised to access. Data files are not part of this release.

## Cache schema

```text
features/
  index.json
  session_a.feats.npy
  session_a.meta.npy
  session_b.feats.npy
  session_b.meta.npy
```

Example `index.json`:

```json
{
  "base_hz": 10.0,
  "sessions": {
    "session_a": {"split": "train", "cluster": 0},
    "session_b": {"split": "val", "cluster": 1}
  }
}
```

Each feature file has shape `(N,D)`, stored as float16 or float32. Each metadata
file has shape `(N,3)`, stored as float64, with columns:

1. Camera timestamp in seconds.
2. Steering-wheel angle in degrees; NaN means unavailable or invalid.
3. Speed in metres per second; NaN means unavailable or invalid.

Feature and metadata rows refer to the same frames. `FeatureCache` memory maps
the arrays, converts features to float32 per batch and retains float64 timestamps
until elapsed/relative times have been computed. All sessions share the same `D`.
Session identifiers are local file stems. Splits are `train`, `val` and `test`.
`cluster` is optional and supports a caller's route-level analysis.

## Window and label rules

The default window contains 30 observations at 10 Hz, nominally three seconds.
A target must have a finite steering label and finite speed greater than 3 m/s.
Every interval within the history must be positive and at most 0.2 s. Windows
never cross session boundaries. Preserve these rules when writing another loader.
Invalid labels should already be marked NaN before creating the cache; the reader
cannot reconstruct CAN alignment or identify invalid logging defaults from values
alone.

For `task="current"`, the target is the final frame's angle, shape `(B,)`.
For `task="anticipation"`, targets are angles at 0, 0.5, 1 and 2 s after that
frame, shape `(B,4)`. Future angles are linearly interpolated between cached
steering observations. A label is invalid if the session ends first, either
interpolation endpoint is invalid, the bracket spans more than 0.2 s, or the clock
steps backwards before the bracket. Only targets valid at all four horizons are
retained, so horizons share identical target frames.

The provided future-label lookup scans at most 64 subsequent frames, sufficient
for the supported two-second horizon at 10 Hz. Its supported task configuration
uses the four horizons above.

## Feature preparation used for the reported setup

The visual encoder is frozen DINOv2 ViT-B/14. Resize RGB frames to width 392 and
height 224, scale to `[0,1]`, then normalise channels by mean `(0.485, 0.456, 0.406)`
and standard deviation `(0.229, 0.224, 0.225)`. Concatenate the normalised CLS token
and mean of normalised patch tokens to obtain 1536 features per frame. Cache
camera-aligned observations at 10 Hz. The image encoder and raw-data preparation
are implemented by the user of this core package.

## Inference inputs

`Predictor.predict(features, timestamps)` uses unpadded chronological windows
with the current frame last. It accepts one through 30 frames for a default
checkpoint, including irregular gaps. Steering labels and speed are not inputs
to the predictor. An anticipation checkpoint returns four columns, ordered as
`predictor.horizons`. Predictions are already transformed back to degrees.

# Coarse-to-Fine Explicit-Matching Model — Experimental

**Status: experimental research/testing artifact only. Not clinically validated.
Not approved for clinical use or treatment/positioning decisions.**

This is a new, **non-U-Net** architecture. It is separate from the Region-Guided
V8 U-Net checkpoints and is not loadable through `NativeRegionGuidedV8` or the
existing V8 shift adapter unchanged. Use the included `model.py` loader for
offline experiments only.

## Important accuracy limitation

The selected checkpoint demonstrates strong performance on generated rigid
translations but poor performance on real validation and diagnostic held-out
pairs. The real-pair results are not acceptable evidence of useful real-world
motion estimation. Synthetic success does not establish clinical or deployment
performance.

| Evaluation subset | Cases | Mean error (px) | p95 error (px) |
|---|---:|---:|---:|
| Validation: synthetic 0 px | 24 | 0.00 | 0.00 |
| Validation: synthetic 10 px | 96 | 0.18 | 0.45 |
| Validation: synthetic 40 px | 96 | 0.02 | 0.05 |
| Validation: synthetic 80 px | 96 | 0.05 | 0.14 |
| Validation: synthetic 120 px | 192 | 0.09 | 0.24 |
| Validation: real 0–10 px | 48 | 5.74 | 28.96 |
| Validation: real 10–30 px | 48 | 10.72 | 46.54 |
| Validation: real 30–50 px | 48 | 10.10 | 29.41 |
| Validation: real 50–80 px | 48 | 23.41 | 92.88 |
| Validation: real 80–120 px | 48 | 38.50 | 162.45 |
| Diagnostic held-out: real 0–10 px | 24 | 17.13 | 102.71 |
| Diagnostic held-out: real 10–30 px | 24 | 22.50 | 90.37 |
| Diagnostic held-out: real 30–50 px | 24 | 18.03 | 58.85 |
| Diagnostic held-out: real 50–80 px | 24 | 23.20 | 42.96 |
| Diagnostic held-out: real 80–120 px | 24 | 32.29 | 59.89 |

The diagnostic held-out evaluation is a small subset, **not full test
certification**. Do not compare its summary statistics as if they were a
representative clinical performance estimate. See `config.json` for evaluation
scope and the independently recorded aggregate metrics.

## Architecture

The model applies a small shared convolutional feature pyramid to the fixed and
moving grayscale frames. It computes mask-restricted, overlap-normalized linear
correlations at 1/8 and 1/4 internal feature resolutions, selects a coarse
translation and refines it locally with a subpixel peak estimate. These are
internal feature scales: the image input and output displacement units remain
native 640 × 480 pixels.

The method explicitly searches a finite displacement window. Ambiguous texture,
low overlap, occlusion, changing silhouettes, and nonrigid motion can still
cause incorrect matches. Synthetic translations do not capture all of these
real-pair conditions.

## Input/output contract

`model.py` provides `CoarseToFineMatching`. Inputs are grayscale `uint8` frames
of shape `[480,640]` and binary masks of shape `[5,480,640]`, ordered:
`outer_masks`, `body`, `arms`, `face`, `hair`. Pass reference/fixed first and
current/moving second. Each frame is divided by 255 exactly once; model-side
preprocessing standardizes within the supplied outer mask.

The result is five `(dx,dy)` corrections in **native pixels**, direction
**moving-to-fixed**. If the moving frame was translated right by `+d` pixels,
the correction should be left by `-d` pixels. No output multiplier or resize
scale is applied at 640 × 480.

The live adapter's actual mask selection depends on upstream inputs: when a
nonempty parsing map is supplied it uses that map for the outer mask and named
regions; without parsing, it uses the supplied YOLO mask as the outer mask and
leaves the other regions empty. This experimental model was trained using
cached Sapiens-derived masks. Performance with the YOLO-only fallback mask has
not been established.

Example:

```python
import sys
from pathlib import Path

version = Path("model/CoarseToFine-ExplicitMatching-640x480-20261027")
sys.path.insert(0, str(version))
from model import CoarseToFineMatching

predictor = CoarseToFineMatching(version / "model.weights.h5")
corrections = predictor.predict_moving_to_fixed(
    fixed_gray_u8, moving_gray_u8, fixed_masks, moving_masks
)
outer_dx, outer_dy = corrections[0]
```

`tensorflow` and the repository's
`region_guided_v8.preprocessing.AdapterPreprocessing` package must be installed
and importable.

## Training provenance

- Initialized from random weights; no prior checkpoint was loaded.
- Best validation checkpoint: update 600.
- Batch size: 8; Adam with gradient clipping; warm-up/cosine learning-rate schedule.
- Training mixture: 50% real, 15% identical-frame, 35% known synthetic translations.
- Accepted examples were balanced equally across skips 1, 2, 3, 5, 8, 10, 15, 20,
  and 30, with full shuffled pair permutations within each per-skip/type pool.
- The run stopped after eight validation checks without an eligible score improvement.

The experimental archive contains only model weights, configuration, aggregate
metrics, and loader code. It intentionally excludes source frames, masks,
TFRecords, patient identifiers, and per-example predictions. `config.json`
contains only aggregate metrics and no patient-level information.

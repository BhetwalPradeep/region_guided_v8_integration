# Tiered Dense-Flow U-Net: experimental 640x480 checkpoint

**Observation-only research/phantom testing. Not clinical-ready or approved for
automatic couch control.** Real small-motion performance regressed relative to
the earlier U-Net. Do not interpret strong synthetic results as deployment
accuracy. Pixels are not calibrated couch millimeters.

This is a new architecture, not V8 and not RAFT: shared five-level U-Net encoder,
global correspondence at 30x40, local costs with six ConvGRU iterations at
120x160, and native float32 linear flow refinement. It has 319,692 parameters.
Regional rigid corrections use exact component-wise flow medians inside moving
masks. Fewer than 100 pixels in either region mask produces `valid=False`.

## Checkpoint provenance and training

Best checkpoint: sampled epoch 68, 40,800 optimizer updates, random initialization.
Training reached 75 completed validation checks (45,000 updates) before a later
all-foreground-out-of-view synthetic example triggered the loss safety assertion.
This was not normal completion/early stopping. Best/latest saved models survived;
this bundle contains **best only**.

Static existing TFRecords were read without modification; original fixed frames
and masks supplied bases for dynamic CPU translations. Each occurrence sampled
50% log-uniform 1-5px or 50% log-uniform 5-300px magnitude and a uniform direction.
Training used synthetic visible-foreground dense truth, not real-pair labels.
Batch 1, native 480x640, Adam with clip norm 1, 600-step warm-up to 2e-4 and cosine
decay to 1e-5. Normalized sequence Smooth L1 supervises all ten intermediate
outputs with gamma .8. Zero/sub-1px magnitudes were not explicitly sampled.

## Accuracy evidence and limitations

Best synthetic validation mean foreground flow error on 100 deterministic cases:

| Metric | Mean error (px) |
|---|---:|
| Overall | 0.3701 |
| Micro 1-5px | 0.2943 |
| Macro 5-300px | 0.4521 |

These are dense flow errors, not regional median errors at exact boundaries.
The earlier controlled 1px/300px evaluation used an **older** checkpoint and
must not be attributed to this final checkpoint. Final full boundary evaluation
was not yet available when this bundle was prepared.

Matched real validation: 48 pairs per bin, unchanged identities and registration
proxy targets versus the previous scratch U-Net:

| Proxy-motion bin | Previous mean outer error | New mean outer error | New p95 |
|---|---:|---:|---:|
| 0-10px | 1.908 | 8.307 | 33.545 |
| 10-30px | 13.745 | 10.253 | 58.080 |
| 30-50px | 33.891 | 9.107 | 23.710 |
| 50-80px | 57.265 | 11.186 | 20.704 |
| 80-120px | 88.137 | 21.203 | 30.736 |

Real outer Dice averaged .8771 before / .8956 after correction, but **134/240
pairs worsened**. Small-motion outer Dice averaged .9601 before / .9574 after.
Face/hair alignment improved more than outer/arms. These labels are registration
estimates, not independently measured couch movement; no real dense ground truth
or independent held-out accuracy is established. This is a mixed-result
experimental checkpoint, not a replacement accuracy release.

## Loading

The default loader uses **`model.weights.h5`**, the same weights-only checkpoint
format as previous releases. The architecture is new: do not load these weights
with the old V8 builder. `model.keras` is also retained as a full-model export.
After cloning the branch, run `git lfs pull` to fetch the H5 checkpoint.

## Socket-ready handoff

Use the **server inside this model folder**, not the repository-root legacy
256x256 V8 server. It retains the previous newline-delimited JSON over TCP
protocol and path-based request keys. This is not a WebSocket service.

From the repository root:

```bash
git checkout native-640x480-inference
git lfs pull
CUDA_VISIBLE_DEVICES=-1 python \
  model/Tiered-DenseFlow-UNet-640x480-20261028/socket_server.py \
  --host 127.0.0.1 --port 5057 \
  --checkpoint model/Tiered-DenseFlow-UNet-640x480-20261028/model.weights.h5
```

CPU is selected above to avoid allocating uncontrolled GPU memory on a shared
server. Choose and configure a GPU explicitly in your deployment environment.
Tested runtime is TensorFlow 2.21; use it for initial integration.

Send one JSON object followed by a newline:

```json
{"fixed_frame_path":"/shared/fixed.npy","moving_frame_path":"/shared/moving.npy","fixed_masks_path":"/shared/fixed_masks.npy","moving_masks_path":"/shared/moving_masks.npy"}
```

Paths must exist **on the server**. Frames must be `(480,640)` grayscale uint8,
masks `(5,480,640)` in outer/body/arms/face/hair order, with values 0/1 or 0/255.
For outer-only mode, populate channel 0 and zero the other channels. Separate
YOLO-path overrides are explicitly rejected: supply already-selected masks.

Responses preserve `ok`, `mask_order`, `fixed_to_moving_dxdy`,
`moving_to_fixed_dxdy`, and `registered_output_path`. Both direction arrays
have shape `(5,2)`. Use **`moving_to_fixed_dxdy[0]`** for the outer/global
correction; do not negate it again. Native 640x480 scaling is **1:1**.
Additional fields include `valid_regions`, `global_couch_shift`, pixel units,
and per-region Dice/IoU before/after. Invalid non-outer channels contain zero
placeholders: ignore their shifts/overlap scores when `valid_regions` is false.
An invalid outer mask returns `ok:false` with an explicit error.

An optional `registered_output_path` saves a uint8 **outer-rigid preview**.
Unlike the old V8 server, this does not blend five regional shifts into a
deformation field. This intentional difference is identified in the response
as `registration_mode:"outer_rigid"`.

The endpoint accepts filesystem paths and can save a requested output file.
Use only a trusted local client/network, no public exposure or untrusted
requests; this legacy-compatible transport has no authentication/TLS.
No couch movement/control is implemented. Keep it observation-only.

## Python loader

Tested with Python 3.12 and TensorFlow 2.21/Keras in float32. Older versions are
unverified. Put this directory on `sys.path` before importing:

```python
import sys
sys.path.insert(0, "/absolute/path/to/Tiered-DenseFlow-UNet-640x480-20261028")
from loader import Predictor

predictor = Predictor()
result = predictor.predict(packed)
# result["flow"]: [B,480,640,2], moving-grid (Vx,Vy)
# result["shifts"]: [B,5,2], moving-to-fixed corrections
# result["valid"]: [B,5], MUST check before using any region
# result["global_couch_shift"]: [B,2], outer median
# result["dice_before"], ["dice_after"], ["iou_before"], ["iou_after"]: [B,5]
ten_values = predictor.predict_moving_to_fixed(packed)
```

`packed` is float32 `[B,480,640,12]`: fixed gray, moving gray, five fixed
masks, five moving masks. Gray is unmasked `[0,1]`, divided by 255 **once**.
Masks are binary, ordered **outer, body, arms, face, hair**. The model performs
outer-mask standardization internally. Do not standardize again externally.
No resize, physical-unit conversion or output multiplier is applied.

Positive synthetic translation +40px right must predict approximately -40px
correction. Flow obeys `fixed_coordinate = moving_coordinate + flow`.
The aggregator applies a constant regional rigid translation when calculating
after-overlap, not an arbitrary nonrigid moving-grid warp.

The existing RegionGuidedV8ShiftAdapter is **not automatically compatible**:
select this loader explicitly and verify packing, sign, units, validity and
camera axes before observation-only testing. `predict_moving_to_fixed` rejects
an invalid outer region, but missing non-outer regions are zero placeholders;
use `predict()` for explicit per-region validity.

No patient images, masks, record identities, training data or per-patient
evaluation outputs are included. Only model weights, inference code and
aggregate metrics are published.

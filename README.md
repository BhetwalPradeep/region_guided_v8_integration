# Region-Guided V8 Clinical Integration Package

This package contains the locked region-guided V8 model and a socket-ready inference wrapper for clinical moving-to-fixed registration.

## Final-14p integration-test release

The versioned `V8-RG-final-14p-20261005` release is a candidate for supervised,
non-treatment-affecting integration testing, not a validated clinical positioning
system. The previous epoch-65 release remains the default below.

Download the new checkpoint and matching configuration into its dated folder without replacing the legacy default:

```bash
python download_artifacts.py --release final-14p
python socket_server.py --host 127.0.0.1 --port 5056 \
  --checkpoint model/V8-RG-final-14p-20261005/model.weights.h5 \
  --config model/V8-RG-final-14p-20261005/config.json
```

Use a separate test client/port; no automatic couch control. The transport is
newline-delimited JSON over TCP, not WebSocket. Client file paths must be readable
on the server. No segmentation models are bundled: the caller supplies frames and masks.

The wrapper reads `target_direction` from the checkpoint config. Final-14p
predicts **moving-to-fixed directly**. Both response fields retain their names
and meanings: `moving_to_fixed_dxdy` is the native correction, while
`fixed_to_moving_dxdy` is its negation. Do not negate the correction again in the
client. Legacy configs without a direction retain fixed-to-moving behavior.

Python usage:

```python
model = RegionGuidedV8(
  checkpoint="model/V8-RG-final-14p-20261005/model.weights.h5",
  config="model/V8-RG-final-14p-20261005/config.json",
)
```

Release details and SHA-256 are in `releases/final-14p.json`; the downloader
verifies the downloaded weights and configuration direction. Model versions live
under dated folders in `model/`; for example:

```text
model/
  checkpoint_best.weights.h5                    # legacy default, unchanged
  config.json                                    # legacy default, unchanged
  V8-RG-final-14p-20261005/
    model.weights.h5                            # tracked with Git LFS
  config.json
```

`.gitattributes` tracks `*.weights.h5` with Git LFS so future model checkpoints
are versioned with Git without storing their binary contents as ordinary Git
blobs. After cloning, install Git LFS and fetch the pointers' contents with
`git lfs pull` if automatic checkout did not fetch them. LFS storage/bandwidth
is subject to the GitHub account's quota. Use `model/V8-RG-<version>-YYYYMMDD/`
for each new checkpoint version, with a `model.weights.h5` filename and its
matching `config.json`; never overwrite an older dated directory. S3 remains a versioned backup/mirror;
TFRecords, raw frames, masks, and per-patient evaluation files remain outside
Git. Do not commit patient-identifying data.

To track a future weight file, `*.weights.h5` is already covered by `.gitattributes`.
Stage the dated weight normally; Git LFS converts it to a pointer in the Git commit
and uploads the binary object on push. Verify with `git lfs ls-files` before pushing.

### Training and evaluation

- 14 patients (former train + validation), 16,251 frames, 125,199 record pairs;
  all three acquisition sessions represented in training.
- Corrected loader continues across epochs through the record stream.
- Fixed 29 epochs based on the earlier FullData validation-selected epoch;
  no validation in this final fit. Despite its filename, `checkpoint_best` is
  the final epoch, not a validation-selected checkpoint.
- Cosine horizon retained at 100 x 400 optimizer steps; the inherited loop
  applies 799 optimizer updates per 400-batch epoch. This behavior was retained,
  not corrected, to match the selection run's schedule.
- Same 1,755 skip-10 test pairs from three patients excluded from training;
  this test set has been reused in multiple experiments.

| Region | Direct run | Final-14p | Valid pairs |
|---|---:|---:|---:|
| Outer | 0.9319 | 0.9322 | 1755 |
| Body | 0.8354 | 0.8352 | 1755 |
| Arms | 0.6329 | 0.6457 | 1023 |
| Face | 0.8075 | 0.8028 | 1505 |
| Hair | 0.8294 | 0.8278 | 1755 |

Values are mean Dice after full five-region displacement-field correction on
valid pairs. Final-14p is not an across-the-board improvement over Direct.

### Integration checks

Outputs are pixels of the 256 x 256 model image, not mm or couch coordinates.
For a full-frame resize from 640 x 480 to 256 x 256, map back with
`dx_camera = dx_model * 640/256` and `dy_camera = dy_model * 480/256`.
Cropping, padding, camera orientation, and physical calibration require their
own transforms. The five-vector blended field is not independently predicted
per-pixel optical flow and does not provide a measured 3D couch correction.

Verify signs with known phantom translations; check readout/arrow agreement,
mask failures, stale frames, latency, and connectivity before approved shadow-mode
testing. Do not interpret display thresholds as clinically validated tolerances.

Run compatibility tests:

```bash
python -m unittest discover -s tests -v
```

## Contents

```text
model/
  checkpoint_best.weights.h5
  config.json
  V8-RG-final-14p-20261005/
    model.weights.h5
    config.json
releases/
  final-14p.json
region_guided_v8/
  inference.py
  clinical_registration.py
  train_consecutive.py
  models.py
  warp_fn.py
socket_server.py
requirements.txt
docs/
  test_eval_region_guided_epoch65_clinical_moving_to_fixed.csv
  test_eval_region_guided_epoch65_yolo_masked.csv
  clinical_overlays_10_cases_full_dvf.tar.gz
```

## Artifact Storage (S3)

The legacy model and large training/QC artifacts remain in the shared S3 prefix;
new dated model versions are stored in Git LFS and mirrored to versioned S3 prefixes:

```text
s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/
```

Recommended layout:

```text
s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/
  model/
    checkpoint_best.weights.h5
    config.json
  docs/
    clinical_overlays_10_cases_full_dvf.tar.gz
    test_eval_region_guided_epoch65_clinical_moving_to_fixed.csv
    test_eval_region_guided_epoch65_yolo_masked.csv
  training/
    tfrecords/
    raw_frames/
    masks/
```

GitHub stores model checkpoint versions through Git LFS (not normal Git blobs).
Large TFRecords, raw frame caches, masks, and QC archives remain in S3 and are
not committed to Git.

## Download artifacts locally

If the model weights or validation docs are missing locally, download them from S3 before running inference:

```bash
python download_artifacts.py
```

This script downloads the locked checkpoint, config, and validation docs into the local repo layout.

If you prefer to do it manually:

```bash
mkdir -p model docs
aws s3 cp s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/model/checkpoint_best.weights.h5 model/checkpoint_best.weights.h5
aws s3 cp s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/model/config.json model/config.json
aws s3 cp s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/docs/ docs/ --recursive --exclude '*' --include '*.csv' --include '*.tar.gz'
```

## Inference configuration

The repository now includes a dedicated runtime config file for the clinical inference settings:

```text
inference_config.json
```

This file captures the locked runtime contract, including:

- checkpoint path
- config path
- model architecture
- image shape and mask count
- region order
- clinical direction convention
- socket host/port
- S3 artifact reference

This is the explicit inference configuration for reporting and integration handoff.

## Locked Model

| Item | Value |
|---|---|
| Architecture | `region_guided_attention` |
| Checkpoint | `model/checkpoint_best.weights.h5` |
| Best epoch | 65 |
| Input shape | `256 x 256 x 12` |
| Output shape | `5 x (dx, dy)` = 10 values |
| Region order | outer, body, arms, face, hair |
| Parameters | 13,387,898 |

## Training Settings

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Minimum learning rate | `1e-6` |
| Batch size | 64 |
| Steps per epoch | 400 |
| Validation steps | 70 |
| Max epochs | 100 |
| Early stopping patience | 25 |
| Synthetic shift probability | 0.7 |
| Synthetic shift range | +/-50 px |
| Shift distribution | mixture |
| Border mode | reflect |
| Training patients | 12 |
| Validation patients | 2 |
| Test patients | 3 |

Frame preprocessing used by the TFRecords: YOLO outer masking with masked normalization. The five Sapiens masks are included as region-guidance channels.

## Clinical Inference Convention

The model predicts displacement from the fixed/reference image to the moving/later image:

```text
fixed -> moving
```

Clinical registration needs the opposite operation: keep the fixed image unchanged and warp the moving image back to fixed coordinates.

```text
registered_moving = warp(moving, -predicted_dxdy)
```

This package applies that inverse direction in `RegionGuidedV8.register_moving_to_fixed()`.

## Python API

```python
import numpy as np
from region_guided_v8 import RegionGuidedV8

model = RegionGuidedV8(
    checkpoint='model/checkpoint_best.weights.h5',
    config='model/config.json',
)

fixed_frame = np.load('fixed_frame.npy')      # shape (256, 256), uint8
moving_frame = np.load('moving_frame.npy')    # shape (256, 256), uint8
fixed_masks = np.load('fixed_masks.npy')      # shape (5, 256, 256), uint8/bool
moving_masks = np.load('moving_masks.npy')    # shape (5, 256, 256), uint8/bool

result = model.predict_and_register(
    fixed_frame_u8=fixed_frame,
    moving_frame_u8=moving_frame,
    fixed_masks=fixed_masks,
    moving_masks=moving_masks,
)

print(result['fixed_to_moving_dxdy'])
print(result['moving_to_fixed_dxdy'])
registered_moving = result['registered_moving']
```

## Socket Server

Start server:

```bash
python socket_server.py --host 127.0.0.1 --port 5055
```

Request protocol: one JSON object per line.

```json
{
  "fixed_frame_path": "/path/fixed_frame.npy",
  "moving_frame_path": "/path/moving_frame.npy",
  "fixed_masks_path": "/path/fixed_masks.npy",
  "moving_masks_path": "/path/moving_masks.npy",
  "fixed_yolo_path": "/path/fixed_yolo.npy",
  "moving_yolo_path": "/path/moving_yolo.npy",
  "registered_output_path": "/path/registered_moving.npy"
}
```

`fixed_yolo_path`, `moving_yolo_path`, and `registered_output_path` are optional. If YOLO masks are not provided, the outer Sapiens mask is used as fallback.

Response:

```json
{
  "ok": true,
  "mask_order": ["outer_masks", "body", "arms", "face", "hair"],
  "fixed_to_moving_dxdy": [[dx, dy], ...],
  "moving_to_fixed_dxdy": [[-dx, -dy], ...],
  "registered_output_path": "/path/registered_moving.npy"
}
```

## Validation

Clinical moving-to-fixed test CSV:

```text
docs/test_eval_region_guided_epoch65_clinical_moving_to_fixed.csv
```

Mean clinical Dice delta by region:

| Region | Mean Dice delta |
|---|---:|
| Outer | +0.015449 |
| Body | +0.031707 |
| Arms | -0.002784 |
| Face | +0.187950 |
| Hair | +0.237180 |

Synthetic sign validation: a known `(12, -7)` px shift corrected from Dice `0.7067` to `1.0000` using inverse displacement.

## Notes for FDA Documentation

Training configuration and inference configuration should be reported separately. Training augmentation is a development setting and is not active during inference. The clinical integration direction is moving-to-fixed, even though the model prediction convention is fixed-to-moving.

# Region-Guided V8 Clinical Integration Package

This package contains the locked region-guided V8 model and a socket-ready inference wrapper for clinical moving-to-fixed registration.

## Contents

```text
model/
  checkpoint_best.weights.h5
  config.json
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

The source-of-truth model and validation assets live in the shared S3 prefix:

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

GitHub should keep only the code, API contract, lightweight docs, and a manifest pointing to the S3 artifacts. Large files such as checkpoints, TFRecords, raw frame caches, and large QC archives are intentionally not committed to Git.

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

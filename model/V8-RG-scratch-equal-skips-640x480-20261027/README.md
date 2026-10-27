# Native 640x480 scratch-training experiment

**Experimental; not clinically validated; not release approved.** Large-motion errors remain high. Consult the aggregate metrics in [config.json](./config.json); this is not a successful accuracy release.

Trained from random weights with equal accepted contributions from skips 1, 2, 3, 5, 8, 10, 15, 20, and 30. Full per-skip pair permutations are used, but early stopping need not visit every pair. Real, identical, and synthetic examples use separate permutation streams. Real labels are registration proxies; synthetic motion extends through 120 px.

Use `model.weights.h5` with this config through the updated `NativeRegionGuidedV8` loader. Input is reference/current grayscale divided by 255 once plus five binary masks each, shape [B,480,640,12]. Internal outer-mask standardization is required. Output is ten moving-to-fixed native-pixel components; no output multiplier. Adapter API is unchanged.

Selected update: 3000. Weights SHA-256: `648a0eb39c1dca595d63be9fbb5d46b80d5058a861b36aa38dbef669970270c7`.

Only weights, configuration, aggregate metrics, and parity verification are published. No frames, masks, patient identifiers, TFRecords, or per-example predictions are included. Held-out metrics describe a diagnostic subset, not full test certification.

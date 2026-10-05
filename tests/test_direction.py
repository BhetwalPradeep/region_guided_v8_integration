import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from region_guided_v8.inference import RegionGuidedV8


class DirectionTest(unittest.TestCase):
    def test_checkpoint_directions(self):
        correction = np.tile([4.0, -2.0], (5, 1)).astype(np.float32)
        masks = np.ones((5, 256, 256), np.uint8)
        frame = np.zeros((256, 256), np.uint8)
        frame[100, 100] = 255
        for direction in (None, "fixed_to_moving", "moving_to_fixed"):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as folder:
                config_path = Path(folder) / "config.json"
                config_path.write_text(json.dumps({} if direction is None else {"target_direction": direction}))
                native = correction if direction == "moving_to_fixed" else -correction
                network = Mock()
                network.return_value.numpy.return_value = native.reshape(1, 10)
                with patch("region_guided_v8.inference.build_region_guided_attention", return_value=network):
                    model = RegionGuidedV8(Path(folder) / "weights.h5", config_path)
                result = model.predict_and_register(frame, frame, masks, masks)
                np.testing.assert_array_equal(result["moving_to_fixed_dxdy"], correction)
                np.testing.assert_array_equal(result["fixed_to_moving_dxdy"], -correction)
                self.assertEqual(result["registered_moving"][98, 104], 255)
                self.assertEqual(result["registered_moving"][100, 100], 0)

    def test_unknown_direction_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            config_path.write_text(json.dumps({"target_direction": "unknown"}))
            with self.assertRaises(ValueError):
                RegionGuidedV8(Path(folder) / "weights.h5", config_path)


if __name__ == "__main__":
    unittest.main()
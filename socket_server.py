#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socketserver
from pathlib import Path

import numpy as np

from region_guided_v8 import RegionGuidedV8

MODEL: RegionGuidedV8 | None = None


def load_array(payload: dict, key: str) -> np.ndarray:
    path_key = f"{key}_path"
    if path_key not in payload:
        raise KeyError(f"missing required field: {path_key}")
    return np.load(payload[path_key])


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        assert MODEL is not None
        for line in self.rfile:
            try:
                payload = json.loads(line.decode("utf-8"))
                fixed_frame = load_array(payload, "fixed_frame")
                moving_frame = load_array(payload, "moving_frame")
                fixed_masks = load_array(payload, "fixed_masks")
                moving_masks = load_array(payload, "moving_masks")
                fixed_yolo = np.load(payload["fixed_yolo_path"]) if payload.get("fixed_yolo_path") else None
                moving_yolo = np.load(payload["moving_yolo_path"]) if payload.get("moving_yolo_path") else None
                result = MODEL.predict_and_register(fixed_frame, moving_frame, fixed_masks, moving_masks, fixed_yolo, moving_yolo)
                output_path = payload.get("registered_output_path")
                if output_path:
                    np.save(output_path, result["registered_moving"])
                response = {
                    "ok": True,
                    "mask_order": ["outer_masks", "body", "arms", "face", "hair"],
                    "fixed_to_moving_dxdy": result["fixed_to_moving_dxdy"].tolist(),
                    "moving_to_fixed_dxdy": result["moving_to_fixed_dxdy"].tolist(),
                    "registered_output_path": output_path,
                }
            except Exception as exc:
                response = {"ok": False, "error": repr(exc)}
            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description="Region-guided V8 socket inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--checkpoint", default="model/checkpoint_best.weights.h5")
    parser.add_argument("--config", default="model/config.json")
    args = parser.parse_args()
    MODEL = RegionGuidedV8(Path(args.checkpoint), Path(args.config))
    with socketserver.ThreadingTCPServer((args.host, args.port), Handler) as server:
        print(f"listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

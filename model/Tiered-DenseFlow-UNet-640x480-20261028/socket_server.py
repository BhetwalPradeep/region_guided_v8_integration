#!/usr/bin/env python3
"""Observation-only native server; matches the repository's newline-JSON/path protocol."""
import argparse
import json
from pathlib import Path
import socketserver
import threading

import numpy as np

from loader import Predictor
from tiered_flow.geometry import translate


MASK_ORDER = ["outer_masks", "body", "arms", "face", "hair"]


def build_input(fixed, moving, fixed_masks, moving_masks):
    frames = [np.asarray(fixed), np.asarray(moving)]
    masks = [np.asarray(fixed_masks), np.asarray(moving_masks)]
    for frame in frames:
        if frame.shape != (480, 640) or frame.dtype != np.uint8:
            raise ValueError("Frames must be native (480,640) uint8 grayscale")
    for mask in masks:
        if mask.shape != (5, 480, 640) or not np.isfinite(mask).all():
            raise ValueError("Masks must be finite (5,480,640), outer/body/arms/face/hair")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Masks must contain only 0/1 or 0/255")
    return np.concatenate([
        np.stack(frames, -1).astype(np.float32) / 255.,
        (masks[0] > 0).transpose(1, 2, 0), (masks[1] > 0).transpose(1, 2, 0),
    ], -1)[None].astype(np.float32)


def process_request(predictor, payload):
    for key in ("fixed_yolo_path", "moving_yolo_path"):
        if payload.get(key):
            raise ValueError("Supply selected masks directly; separate YOLO overrides are unsupported")
    arrays = {}
    for key in ("fixed_frame", "moving_frame", "fixed_masks", "moving_masks"):
        arrays[key] = np.load(payload[f"{key}_path"], allow_pickle=False)
    packed = build_input(**dict(
        fixed=arrays["fixed_frame"], moving=arrays["moving_frame"],
        fixed_masks=arrays["fixed_masks"], moving_masks=arrays["moving_masks"]))
    result = predictor.predict(packed)
    if not bool(result["global_valid"][0]):
        raise ValueError("Outer region has fewer than 100 pixels; no valid global correction")
    shifts = np.asarray(result["shifts"])[0]
    output = payload.get("registered_output_path")
    if output:
        # A global rigid preview, not the old blended five-region deformation.
        registered = translate(arrays["moving_frame"][None, ..., None],
                               shifts[None, 0]).numpy()[0, ..., 0]
        np.save(output, np.clip(np.rint(registered), 0, 255).astype(np.uint8))
    return dict(
        ok=True, mask_order=MASK_ORDER, moving_to_fixed_dxdy=shifts.tolist(),
        fixed_to_moving_dxdy=(-shifts).tolist(),
        valid_regions=np.asarray(result["valid"])[0].tolist(),
        global_couch_shift=shifts[0].tolist(), output_units="native_pixels",
        registered_output_path=output, registration_mode="outer_rigid",
        dice_before=np.asarray(result["dice_before"])[0].tolist(),
        dice_after=np.asarray(result["dice_after"])[0].tolist(),
        iou_before=np.asarray(result["iou_before"])[0].tolist(),
        iou_after=np.asarray(result["iou_after"])[0].tolist(),
        release_approved=False)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(120)
        for line in self.rfile:
            try:
                payload = json.loads(line.decode("utf-8"))
                with self.server.inference_lock:
                    response = process_request(self.server.predictor, payload)
            except Exception as error:
                response = dict(ok=False, error=repr(error))
            self.wfile.write((json.dumps(response, allow_nan=False) + "\n").encode())
            self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).with_name("model.weights.h5"))
    args = parser.parse_args()
    predictor = Predictor(checkpoint=args.checkpoint)
    with Server((args.host, args.port), Handler) as server:
        server.predictor = predictor
        server.inference_lock = threading.Lock()
        print(f"listening on {args.host}:{args.port}; experimental observation-only", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Consecutive-frame contourwise shift prediction on CLEAN_RESULTS.

Pipeline:
  - Input: 2-channel grayscale (frame_t, frame_t+1) resized to 256x256
  - Target: 10-dim vector (dx,dy for 5 masks: outer_masks, body, arms, face, hair)
  - Ground truth: centroid shift of largest contour between consecutive mask frames
  - Model: FlexUNet encoder-decoder with GlobalAveragePooling → dense head
  - Loss: Huber (validity-weighted per mask component)
  - Data: reads JPG/PNG directly from CLEAN_RESULTS (no preprocessing step)
  - Split: patient-level 10 train / 3 val / 2 test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

# Prevent TF from grabbing all GPU memory at once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from tensorflow.keras import layers

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ─── Constants ────────────────────────────────────────────────────────────────

MASK_TYPES = ["outer_masks", "body", "arms", "face", "hair"]
OUTPUT_DIMS = len(MASK_TYPES) * 2  # 10

SESSIONS = {
    "061526": "FINAL_061526_ARTIFACT_PACKAGE_20260813",
    "061626": "FINAL_061626_ARTIFACT_PACKAGE_20260813",
}

CAMERAS = ["PV002", "PV015", "PV016", "PV017", "PV019"]

TRAIN_PATIENTS = [
    "anand", "babu", "benvogel", "bhawin", "ivyatabong",
    "julievasu", "pradeep", "rajesh", "shivam", "tupendra",
    "lorirobinson", "tarwinder",
]
VAL_PATIENTS = ["chrisvolpe", "hayleyjohnson"]
TEST_PATIENTS = ["kevinmichaels", "lizpeters", "namanesh"]

IMG_SIZE = (256, 256)  # (H, W)


# ─── Data discovery ──────────────────────────────────────────────────────────

def discover_sequences(clean_root: Path, patients: List[str]) -> List[dict]:
    """Find all consecutive-frame sequences for given patients.

    Returns list of dicts, each describing a continuous run of frames:
      {session, camera, patient, frame_paths: [Path,...], mask_dirs: {type: Path}}
    """
    sequences = []

    for session_id, artifact_dir in SESSIONS.items():
        base = clean_root / artifact_dir
        frames_base = base / "raw_frames" / session_id
        masks_base = base / "final_masks_parts_previews" / session_id

        for camera in CAMERAS:
            cam_frames = frames_base / camera
            cam_masks = masks_base / camera / "output"
            if not cam_frames.exists():
                continue

            # Group frames by patient
            patient_files: Dict[str, List[Path]] = defaultdict(list)
            for f in sorted(cam_frames.iterdir()):
                if f.suffix.lower() != ".jpg" or f.name.startswith("."):
                    continue
                parts = f.stem.split("_")
                if len(parts) < 4 or parts[0].isdigit():
                    continue
                patient_name = parts[0]
                if patient_name in patients:
                    patient_files[patient_name].append(f)

            for patient, fpaths in patient_files.items():
                fpaths = sorted(fpaths)
                if len(fpaths) < 2:
                    continue

                # Check which mask dirs exist
                mask_dirs = {}
                for mt in MASK_TYPES:
                    md = cam_masks / mt
                    if md.exists():
                        mask_dirs[mt] = md

                # Split into continuous runs (gap > 5s = new sequence)
                runs = []
                current_run = [fpaths[0]]
                for i in range(1, len(fpaths)):
                    t_prev = _parse_time_sec(fpaths[i - 1].stem)
                    t_curr = _parse_time_sec(fpaths[i].stem)
                    if t_prev is not None and t_curr is not None and (t_curr - t_prev) > 5:
                        if len(current_run) >= 2:
                            runs.append(current_run)
                        current_run = [fpaths[i]]
                    else:
                        current_run.append(fpaths[i])
                if len(current_run) >= 2:
                    runs.append(current_run)

                for run in runs:
                    sequences.append({
                        "session": session_id,
                        "camera": camera,
                        "patient": patient,
                        "frame_paths": run,
                        "mask_dirs": mask_dirs,
                    })

    return sequences


def _parse_time_sec(stem: str) -> int | None:
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    t = parts[2]
    if len(t) != 6 or not t.isdigit():
        return None
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


# ─── Image loading ───────────────────────────────────────────────────────────

def load_gray_resized(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_mask_resized(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(IMG_SIZE, dtype=np.float32)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    return (img > 127).astype(np.float32)


# ─── Target computation ──────────────────────────────────────────────────────

def centroid_of_largest(mask: np.ndarray) -> Tuple[float, float] | None:
    binary = (mask >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def compute_target(
    mask_dirs: Dict[str, Path],
    stem_t: str,
    stem_t1: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centroid shift targets between frame t and t+1 for all mask types."""
    targets = np.zeros(OUTPUT_DIMS, dtype=np.float32)
    valid = np.zeros(OUTPUT_DIMS, dtype=np.float32)

    for mi, mt in enumerate(MASK_TYPES):
        if mt not in mask_dirs:
            continue
        mask_t_path = mask_dirs[mt] / (stem_t + ".png")
        mask_t1_path = mask_dirs[mt] / (stem_t1 + ".png")
        if not mask_t_path.exists() or not mask_t1_path.exists():
            continue

        mask_t = load_mask_resized(mask_t_path)
        mask_t1 = load_mask_resized(mask_t1_path)

        c_t = centroid_of_largest(mask_t)
        c_t1 = centroid_of_largest(mask_t1)
        if c_t is None or c_t1 is None:
            continue

        dx = c_t1[0] - c_t[0]
        dy = c_t1[1] - c_t[1]
        targets[mi * 2] = dx
        targets[mi * 2 + 1] = dy
        valid[mi * 2] = 1.0
        valid[mi * 2 + 1] = 1.0

    return targets, valid


# ─── Batch index and generation ──────────────────────────────────────────────

def build_pair_index(sequences: List[dict], frame_skip: int = 1) -> List[Tuple[int, int]]:
    """Build flat index of (seq_idx, frame_offset) for pairs with given skip."""
    pairs = []
    for si, seq in enumerate(sequences):
        n = len(seq["frame_paths"])
        for fi in range(n - frame_skip):
            pairs.append((si, fi))
    return pairs


def make_batch(
    sequences: List[dict],
    pair_indices: List[Tuple[int, int]],
    batch_indices: np.ndarray,
    frame_skip: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_batch, y_batch, w_batch = [], [], []

    for idx in batch_indices:
        si, fi = pair_indices[idx]
        seq = sequences[si]
        fp_t = seq["frame_paths"][fi]
        fp_t1 = seq["frame_paths"][fi + frame_skip]

        frame_t = load_gray_resized(fp_t)
        frame_t1 = load_gray_resized(fp_t1)

        target, valid = compute_target(seq["mask_dirs"], fp_t.stem, fp_t1.stem)

        x_batch.append(np.stack([frame_t, frame_t1], axis=-1))
        y_batch.append(target)
        w_batch.append(valid)

    return np.stack(x_batch), np.stack(y_batch), np.stack(w_batch)


# ─── Model ────────────────────────────────────────────────────────────────────

def conv_block(x, filters, n_convs=2, name_prefix="", norm="layer"):
    for i in range(n_convs):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_c{i}")(x)
        if norm == "batch":
            x = layers.BatchNormalization(name=f"{name_prefix}_bn{i}")(x)
        else:
            x = layers.LayerNormalization(axis=-1, name=f"{name_prefix}_ln{i}")(x)
    return x


def build_flexunet(input_shape=(256, 256, 2), output_dims=10, norm="layer") -> keras.Model:
    inp = keras.Input(shape=input_shape, name="input")
    x = inp
    skips = []
    ch = 32

    for i in range(5):
        x = conv_block(x, ch, 2, name_prefix=f"enc{i}", norm=norm)
        skips.append(x)
        if i < 4:
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            ch = min(ch * 2, 512)

    x = conv_block(x, ch, 2, name_prefix="bottleneck", norm=norm)

    for i in range(5):
        skip_idx = 4 - i
        if i < 4:
            x = layers.UpSampling2D(2, name=f"up{i}")(x)
            ch = max(ch // 2, 32)
        skip = skips[skip_idx]
        x = layers.Lambda(
            lambda t: tf.image.resize(t[0], [tf.shape(t[1])[1], tf.shape(t[1])[2]], method="bilinear"),
            name=f"resize{i}",
        )([x, skip])
        x = layers.Concatenate(name=f"cat{i}")([x, skip])
        x = conv_block(x, ch, 2, name_prefix=f"dec{i}", norm=norm)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(ch, activation="relu", name="fc1")(x)
    out = layers.Dense(output_dims, name="output")(x)
    return keras.Model(inp, out, name="flexunet_consecutive_shift")


def build_flexunet_attention(input_shape=(256, 256, 2), n_masks=5, norm="layer") -> keras.Model:
    """FlexUNet with per-mask spatial attention heads."""
    inp = keras.Input(shape=input_shape, name="input")
    x = inp
    skips = []
    ch = 32

    for i in range(5):
        x = conv_block(x, ch, 2, name_prefix=f"enc{i}", norm=norm)
        skips.append(x)
        if i < 4:
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            ch = min(ch * 2, 512)

    x = conv_block(x, ch, 2, name_prefix="bottleneck", norm=norm)

    for i in range(5):
        skip_idx = 4 - i
        if i < 4:
            x = layers.UpSampling2D(2, name=f"up{i}")(x)
            ch = max(ch // 2, 32)
        skip = skips[skip_idx]
        x = layers.Lambda(
            lambda t: tf.image.resize(t[0], [tf.shape(t[1])[1], tf.shape(t[1])[2]], method="bilinear"),
            name=f"resize{i}",
        )([x, skip])
        x = layers.Concatenate(name=f"cat{i}")([x, skip])
        x = conv_block(x, ch, 2, name_prefix=f"dec{i}", norm=norm)

    # Spatial attention: one attention map per mask
    attn = layers.Conv2D(n_masks, 1, padding="same", name="attn_conv")(x)
    attn = layers.Activation("softmax", name="attn_softmax")(
        layers.Reshape((-1, n_masks), name="attn_reshape_pre")(attn)
    )  # [B, H*W, n_masks]

    feat_flat = layers.Reshape((-1, ch), name="feat_flatten")(x)  # [B, H*W, ch]

    mask_outputs = []
    for mi in range(n_masks):
        # Extract attention for this mask: [B, H*W, 1]
        a_i = layers.Lambda(lambda t: t[:, :, mi:mi+1], name=f"attn_slice_{mi}")(attn)
        # Weighted pool: [B, ch]
        pooled = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name=f"attn_pool_{mi}")([feat_flat, a_i])
        # Per-mask head: [B, 2]
        h = layers.Dense(ch // 2, activation="relu", name=f"mask_fc_{mi}")(pooled)
        dxdy = layers.Dense(2, name=f"mask_out_{mi}")(h)
        mask_outputs.append(dxdy)

    out = layers.Concatenate(name="output")(mask_outputs)  # [B, n_masks*2]
    return keras.Model(inp, out, name="flexunet_spatial_attention")


def build_region_guided_attention(input_shape=(256, 256, 12), n_masks=5,
                                  norm="layer") -> keras.Model:
    """V8 U-Net with explicit per-frame masks and regional feature pooling."""
    inp = keras.Input(shape=input_shape, name="input")
    x = inp[:, :, :, :2]
    skips = []
    ch = 32
    for i in range(5):
        x = conv_block(x, ch, 2, name_prefix=f"enc{i}", norm=norm)
        skips.append(x)
        if i < 4:
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            ch = min(ch * 2, 512)
    x = conv_block(x, ch, 2, name_prefix="bottleneck", norm=norm)
    for i in range(5):
        skip_idx = 4 - i
        if i < 4:
            x = layers.UpSampling2D(2, name=f"up{i}")(x)
            ch = max(ch // 2, 32)
        skip = skips[skip_idx]
        x = layers.Lambda(
            lambda t: tf.image.resize(t[0], [tf.shape(t[1])[1], tf.shape(t[1])[2]], method="bilinear"),
            name=f"resize{i}",
        )([x, skip])
        x = layers.Concatenate(name=f"cat{i}")([x, skip])
        x = conv_block(x, ch, 2, name_prefix=f"dec{i}", norm=norm)

    masks = inp[:, :, :, 2:]
    outputs = []
    for mi in range(n_masks):
        region = layers.Lambda(
            lambda t, i=mi: tf.maximum(t[:, :, :, i], t[:, :, :, i + n_masks]),
            name=f"region_{mi}",
        )(masks)
        region = layers.Lambda(lambda t: t[:, :, :, None], name=f"region_expand_{mi}")(region)
        pooled = layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=[1, 2]) /
            tf.maximum(tf.reduce_sum(t[1], axis=[1, 2]), 1.0),
            name=f"region_pool_{mi}",
        )([x, region])
        h = layers.Dense(ch // 2, activation="relu", name=f"region_fc_{mi}")(pooled)
        outputs.append(layers.Dense(2, name=f"region_out_{mi}")(h))
    return keras.Model(inp, layers.Concatenate(name="output")(outputs),
                       name="region_guided_attention")


# ─── Training steps ──────────────────────────────────────────────────────────

@tf.function
def train_step_mse(model, optimizer, x, y, w):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        diff = pred - y
        mse = tf.square(diff)
        gt_mag = tf.sqrt(tf.reduce_sum(tf.reshape(y, [-1, 5, 2]) ** 2, axis=-1))
        mag_weight = 1.0 + gt_mag
        mag_weight_expanded = tf.repeat(mag_weight, 2, axis=-1)
        combined_w = w * mag_weight_expanded
        loss = tf.reduce_sum(mse * combined_w) / tf.maximum(tf.reduce_sum(combined_w), 1.0)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def train_step_huber(model, optimizer, x, y, w):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        diff = pred - y
        abs_diff = tf.abs(diff)
        huber = tf.where(abs_diff < 1.0, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
        gt_mag = tf.sqrt(tf.reduce_sum(tf.reshape(y, [-1, 5, 2]) ** 2, axis=-1))
        mag_weight = 1.0 + gt_mag
        mag_weight_expanded = tf.repeat(mag_weight, 2, axis=-1)
        combined_w = w * mag_weight_expanded
        loss = tf.reduce_sum(huber * combined_w) / tf.maximum(tf.reduce_sum(combined_w), 1.0)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def val_step_mse(model, x, y, w):
    pred = model(x, training=False)
    diff = pred - y
    mse = tf.square(diff)
    gt_mag = tf.sqrt(tf.reduce_sum(tf.reshape(y, [-1, 5, 2]) ** 2, axis=-1))
    mag_weight = 1.0 + gt_mag
    mag_weight_expanded = tf.repeat(mag_weight, 2, axis=-1)
    combined_w = w * mag_weight_expanded
    return tf.reduce_sum(mse * combined_w) / tf.maximum(tf.reduce_sum(combined_w), 1.0)


@tf.function
def val_step_huber(model, x, y, w):
    pred = model(x, training=False)
    diff = pred - y
    abs_diff = tf.abs(diff)
    huber = tf.where(abs_diff < 1.0, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    gt_mag = tf.sqrt(tf.reduce_sum(tf.reshape(y, [-1, 5, 2]) ** 2, axis=-1))
    mag_weight = 1.0 + gt_mag
    mag_weight_expanded = tf.repeat(mag_weight, 2, axis=-1)
    combined_w = w * mag_weight_expanded
    return tf.reduce_sum(huber * combined_w) / tf.maximum(tf.reduce_sum(combined_w), 1.0)


# ─── Resume helper ────────────────────────────────────────────────────────────

def infer_resume_state(log_path: Path) -> Tuple[int, float, int, int]:
    if not log_path.exists():
        return 0, float("inf"), 0, 0
    ep_pat = re.compile(r"^Ep\s+(\d+)/\d+\s+train=([-+0-9.eE]+)\s+val=([-+0-9.eE]+)")
    last_epoch, best_val, best_epoch, no_improve = 0, float("inf"), 0, 0
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = ep_pat.match(line.strip())
            if not m:
                continue
            ep, val = int(m.group(1)), float(m.group(3))
            last_epoch = max(last_epoch, ep)
            if val < best_val:
                best_val, best_epoch, no_improve = val, ep, 0
            else:
                no_improve += 1
    return last_epoch, best_val, best_epoch, no_improve


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train consecutive-frame contourwise shift model.")
    p.add_argument("--clean-root", type=Path, default=Path("../CLEAN_RESULTS"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frame-skip", type=int, default=1, help="Temporal gap between frame pairs (1=consecutive, 5=5s gap)")
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--val-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm", choices=["batch", "layer"], default="layer")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--model", choices=["gap", "attention"], default="gap", help="Head type: gap=GlobalAvgPool, attention=per-mask spatial attention")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-dir", type=Path, default=Path("./run_consecutive"))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"
    log_f = open(str(log_path), "a")

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log("=" * 60)
    log("Consecutive-frame contourwise shift training")
    log(f"  mask_types={MASK_TYPES}")
    log(f"  epochs={args.epochs}, bs={args.batch_size}, skip={args.frame_skip}, steps/ep={args.steps_per_epoch}")
    log(f"  lr={args.lr}, patience={args.patience}, norm={args.norm}")
    log("=" * 60)

    # Discover data
    log("Discovering training sequences...")
    train_seqs = discover_sequences(args.clean_root, TRAIN_PATIENTS)
    train_pairs = build_pair_index(train_seqs, args.frame_skip)
    log(f"  Train: {len(train_seqs)} sequences, {len(train_pairs)} pairs (skip={args.frame_skip})")

    log("Discovering validation sequences...")
    val_seqs = discover_sequences(args.clean_root, VAL_PATIENTS)
    val_pairs = build_pair_index(val_seqs, args.frame_skip)
    log(f"  Val: {len(val_seqs)} sequences, {len(val_pairs)} pairs (skip={args.frame_skip})")

    log("Discovering test sequences...")
    test_seqs = discover_sequences(args.clean_root, TEST_PATIENTS)
    test_pairs = build_pair_index(test_seqs, args.frame_skip)
    log(f"  Test: {len(test_seqs)} sequences, {len(test_pairs)} pairs (skip={args.frame_skip})")

    if not train_pairs:
        raise RuntimeError("No training pairs found.")

    # Build model
    tf.keras.backend.clear_session()
    if args.model == "attention":
        model = build_flexunet_attention(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 2), n_masks=len(MASK_TYPES), norm=args.norm)
    else:
        model = build_flexunet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 2), output_dims=OUTPUT_DIMS, norm=args.norm)
    log(f"Model: {args.model}, params: {model.count_params():,}")

    # Compute actual steps per epoch for LR schedule
    if args.steps_per_epoch > 0:
        effective_steps = args.steps_per_epoch
    else:
        effective_steps = max(1, len(train_pairs) // args.batch_size)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        args.lr,
        decay_steps=max(1, args.epochs * effective_steps),
        alpha=args.lr_min / args.lr,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Resume
    start_epoch, best_val, best_epoch, no_improve = 0, float("inf"), 0, 0
    ckpt_last = run_dir / "checkpoint_last.weights.h5"
    if args.resume and ckpt_last.exists():
        model.load_weights(str(ckpt_last))
        start_epoch, best_val, best_epoch, no_improve = infer_resume_state(log_path)
        log(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f} @ ep{best_epoch}")

    # Save config
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["train_patients"] = TRAIN_PATIENTS
    config["val_patients"] = VAL_PATIENTS
    config["test_patients"] = TEST_PATIENTS
    config["mask_types"] = MASK_TYPES
    config["train_pairs"] = len(train_pairs)
    config["val_pairs"] = len(val_pairs)
    config["test_pairs"] = len(test_pairs)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    rng = np.random.default_rng(args.seed)

    # Select loss functions
    train_step = train_step_mse if args.loss == "mse" else train_step_huber
    val_step = val_step_mse if args.loss == "mse" else val_step_huber
    log(f"Loss function: {args.loss} (magnitude-weighted)")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # Shuffle training pairs each epoch for full-pass training
        train_order = rng.permutation(len(train_pairs))
        train_losses = []

        if args.steps_per_epoch > 0:
            # Fixed steps mode: random sampling
            for step in range(args.steps_per_epoch):
                batch_idx = rng.integers(0, len(train_pairs), size=args.batch_size)
                xb, yb, wb = make_batch(train_seqs, train_pairs, batch_idx, args.frame_skip)
                loss = train_step(model, optimizer, xb, yb, wb)
                train_losses.append(float(loss))
        else:
            # Full-pass mode: iterate through all pairs once
            for i in range(0, len(train_order), args.batch_size):
                batch_idx = train_order[i:i + args.batch_size]
                if len(batch_idx) < 2:
                    continue
                xb, yb, wb = make_batch(train_seqs, train_pairs, batch_idx, args.frame_skip)
                loss = train_step(model, optimizer, xb, yb, wb)
                train_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))

        # Deterministic full validation pass
        val_losses = []
        if val_pairs:
            if args.val_steps > 0:
                # Fixed steps with seeded rng for reproducibility
                val_rng = np.random.default_rng(args.seed + epoch)
                for step in range(args.val_steps):
                    batch_idx = val_rng.integers(0, len(val_pairs), size=args.batch_size)
                    xb, yb, wb = make_batch(val_seqs, val_pairs, batch_idx, args.frame_skip)
                    vl = val_step(model, xb, yb, wb)
                    val_losses.append(float(vl))
            else:
                # Full val pass: evaluate all pairs
                for i in range(0, len(val_pairs), args.batch_size):
                    batch_idx = np.arange(i, min(i + args.batch_size, len(val_pairs)))
                    xb, yb, wb = make_batch(val_seqs, val_pairs, batch_idx, args.frame_skip)
                    vl = val_step(model, xb, yb, wb)
                    val_losses.append(float(vl))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        elapsed = time.time() - t0
        log(f"Ep {epoch}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.1f}s)")

        # Checkpoint
        model.save_weights(str(ckpt_last))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            model.save_weights(str(run_dir / "checkpoint_best.weights.h5"))
            log(f"  ** New best val={best_val:.6f} @ ep{epoch}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            log(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    log(f"\nTraining complete. Best val={best_val:.6f} @ epoch {best_epoch}")
    log(f"Checkpoints in: {run_dir}")
    log_f.close()


if __name__ == "__main__":
    main()

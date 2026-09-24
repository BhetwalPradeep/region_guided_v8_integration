#!/usr/bin/env python3
"""
Models for consecutive-frame shift regression.

`build_corrnet` is the new architecture. The previous FlexUNet regressed motion
by GlobalAveragePooling a U-Net decoder, which averages away the very thing
motion estimation needs: *where* each feature sits in each frame. All four
previous runs collapsed to predicting ~0 (Euclidean error equal to the ground
truth magnitude).

CorrNet instead follows FlowNet-C: encode both frames with a shared (tied)
encoder, then form an explicit cost volume by correlating frame-t features
against a neighbourhood of frame-t+1 features. Displacement then appears
directly as the position of the peak in that volume, which a small conv head can
read off. Per-mask spatial attention pools the volume separately for each body
part, so a part can be localised by the region it occupies.

All models take the same input, [H, W, 2] = (frame_t, frame_t+1), and emit 20
values: [:10] centroid-target dx,dy per mask, [10:] dice-optimal dx,dy per mask.
"""
from __future__ import annotations

# Imported first, and at module level: train_consecutive calls
# set_memory_growth at import time, which TF rejects once the device context has
# been initialised by building a model. Importing it up front keeps the previous
# architectures usable for the ablation.
from train_consecutive import (build_flexunet, build_flexunet_attention,
                                build_region_guided_attention)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_MASKS = 5
OUT_DIMS = N_MASKS * 2 * 2  # 5 masks x (dx,dy) x 2 heads


def _norm(x, norm: str, name: str):
    if norm == "batch":
        return layers.BatchNormalization(name=name)(x)
    return layers.LayerNormalization(axis=-1, name=name)(x)


def _conv_block(x, filters, norm, prefix, n=2, stride=1):
    for i in range(n):
        x = layers.Conv2D(filters, 3, strides=stride if i == 0 else 1,
                          padding="same", name=f"{prefix}_c{i}")(x)
        x = _norm(x, norm, f"{prefix}_n{i}")
        x = layers.Activation("relu", name=f"{prefix}_a{i}")(x)
    return x


class Correlation(layers.Layer):
    """Cost volume between two feature maps.

    out[b, y, x, k] = mean_c f0[b, y, x, c] * f1[b, y + dy_k, x + dx_k, c]

    for every (dx_k, dy_k) in a (2*max_disp+1)^2 neighbourhood. At 1/8
    resolution max_disp=8 covers +/-64 px of input-scale displacement, which
    spans the real motion in this dataset (p99 ~53 px).
    """

    def __init__(self, max_disp: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_disp = max_disp
        self.k = 2 * max_disp + 1

    def call(self, inputs):
        f0, f1 = inputs
        d = self.max_disp
        # One displacement at a time. tf.image.extract_patches would build a
        # [B, H, W, k*k*C] tensor -- 4.8 GB at batch 32, and past 2^31 elements
        # at batch 64, where its gradient overflows int32 and aborts the process.
        # Slicing keeps peak memory at k*k * [B, H, W] instead.
        padded = tf.pad(f1, [[0, 0], [d, d], [d, d], [0, 0]])
        h, w = tf.shape(f0)[1], tf.shape(f0)[2]
        outs = []
        for dy in range(self.k):
            for dx in range(self.k):
                shifted = tf.slice(padded, [0, dy, dx, 0], [-1, h, w, -1])
                outs.append(tf.reduce_mean(f0 * shifted, axis=-1))
        return tf.stack(outs, axis=-1)

    def compute_output_shape(self, input_shape):
        s = input_shape[0]
        return (s[0], s[1], s[2], self.k * self.k)

    def get_config(self):
        return {**super().get_config(), "max_disp": self.max_disp}


def _shared_encoder(norm: str, base: int = 32):
    """Encoder applied to each frame separately with tied weights."""
    inp = keras.Input(shape=(None, None, 1), name="enc_in")
    x = _conv_block(inp, base, norm, "enc0", n=2, stride=2)          # 1/2
    x = _conv_block(x, base * 2, norm, "enc1", n=2, stride=2)        # 1/4
    x = _conv_block(x, base * 4, norm, "enc2", n=2, stride=2)        # 1/8
    return keras.Model(inp, x, name="shared_encoder")


def _shared_encoder_ms(norm: str, base: int = 32):
    """Multi-scale encoder returning features at 1/4 and 1/8 resolution."""
    inp = keras.Input(shape=(None, None, 1), name="enc_in")
    x = _conv_block(inp, base, norm, "enc0", n=2, stride=2)          # 1/2
    x4 = _conv_block(x, base * 2, norm, "enc1", n=2, stride=2)       # 1/4
    x8 = _conv_block(x4, base * 4, norm, "enc2", n=2, stride=2)      # 1/8
    return keras.Model(inp, [x4, x8], name="shared_encoder_ms")


def build_corrnet(
    input_shape=(256, 256, 2),
    norm: str = "layer",
    max_disp: int = 8,
    base: int = 32,
    n_masks: int = N_MASKS,
) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="input")
    f_t = layers.Lambda(lambda t: t[..., 0:1], name="split_t")(inp)
    f_t1 = layers.Lambda(lambda t: t[..., 1:2], name="split_t1")(inp)

    encoder = _shared_encoder(norm, base)
    e0 = encoder(f_t)
    e1 = encoder(f_t1)

    # Project both frames through the *same* 1x1 conv before correlating: the
    # cost volume is only meaningful if both sides live in one feature space,
    # and fewer channels makes the k*k inner products much cheaper.
    corr_proj = layers.Conv2D(64, 1, padding="same", name="corr_proj")
    corr = Correlation(max_disp=max_disp, name="correlation")([corr_proj(e0), corr_proj(e1)])

    # Keep appearance features alongside the cost volume: the volume says how
    # things moved, the features say which body part is there.
    ctx = layers.Conv2D(base * 2, 1, padding="same", name="ctx_proj")(e0)
    x = layers.Concatenate(name="corr_ctx")([corr, ctx])

    x = _conv_block(x, 256, norm, "head0", n=2)
    x = _conv_block(x, 256, norm, "head1", n=2, stride=2)   # 1/16
    x = _conv_block(x, 256, norm, "head2", n=2)

    # Per-mask spatial attention: each part pools the volume over its own region.
    attn = layers.Conv2D(n_masks, 1, padding="same", name="attn_logits")(x)
    attn = layers.Reshape((-1, n_masks), name="attn_flat")(attn)
    attn = layers.Softmax(axis=1, name="attn_softmax")(attn)
    feat = layers.Reshape((-1, 256), name="feat_flat")(x)

    outs = []
    for mi in range(n_masks):
        a = layers.Lambda(lambda t, i=mi: t[:, :, i:i + 1], name=f"attn_{mi}")(attn)
        pooled = layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name=f"pool_{mi}"
        )([feat, a])
        h = layers.Dense(128, activation="relu", name=f"fc_{mi}")(pooled)
        # 4 values per mask: (dx, dy) for the centroid head and the dice head
        outs.append(layers.Dense(4, name=f"out_{mi}")(h))

    stacked = layers.Concatenate(name="concat_masks")(outs)          # [B, n_masks*4]
    # Reorder to [centroid(10) | dice_opt(10)]
    out = layers.Lambda(
        lambda t: tf.concat(
            [tf.reshape(t, [-1, n_masks, 4])[:, :, 0:2],
             tf.reshape(t, [-1, n_masks, 4])[:, :, 2:4]], axis=1
        ),
        name="reorder",
    )(stacked)
    out = layers.Reshape((n_masks * 4,), name="output")(out)
    return keras.Model(inp, out, name="corrnet")


def build_corrnet_v2(
    input_shape=(256, 256, 2),
    norm: str = "layer",
    max_disp: int = 8,
    base: int = 32,
    n_masks: int = N_MASKS,
) -> keras.Model:
    """Multi-scale corrnet: correlates at 1/8 (coarse) and 1/4 (fine)."""
    inp = keras.Input(shape=input_shape, name="input")
    f_t = layers.Lambda(lambda t: t[..., 0:1], name="split_t")(inp)
    f_t1 = layers.Lambda(lambda t: t[..., 1:2], name="split_t1")(inp)

    encoder = _shared_encoder_ms(norm, base)
    e0_4, e0_8 = encoder(f_t)
    e1_4, e1_8 = encoder(f_t1)

    # Coarse correlation at 1/8 (max_disp covers ±64 px)
    corr_proj_c = layers.Conv2D(64, 1, padding="same", name="corr_proj_c")
    corr_c = Correlation(max_disp=max_disp, name="corr_coarse")(
        [corr_proj_c(e0_8), corr_proj_c(e1_8)])

    # Fine correlation at 1/4 (max_disp=4 covers ±16 px, precise for small shifts)
    fine_disp = min(max_disp // 2, 4)
    corr_proj_f = layers.Conv2D(32, 1, padding="same", name="corr_proj_f")
    corr_f = Correlation(max_disp=fine_disp, name="corr_fine")(
        [corr_proj_f(e0_4), corr_proj_f(e1_4)])
    # Compress and downsample fine volume to match coarse spatial dims (1/8)
    corr_f = _conv_block(corr_f, 64, norm, "fine_compress", n=1, stride=2)

    ctx = layers.Conv2D(base * 2, 1, padding="same", name="ctx_proj")(e0_8)
    x = layers.Concatenate(name="corr_ctx")([corr_c, corr_f, ctx])

    x = _conv_block(x, 256, norm, "head0", n=2)
    x = _conv_block(x, 256, norm, "head1", n=2, stride=2)   # 1/16
    x = _conv_block(x, 256, norm, "head2", n=2)

    attn = layers.Conv2D(n_masks, 1, padding="same", name="attn_logits")(x)
    attn = layers.Reshape((-1, n_masks), name="attn_flat")(attn)
    attn = layers.Softmax(axis=1, name="attn_softmax")(attn)
    feat = layers.Reshape((-1, 256), name="feat_flat")(x)

    outs = []
    for mi in range(n_masks):
        a = layers.Lambda(lambda t, i=mi: t[:, :, i:i + 1], name=f"attn_{mi}")(attn)
        pooled = layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name=f"pool_{mi}"
        )([feat, a])
        h = layers.Dense(128, activation="relu", name=f"fc_{mi}")(pooled)
        outs.append(layers.Dense(4, name=f"out_{mi}")(h))

    stacked = layers.Concatenate(name="concat_masks")(outs)
    out = layers.Lambda(
        lambda t: tf.concat(
            [tf.reshape(t, [-1, n_masks, 4])[:, :, 0:2],
             tf.reshape(t, [-1, n_masks, 4])[:, :, 2:4]], axis=1
        ),
        name="reorder",
    )(stacked)
    out = layers.Reshape((n_masks * 4,), name="output")(out)
    return keras.Model(inp, out, name="corrnet_v2")


class CrossAttentionBlock(layers.Layer):
    """Bidirectional cross-attention: each frame attends to the other."""

    def __init__(self, dim: int, n_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_heads = n_heads
        self.cross_attn_01 = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=dim // n_heads, name="ca_0to1")
        self.cross_attn_10 = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=dim // n_heads, name="ca_1to0")
        self.norm0 = layers.LayerNormalization(axis=-1, name="ln0")
        self.norm1 = layers.LayerNormalization(axis=-1, name="ln1")
        self.ffn0 = keras.Sequential([
            layers.Dense(dim * 2, activation="gelu"), layers.Dense(dim),
        ], name="ffn0")
        self.ffn1 = keras.Sequential([
            layers.Dense(dim * 2, activation="gelu"), layers.Dense(dim),
        ], name="ffn1")
        self.norm_ff0 = layers.LayerNormalization(axis=-1, name="ln_ff0")
        self.norm_ff1 = layers.LayerNormalization(axis=-1, name="ln_ff1")

    def call(self, inputs):
        f0, f1 = inputs
        # cross-attention: f0 queries f1, f1 queries f0
        f0 = f0 + self.cross_attn_01(query=self.norm0(f0), key=f1, value=f1)
        f1 = f1 + self.cross_attn_10(query=self.norm1(f1), key=f0, value=f0)
        f0 = f0 + self.ffn0(self.norm_ff0(f0))
        f1 = f1 + self.ffn1(self.norm_ff1(f1))
        return f0, f1

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "n_heads": self.n_heads}


def build_crossattn(
    input_shape=(256, 256, 2),
    norm: str = "layer",
    base: int = 48,
    n_heads: int = 4,
    n_cross_layers: int = 3,
    n_masks: int = N_MASKS,
    **_kwargs,
) -> keras.Model:
    """Siamese CNN encoder + cross-attention + per-mask pooling."""
    inp = keras.Input(shape=input_shape, name="input")
    f_t = layers.Lambda(lambda t: t[..., 0:1], name="split_t")(inp)
    f_t1 = layers.Lambda(lambda t: t[..., 1:2], name="split_t1")(inp)

    # Shared CNN encoder to 1/8 resolution
    encoder = _shared_encoder(norm, base)
    e0 = encoder(f_t)    # [B, H/8, W/8, base*4]
    e1 = encoder(f_t1)

    feat_dim = base * 4  # 192
    # Flatten spatial dims for attention: [B, H/8*W/8, C]
    e0_flat = layers.Reshape((-1, feat_dim), name="flat_t")(e0)
    e1_flat = layers.Reshape((-1, feat_dim), name="flat_t1")(e1)

    # Bidirectional cross-attention layers
    for i in range(n_cross_layers):
        ca = CrossAttentionBlock(feat_dim, n_heads, name=f"cross_{i}")
        e0_flat, e1_flat = ca([e0_flat, e1_flat])

    # Motion = difference after cross-attention (each frame's features now
    # contain information about the other frame's content at corresponding
    # locations, so their difference encodes displacement)
    diff = layers.Subtract(name="motion_diff")([e0_flat, e1_flat])
    cat = layers.Concatenate(name="motion_cat")([e0_flat, diff])

    # Refine with a small conv head (reshape back to spatial)
    sp_h = input_shape[0] // 8
    sp_w = input_shape[1] // 8
    x = layers.Reshape((sp_h, sp_w, feat_dim * 2), name="to_spatial")(cat)
    x = _conv_block(x, 256, norm, "ref0", n=2)
    x = _conv_block(x, 256, norm, "ref1", n=2, stride=2)   # 1/16
    x = _conv_block(x, 256, norm, "ref2", n=2)

    # Per-mask spatial attention pooling
    attn = layers.Conv2D(n_masks, 1, padding="same", name="attn_logits")(x)
    attn = layers.Reshape((-1, n_masks), name="attn_flat")(attn)
    attn = layers.Softmax(axis=1, name="attn_softmax")(attn)
    feat = layers.Reshape((-1, 256), name="feat_flat")(x)

    outs = []
    for mi in range(n_masks):
        a = layers.Lambda(lambda t, i=mi: t[:, :, i:i + 1], name=f"attn_{mi}")(attn)
        pooled = layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name=f"pool_{mi}"
        )([feat, a])
        h = layers.Dense(128, activation="relu", name=f"fc_{mi}")(pooled)
        outs.append(layers.Dense(4, name=f"out_{mi}")(h))

    stacked = layers.Concatenate(name="concat_masks")(outs)
    out = layers.Lambda(
        lambda t: tf.concat(
            [tf.reshape(t, [-1, n_masks, 4])[:, :, 0:2],
             tf.reshape(t, [-1, n_masks, 4])[:, :, 2:4]], axis=1
        ),
        name="reorder",
    )(stacked)
    out = layers.Reshape((n_masks * 4,), name="output")(out)
    return keras.Model(inp, out, name="crossattn")


def _mask_pool_head(x, n_masks, head_dim=256, prefix=""):
    """Shared per-mask attention pooling + MLP head. Returns [B, n_masks*4]."""
    attn = layers.Conv2D(n_masks, 1, padding="same", name=f"{prefix}attn_logits")(x)
    attn = layers.Reshape((-1, n_masks), name=f"{prefix}attn_flat")(attn)
    attn = layers.Softmax(axis=1, name=f"{prefix}attn_softmax")(attn)
    feat = layers.Reshape((-1, head_dim), name=f"{prefix}feat_flat")(x)
    outs = []
    for mi in range(n_masks):
        a = layers.Lambda(lambda t, i=mi: t[:, :, i:i + 1], name=f"{prefix}attn_{mi}")(attn)
        pooled = layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name=f"{prefix}pool_{mi}"
        )([feat, a])
        h = layers.Dense(128, activation="relu", name=f"{prefix}fc_{mi}")(pooled)
        outs.append(layers.Dense(4, name=f"{prefix}out_{mi}")(h))
    stacked = layers.Concatenate(name=f"{prefix}concat_masks")(outs)
    out = layers.Lambda(
        lambda t: tf.concat(
            [tf.reshape(t, [-1, n_masks, 4])[:, :, 0:2],
             tf.reshape(t, [-1, n_masks, 4])[:, :, 2:4]], axis=1
        ),
        name=f"{prefix}reorder",
    )(stacked)
    return layers.Reshape((n_masks * 4,), name=f"{prefix}output")(out)


# ── ConvNeXt Block ────────────────────────────────────────────────────────────

class ConvNeXtBlock(layers.Layer):
    """ConvNeXt v1 block: depthwise conv → LN → 1×1 → GELU → 1×1."""

    def __init__(self, dim, expansion=4, **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(7, padding="same")
        self.ln = layers.LayerNormalization(axis=-1)
        self.pw1 = layers.Dense(dim * expansion)
        self.act = layers.Activation("gelu")
        self.pw2 = layers.Dense(dim)

    def call(self, x):
        shortcut = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + shortcut


def build_convnext_attn(
    input_shape=(256, 256, 2),
    dims=(48, 96, 192, 384),
    depths=(2, 2, 4, 2),
    n_masks: int = N_MASKS,
    **_kwargs,
) -> keras.Model:
    """ConvNeXt encoder + U-Net decoder + per-mask attention pooling."""
    inp = keras.Input(shape=input_shape, name="input")

    # Stem: 4×4 non-overlapping conv (like ConvNeXt/Swin)
    x = layers.Conv2D(dims[0], 4, strides=4, padding="same", name="stem")(inp)
    x = layers.LayerNormalization(axis=-1, name="stem_ln")(x)  # 1/4

    skips = []
    for stage, (dim, depth) in enumerate(zip(dims, depths)):
        if stage > 0:
            # Downsample: LN + 2×2 strided conv
            x = layers.LayerNormalization(axis=-1, name=f"ds{stage}_ln")(x)
            x = layers.Conv2D(dim, 2, strides=2, padding="same", name=f"ds{stage}")(x)
        for b in range(depth):
            x = ConvNeXtBlock(dim, name=f"cnx{stage}_{b}")(x)
        skips.append(x)

    # Decoder with skip connections
    for stage in range(len(dims) - 2, -1, -1):
        x = layers.UpSampling2D(2, name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"skip{stage}")([x, skips[stage]])
        x = layers.Conv2D(dims[stage], 1, padding="same", name=f"dec{stage}_proj")(x)
        x = ConvNeXtBlock(dims[stage], name=f"dec{stage}_cnx")(x)

    # Final conv to head_dim, then pool
    x = layers.Conv2D(256, 1, padding="same", name="head_proj")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", name="head_ds")(x)
    x = layers.LayerNormalization(axis=-1, name="head_ln")(x)
    x = layers.Activation("gelu", name="head_act")(x)

    out = _mask_pool_head(x, n_masks, head_dim=256, prefix="")
    return keras.Model(inp, out, name="convnext_attn")


# ── Swin Transformer Block ───────────────────────────────────────────────────

class WindowAttention(layers.Layer):
    """Window-based multi-head self-attention."""

    def __init__(self, dim, window_size, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=dim // n_heads)
        self.ln = layers.LayerNormalization(axis=-1)

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        ws = self.window_size
        nH, nW = H // ws, W // ws
        # Partition into windows: [B*nH*nW, ws*ws, C]
        x_win = tf.reshape(x, [B, nH, ws, nW, ws, C])
        x_win = tf.transpose(x_win, [0, 1, 3, 2, 4, 5])
        x_win = tf.reshape(x_win, [-1, ws * ws, C])
        x_norm = self.ln(x_win)
        attn_out = self.mha(x_norm, x_norm)
        x_win = x_win + attn_out
        # Reverse partition
        x_win = tf.reshape(x_win, [B, nH, nW, ws, ws, C])
        x_win = tf.transpose(x_win, [0, 1, 3, 2, 4, 5])
        return tf.reshape(x_win, [B, H, W, C])

    def get_config(self):
        return {**super().get_config(), "dim": self.dim,
                "window_size": self.window_size, "n_heads": self.n_heads}


class SwinBlock(layers.Layer):
    """Swin Transformer block: window attention + FFN."""

    def __init__(self, dim, window_size=8, n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.wa = WindowAttention(dim, window_size, n_heads)
        self.ln = layers.LayerNormalization(axis=-1)
        self.ffn = keras.Sequential([
            layers.Dense(dim * 4, activation="gelu"), layers.Dense(dim)])

    def call(self, x):
        x = x + self.wa(x)
        x = x + self.ffn(self.ln(x))
        return x


def build_swinunet(
    input_shape=(256, 256, 2),
    dims=(48, 96, 192, 384),
    depths=(2, 2, 2, 2),
    window_size=8,
    n_masks: int = N_MASKS,
    **_kwargs,
) -> keras.Model:
    """Swin Transformer encoder + conv decoder + per-mask attention pooling."""
    inp = keras.Input(shape=input_shape, name="input")

    # Stem: patch embedding 4×4
    x = layers.Conv2D(dims[0], 4, strides=4, padding="same", name="stem")(inp)
    x = layers.LayerNormalization(axis=-1, name="stem_ln")(x)  # 64×64

    skips = []
    for stage, (dim, depth) in enumerate(zip(dims, depths)):
        if stage > 0:
            x = layers.LayerNormalization(axis=-1, name=f"ds{stage}_ln")(x)
            x = layers.Conv2D(dim, 2, strides=2, padding="same", name=f"ds{stage}")(x)
        for b in range(depth):
            x = SwinBlock(dim, window_size=window_size, n_heads=dim // 24,
                          name=f"swin{stage}_{b}")(x)
        skips.append(x)

    # Conv decoder with skip connections
    for stage in range(len(dims) - 2, -1, -1):
        x = layers.UpSampling2D(2, name=f"up{stage}")(x)
        x = layers.Concatenate(name=f"skip{stage}")([x, skips[stage]])
        x = layers.Conv2D(dims[stage], 3, padding="same", activation="gelu",
                          name=f"dec{stage}_c0")(x)
        x = layers.LayerNormalization(axis=-1, name=f"dec{stage}_ln")(x)
        x = layers.Conv2D(dims[stage], 3, padding="same", activation="gelu",
                          name=f"dec{stage}_c1")(x)

    x = layers.Conv2D(256, 1, padding="same", name="head_proj")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", name="head_ds")(x)
    x = layers.LayerNormalization(axis=-1, name="head_ln")(x)
    x = layers.Activation("gelu", name="head_act")(x)

    out = _mask_pool_head(x, n_masks, head_dim=256, prefix="")
    return keras.Model(inp, out, name="swinunet")


def get_model(name: str, norm: str = "layer", input_shape=(256, 256, 2),
              max_disp: int = 8, n_masks: int = N_MASKS) -> keras.Model:
    """Dispatcher. 'flexunet'/'attention' are the previous architectures, kept
    unchanged for ablation; both are widened to 20 outputs for the two heads."""
    if name == "corrnet":
        return build_corrnet(input_shape=input_shape, norm=norm, max_disp=max_disp)
    if name == "corrnet_v2":
        return build_corrnet_v2(input_shape=input_shape, norm=norm, max_disp=max_disp)
    if name == "crossattn":
        return build_crossattn(input_shape=input_shape, norm=norm)
    if name == "convnext_attn":
        return build_convnext_attn(input_shape=input_shape)
    if name == "swinunet":
        return build_swinunet(input_shape=input_shape)
    if name == "flexunet":
        return build_flexunet(input_shape=input_shape, output_dims=OUT_DIMS, norm=norm)
    if name == "attention":
        # n_masks=10 gives 20 outputs: one attention map per (mask, head).
        return build_flexunet_attention(input_shape=input_shape, n_masks=10, norm=norm)
    if name == "attention_single":
        # 10 outputs: one attention map per mask, single dice-optimal head.
        return build_flexunet_attention(input_shape=input_shape, n_masks=n_masks, norm=norm)
    if name == "region_guided_attention":
        return build_region_guided_attention(input_shape=input_shape, n_masks=n_masks, norm=norm)
    raise ValueError(f"unknown model: {name}")


if __name__ == "__main__":
    for n in ("corrnet", "corrnet_v2", "crossattn", "convnext_attn", "swinunet",
              "flexunet", "attention"):
        m = get_model(n)
        print(f"{n:<14} params={m.count_params():>12,}  out={m.output_shape}")

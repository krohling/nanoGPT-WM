"""Evaluation and visualization for nano-world-model.

Helpers (save_grid, to_uint8, codebook_stats) are imported by the train
scripts; the CLI subcommands for dynamics evaluation live in main() below.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import data as data_mod
from model import VQVAE, FRAME_VOCAB


def to_uint8(x):
    """[B,3,64,64] float in [0,1] -> [B,64,64,3] uint8."""
    return (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def save_grid(frames, path, ncol=8, pad=2, scale=2):
    """frames: uint8 [N,H,W,3] (any tile size) -> one PNG grid."""
    n = len(frames)
    nrow = (n + ncol - 1) // ncol
    th, tw = frames[0].shape[:2]
    H, W = th * scale, tw * scale
    canvas = np.full((nrow * (H + pad) + pad, ncol * (W + pad) + pad, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        img = np.asarray(Image.fromarray(f).resize((W, H), Image.NEAREST))
        r, c = divmod(i, ncol)
        canvas[pad + r * (H + pad): pad + r * (H + pad) + H,
               pad + c * (W + pad): pad + c * (W + pad) + W] = img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def codebook_stats(idx_batches):
    """idx_batches: iterable of LongTensor [B,K] -> (usage counts [512], perplexity)."""
    usage = np.zeros(FRAME_VOCAB, dtype=np.int64)
    for idx in idx_batches:
        u, c = np.unique(idx.cpu().numpy(), return_counts=True)
        usage[u] += c
    p = usage / max(usage.sum(), 1)
    nz = p[p > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum()))
    return usage, perplexity

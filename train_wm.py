"""Stage 2: train the world-model GPT on tokenized frames.

The dataset is turned into one long token stream per window:
    [z0_1..z0_K, a0, z1_1..z1_K, a1, ..., z_{T-1}]
Each action sits BETWEEN the frame where it was taken and the frame it
produces, so the token immediately before every frame block is the action
that causes that frame — generate_frame appends an action and samples the
consequence. Training is exactly language modeling (next-token
cross-entropy), except positions whose *target* is an action token are
ignored — the model predicts the world, not the player's mind.
"""
import numpy as np

from model import FRAME_VOCAB


def interleave(tokens, actions):
    """tokens [T,K] int, actions [T] int -> [T*K + T-1] int64.

    Layout: z0 a0 z1 a1 ... z_{T-1}. actions[T-1] is dropped — it would
    produce frame T, which lies outside this window.
    """
    T, K = tokens.shape
    seq = np.empty((T, K + 1), dtype=np.int64)
    seq[:, :K] = tokens
    seq[:-1, K] = np.asarray(actions[: T - 1], dtype=np.int64) + FRAME_VOCAB
    return seq.reshape(-1)[: T * K + T - 1]


def make_targets(seq):
    """Next-token targets for seq[:-1]; positions predicting an action -> -1."""
    tgt = seq[1:].astype(np.int64).copy()
    tgt[tgt >= FRAME_VOCAB] = -1
    return tgt

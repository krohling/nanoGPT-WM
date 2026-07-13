"""Data loading for nano-world-model.

On-disk format (nanoGPT idiom — flat memmapped binaries):
  <root>/<split>/frames.bin   uint8, N*64*64*3 bytes
  <root>/<split>/actions.bin  uint8, N bytes   (actions[t] taken at frames[t] -> frames[t+1])
  <root>/<split>/dones.bin    uint8, N bytes   (1 = frames[t] is the last frame of an episode)
  <root>/<split>/meta.json
  <root>/test_episodes/ep_###.npz  (complete held-out episodes: frames, actions)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_SHAPE = (64, 64, 3)

DEFAULT_DATASET_REPO = os.environ.get("NWM_DATASET_REPO",
                                      "kevin510/nano-world-model-chaser")


def download(root="data/chaser", repo_id=None, episodes_only=False):
    """Fetch the dataset from HuggingFace into `root`. Returns the local path.

    episodes_only=True grabs just the 60 held-out test episodes (~24 MB) —
    enough for rollout evals and playing in the dream with the published
    checkpoints. The full download (~5.4 GB) is only needed for training.
    """
    from huggingface_hub import snapshot_download
    repo_id = repo_id or DEFAULT_DATASET_REPO
    if not repo_id:
        raise SystemExit("Set NWM_DATASET_REPO or pass repo_id (see README).")
    patterns = ["test_episodes/*"] if episodes_only else None
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=root,
                      allow_patterns=patterns)
    return Path(root)


def load_split(root, split):
    """Memmap one split. Returns (frames [N,64,64,3] u8, actions [N] u8, dones [N] u8)."""
    d = Path(root) / split
    meta = json.loads((d / "meta.json").read_text())
    n = meta["num_frames"]
    frames = np.memmap(d / "frames.bin", dtype=np.uint8, mode="r", shape=(n, *FRAME_SHAPE))
    actions = np.memmap(d / "actions.bin", dtype=np.uint8, mode="r", shape=(n,))
    dones = np.memmap(d / "dones.bin", dtype=np.uint8, mode="r", shape=(n,))
    return frames, actions, dones


def valid_starts(dones, seq_len):
    """Start indices i such that frames[i : i+seq_len] stay within one episode.

    A done flag on the window's FINAL frame is fine (the episode ends exactly
    there); a done anywhere earlier means the window straddles two episodes.
    """
    dones = np.asarray(dones, dtype=np.uint8)
    n = len(dones)
    # cumulative count of dones lets us test "any done in dones[i : i+L-2]" in O(1)
    cum = np.concatenate([[0], np.cumsum(dones)])
    starts = np.arange(n - seq_len + 1)
    inner = cum[starts + seq_len - 1] - cum[starts]  # dones within frames[i .. i+L-2]
    return starts[inner == 0].astype(np.int64)


def load_episode(path):
    with np.load(path) as z:
        return {"frames": z["frames"], "actions": z["actions"]}


def list_episodes(root):
    return sorted((Path(root) / "test_episodes").glob("ep_*.npz"))


class FrameDataset(Dataset):
    """Individual frames for tokenizer training. Returns float32 [3,64,64] in [0,1]."""

    def __init__(self, root, split):
        self.frames, _, _ = load_split(root, split)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.frames[i]))  # copy out of the memmap
        return x.permute(2, 0, 1).float() / 255.0

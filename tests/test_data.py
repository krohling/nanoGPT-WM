import json

import numpy as np
import pytest
import torch

import data


def make_split(root, split, n=100, seed=0):
    """Write a tiny synthetic split in the real on-disk format."""
    rng = np.random.default_rng(seed)
    d = root / split
    d.mkdir(parents=True)
    frames = rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)
    actions = rng.integers(0, 15, size=n).astype(np.uint8)
    dones = np.zeros(n, dtype=np.uint8)
    dones[49] = 1  # episode boundary mid-split
    dones[n - 1] = 1
    frames.tofile(d / "frames.bin")
    actions.tofile(d / "actions.bin")
    dones.tofile(d / "dones.bin")
    (d / "meta.json").write_text(json.dumps({
        "num_frames": n, "frame_shape": [64, 64, 3], "num_actions": 15,
        "action_convention": "actions[t] is taken at frames[t] and produces frames[t+1]",
    }))
    return frames, actions, dones


def test_load_split_shapes(tmp_path):
    frames, actions, dones = make_split(tmp_path, "train")
    f, a, d = data.load_split(tmp_path, "train")
    assert f.shape == (100, 64, 64, 3) and f.dtype == np.uint8
    assert a.shape == (100,) and d.shape == (100,)
    assert np.array_equal(f[3], frames[3])


def test_frame_dataset(tmp_path):
    make_split(tmp_path, "train")
    ds = data.FrameDataset(tmp_path, "train")
    assert len(ds) == 100
    x = ds[7]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 64, 64)
    assert x.dtype == torch.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_valid_starts_respects_episode_boundaries(tmp_path):
    _, _, dones = make_split(tmp_path, "train")
    starts = data.valid_starts(dones, seq_len=16)
    # window may END on a done frame but not contain one earlier:
    # dones[49]=1 -> a window whose final frame is 49 is fine
    assert 34 in starts            # frames 34..49, done is the final frame: OK
    for s in range(35, 50):
        assert s not in starts     # window would cross the boundary at 49
    assert 50 in starts            # fresh episode
    assert starts.max() <= 100 - 16


def test_load_episode(tmp_path):
    ep_dir = tmp_path / "test_episodes"
    ep_dir.mkdir()
    frames = np.zeros((30, 64, 64, 3), dtype=np.uint8)
    actions = np.arange(30).astype(np.uint8) % 15
    np.savez_compressed(ep_dir / "ep_000.npz", frames=frames, actions=actions)
    ep = data.load_episode(ep_dir / "ep_000.npz")
    assert ep["frames"].shape == (30, 64, 64, 3)
    assert ep["actions"].shape == (30,)

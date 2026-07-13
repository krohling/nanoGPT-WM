"""Curate gen_dgrl chaser episodes into nano-world-model's .bin format.

Usage: python scripts/curate_dataset.py RAW_DIR OUT_DIR
         [--train-frames 420000] [--val-frames 20000] [--test-eps 60]

Streams episodes straight from the .tar.xz archives (never extracts raw data
to disk — see docs/notes/gen-dgrl-findings.md for the quota story), routing
each episode to one split: first `--test-eps` long-enough episodes become
held-out test episodes, then val fills up, then train. Episode-level split —
no episode contributes to two splits. Variants (expert/suboptimal) arrive
interleaved from the streaming loader.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_gen_dgrl import load_episodes  # noqa: E402

MIN_LEN = 20        # skip degenerate episodes (need seq_len=16 windows)
MIN_TEST_LEN = 100  # test episodes should support long open-loop rollouts

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("out")
ap.add_argument("--train-frames", type=int, default=420_000)
ap.add_argument("--val-frames", type=int, default=20_000)
ap.add_argument("--test-eps", type=int, default=60)
a = ap.parse_args()

out = Path(a.out)
(out / "test_episodes").mkdir(parents=True, exist_ok=True)

META = {"frame_shape": [64, 64, 3], "num_actions": 15,
        "action_convention":
            "actions[t] is taken at frames[t] and produces frames[t+1]"}


class SplitWriter:
    def __init__(self, name, budget):
        self.d = out / name
        self.d.mkdir(parents=True, exist_ok=True)
        self.name, self.budget, self.n, self.eps = name, budget, 0, 0
        self.ff = open(self.d / "frames.bin", "wb")
        self.af = open(self.d / "actions.bin", "wb")
        self.df = open(self.d / "dones.bin", "wb")

    @property
    def full(self):
        return self.n >= self.budget

    def add(self, ep):
        f, act = ep["frames"], ep["actions"]
        dones = np.zeros(len(f), np.uint8)
        dones[-1] = 1
        self.ff.write(f.tobytes())
        self.af.write(act.tobytes())
        self.df.write(dones.tobytes())
        self.n += len(f)
        self.eps += 1

    def close(self):
        for fh in (self.ff, self.af, self.df):
            fh.close()
        (self.d / "meta.json").write_text(json.dumps(dict(META, num_frames=self.n)))
        print(f"{self.name}: {self.n} frames from {self.eps} episodes")


test_n, val_w, train_w = 0, SplitWriter("val", a.val_frames), \
    SplitWriter("train", a.train_frames)
counts = {"expert": 0, "suboptimal": 0}
for ep in load_episodes(Path(a.raw)):
    if len(ep["frames"]) < MIN_LEN:
        continue
    if test_n < a.test_eps and len(ep["frames"]) >= MIN_TEST_LEN:
        np.savez_compressed(out / "test_episodes" / f"ep_{test_n:03d}.npz",
                            frames=ep["frames"], actions=ep["actions"])
        test_n += 1
    elif not val_w.full:
        val_w.add(ep)
    elif not train_w.full:
        train_w.add(ep)
        counts[ep["variant"]] += 1
    else:
        break
val_w.close()
train_w.close()
print(f"test episodes: {test_n}")
print(f"train variant mix: {counts}")

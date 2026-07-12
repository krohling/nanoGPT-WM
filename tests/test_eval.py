import subprocess
import sys

import numpy as np

from tests.test_data import make_split


def test_dynamics_evals_end_to_end(tmp_path):
    make_split(tmp_path / "d", "train", n=120)
    make_split(tmp_path / "d", "val", n=60, seed=1)
    ep_dir = tmp_path / "d" / "test_episodes"
    ep_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        np.savez_compressed(
            ep_dir / f"ep_{i:03d}.npz",
            frames=rng.integers(0, 256, (20, 64, 64, 3), dtype=np.uint8),
            actions=rng.integers(0, 15, 20).astype(np.uint8))

    def run(script_args):
        r = subprocess.run([sys.executable, *script_args], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    run(["train_tokenizer.py", "--data", str(tmp_path / "d"), "--out",
         str(tmp_path / "tok"), "--steps", "5", "--bs", "4", "--device", "cpu",
         "--save-every", "5"])
    base = ["--data", str(tmp_path / "d"), "--tokenizer",
            str(tmp_path / "tok" / "tokenizer.pt")]
    run(["train_wm.py", *base, "--out", str(tmp_path / "wm"), "--prepare",
         "--device", "cpu"])
    run(["train_wm.py", *base, "--out", str(tmp_path / "wm"), "--steps", "10",
         "--bs", "2", "--seq-len", "4", "--n-layer", "2", "--n-head", "2",
         "--n-embd", "32", "--device", "cpu", "--log-every", "5", "--eval-every", "10"])
    wm = [*base, "--wm", str(tmp_path / "wm" / "world_model.pt"),
          "--out", str(tmp_path / "ev"), "--device", "cpu"]
    run(["eval.py", "rollout", *wm, "--episode", "0", "--prime", "2",
         "--horizon", "6", "--seq-len", "4"])
    run(["eval.py", "drift", *wm, "--episodes", "2", "--horizon", "6",
         "--prime", "2", "--seq-len", "4"])
    run(["eval.py", "loss-heatmap", *wm, "--seq-len", "4"])
    assert (tmp_path / "ev" / "rollout_ep0.gif").exists()
    assert (tmp_path / "ev" / "rollout_ep0_mse.png").exists()
    assert (tmp_path / "ev" / "drift.png").exists()
    assert (tmp_path / "ev" / "drift.json").exists()
    assert (tmp_path / "ev" / "loss_heatmap.png").exists()

import subprocess
import sys

import torch

from tests.test_data import make_split


def run(args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def test_prepare_and_overfit(tmp_path):
    make_split(tmp_path / "d", "train", n=120)
    make_split(tmp_path / "d", "val", n=60, seed=1)
    # tiny tokenizer first
    r = run(["train_tokenizer.py", "--data", str(tmp_path / "d"), "--out",
             str(tmp_path / "tok"), "--steps", "5", "--bs", "4", "--device", "cpu",
             "--save-every", "5"])
    assert r.returncode == 0, r.stderr
    common = ["train_wm.py", "--data", str(tmp_path / "d"), "--tokenizer",
              str(tmp_path / "tok" / "tokenizer.pt"), "--out", str(tmp_path / "wm"),
              "--device", "cpu", "--seq-len", "4"]
    r = run(common + ["--prepare"])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "d" / "train" / "tokens.bin").exists()
    r = run(common + ["--steps", "20", "--bs", "4", "--n-layer", "2",
                      "--n-head", "2", "--n-embd", "32", "--overfit",
                      "--log-every", "10", "--eval-every", "20"])
    assert r.returncode == 0, r.stderr
    ck = torch.load(tmp_path / "wm" / "world_model.pt", map_location="cpu")
    assert ck["step"] == 20 and ck["tokens_per_frame"] == 64
    assert ck["config"]["block_size"] == 4 * 64 + 3

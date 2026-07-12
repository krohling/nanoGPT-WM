import subprocess
import sys

import torch

from tests.test_data import make_split


def test_overfit_smoke(tmp_path):
    make_split(tmp_path / "d", "train", n=64)
    make_split(tmp_path / "d", "val", n=64, seed=1)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "train_tokenizer.py", "--data", str(tmp_path / "d"),
         "--out", str(out), "--steps", "30", "--bs", "8", "--overfit",
         "--device", "cpu", "--log-every", "10", "--save-every", "30"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ck = torch.load(out / "tokenizer.pt", map_location="cpu")
    assert ck["grid"] == 8 and ck["step"] == 30
    assert (out / "recon_000030.png").exists()

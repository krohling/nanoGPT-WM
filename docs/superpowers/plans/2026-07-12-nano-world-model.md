# nano-world-model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the nano-world-model repo: a VQ-VAE tokenizer + autoregressive GPT dynamics model trained on procgen Chaser (gen_dgrl data), with an interactive play-in-the-dream demo, per the approved spec at `docs/superpowers/specs/2026-07-12-nano-world-model-design.md`.

**Architecture:** Two trained components — a VQ-VAE that maps 64×64×3 frames to an 8×8 grid of discrete tokens (codebook 512, EMA updates), and a nanoGPT-style decoder-only transformer over interleaved `[action, 64 frame tokens]` sequences. Data is curated from Meta's gen_dgrl offline procgen dataset and re-hosted on HuggingFace as flat memmap `.bin` files (nanoGPT idiom). Full AR sampling with KV cache powers a pygame play demo and a Colab notebook.

**Tech Stack:** PyTorch, numpy, huggingface_hub, pillow + matplotlib (visualization), pygame (play), ipywidgets (notebook), pytest (dev only).

## Global Constraints

- Repo root = `/Users/kevin/Desktop/projects/local-agent/research/simple-world-model` (already a git repo).
- Runtime deps ONLY: `torch`, `numpy`, `huggingface_hub`, `pillow`, `matplotlib`, `pygame`, `ipywidgets`. Dev dep: `pytest`. No Hydra, no Lightning, no torchvision.
- Core code target ~1,000 lines across `model.py`, `data.py`, `train_tokenizer.py`, `train_wm.py`, `eval.py`, `play.py`.
- Frames: 64×64×3 uint8. Tokenizer: codebook **512**, default grid **8×8** (64 tokens/frame), fallback grid 16×16.
- World model vocab: `FRAME_VOCAB=512` frame codes + `NUM_ACTIONS=15` procgen actions = **527**; action token id = `512 + action`. Loss is cross-entropy with action-position targets set to `-1` (`ignore_index`).
- Default WM config: n_layer=8, n_head=6, n_embd=384 (~14M params), seq_len=16 frames → block_size = 16×65 = **1040**.
- Training budget ceilings (Colab T4): tokenizer ≤1 hr, world model ≤3 hr.
- **HARD GATE after Task 6:** tokenizer render panels go to Kevin for qualitative review; NO world-model training runs until he signs off. (WM *code* may be written meanwhile — Tasks 7–9 are CPU-only.)
- Server env vars used throughout: `WM_SERVER` (ssh alias), `HF_NAMESPACE` (HuggingFace namespace, set in Task 0). Long server runs go in `tmux`.
- gen_dgrl data license is CC-BY-NC 4.0 — attribution required in HF dataset card and README.
- Commit after every task (at minimum). Match commit style already in repo history.

## Prerequisites (Task 0 — blocking info from Kevin)

- [ ] SSH access to a GPU server: alias configured in `~/.ssh/config` (referred to as `$WM_SERVER`), CUDA-capable GPU, ≥60 GB free disk, outbound internet, `tmux` available, ability to create a conda/venv.
- [ ] HuggingFace namespace decision (`HF_NAMESPACE`) + a write-scoped token, logged in on the server via `huggingface-cli login` (dataset repo `$HF_NAMESPACE/nano-world-model-chaser`, model repo `$HF_NAMESPACE/nano-world-model`).
- [ ] Confirm dataset + model repos may start **private**, flipped public at release.

---

### Task 1: Repo scaffold + data.py (memmap format, loaders)

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `data.load_split(root, split) -> (frames, actions, dones)` — numpy memmaps: frames uint8 `[N,64,64,3]`, actions uint8 `[N]`, dones uint8 `[N]`.
- Produces: `data.FrameDataset(root, split)` — `__getitem__ -> FloatTensor [3,64,64]` in `[0,1]`.
- Produces: `data.valid_starts(dones, seq_len) -> np.ndarray[int64]` — start indices whose `seq_len`-frame window does not cross an episode boundary (`dones[i : i+seq_len-1]` all zero; a done on the window's final frame is allowed).
- Produces: `data.load_episode(path) -> dict(frames=uint8 [T,64,64,3], actions=uint8 [T])`.
- Produces: `data.download(root="data/chaser", repo_id=None)` — HF snapshot download.
- Data dir layout (produced for real in Task 5): `<root>/{train,val}/frames.bin, actions.bin, dones.bin, meta.json` and `<root>/test_episodes/ep_###.npz`. `meta.json` = `{"num_frames": N, "frame_shape": [64,64,3], "num_actions": 15, "action_convention": "actions[t] is taken at frames[t] and produces frames[t+1]"}`.

- [ ] **Step 1: Scaffold files**

`requirements.txt`:
```
torch>=2.1
numpy
huggingface_hub
pillow
matplotlib
pygame
ipywidgets
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
data/
out/
panels/
raw_gen_dgrl/
.pytest_cache/
.ipynb_checkpoints/
```

- [ ] **Step 2: Write the failing tests**

`tests/test_data.py`:
```python
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
    # dones[49]=1 -> windows starting at 35..49 contain frame 49 not as last-or-later
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/kevin/Desktop/projects/local-agent/research/simple-world-model && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 4: Implement data.py**

```python
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

# Finalized in Task 5 once the HF namespace is decided.
DEFAULT_DATASET_REPO = os.environ.get("NWM_DATASET_REPO", "")


def download(root="data/chaser", repo_id=None):
    """Fetch the dataset from HuggingFace into `root`. Returns the local path."""
    from huggingface_hub import snapshot_download
    repo_id = repo_id or DEFAULT_DATASET_REPO
    if not repo_id:
        raise SystemExit("Set NWM_DATASET_REPO or pass repo_id (see README).")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=root)
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

    A done flag on the window's FINAL frame is fine (episode ends exactly there);
    a done anywhere earlier means the window straddles two episodes -> excluded.
    """
    dones = np.asarray(dones, dtype=np.uint8)
    n = len(dones)
    # cumulative count of dones lets us test "any done in dones[i : i+seq_len-1]" in O(1)
    cum = np.concatenate([[0], np.cumsum(dones)])
    starts = np.arange(n - seq_len + 1)
    inner = cum[starts + seq_len - 1] - cum[starts]  # dones in frames[i .. i+L-2]
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
        x = torch.from_numpy(np.ascontiguousarray(self.frames[i]))
        return x.permute(2, 0, 1).float() / 255.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore data.py tests/test_data.py
git commit -m "feat: repo scaffold and memmap data loading"
```

---

### Task 2: VQ-VAE tokenizer (model.py part 1)

**Files:**
- Create: `model.py`
- Test: `tests/test_vqvae.py`

**Interfaces:**
- Produces: module constants `FRAME_VOCAB = 512`, `NUM_ACTIONS = 15`, `VOCAB_SIZE = 527`.
- Produces: `VQVAE(codebook_size=512, emb_dim=64, grid=8)` with:
  - `forward(x [B,3,64,64] float in [0,1]) -> (recon [B,3,64,64], commit_loss scalar, idx [B, grid*grid] long)`
  - `encode(x) -> LongTensor [B, grid*grid]`
  - `decode(idx [B, grid*grid]) -> FloatTensor [B,3,64,64] in [0,1]`
  - attribute `tokens_per_frame = grid * grid`
- EMA codebook (decay 0.99) + dead-code reinit; straight-through estimator; commitment beta applied by the *caller* (train script uses `0.25 * commit`).

- [ ] **Step 1: Write the failing tests**

`tests/test_vqvae.py`:
```python
import pytest
import torch

from model import VQVAE, FRAME_VOCAB, NUM_ACTIONS, VOCAB_SIZE


def test_constants():
    assert FRAME_VOCAB == 512 and NUM_ACTIONS == 15 and VOCAB_SIZE == 527


@pytest.mark.parametrize("grid", [8, 16])
def test_shapes(grid):
    m = VQVAE(codebook_size=512, emb_dim=64, grid=grid)
    x = torch.rand(2, 3, 64, 64)
    recon, commit, idx = m(x)
    assert recon.shape == x.shape
    assert 0.0 <= recon.min() and recon.max() <= 1.0
    assert idx.shape == (2, grid * grid) and idx.dtype == torch.long
    assert idx.max() < 512
    assert commit.dim() == 0
    assert m.tokens_per_frame == grid * grid


def test_encode_decode():
    m = VQVAE()
    x = torch.rand(2, 3, 64, 64)
    idx = m.encode(x)
    out = m.decode(idx)
    assert out.shape == (2, 3, 64, 64)


def test_gradients_flow_through_encoder():
    m = VQVAE()
    x = torch.rand(2, 3, 64, 64)
    recon, commit, _ = m(x)
    loss = torch.nn.functional.mse_loss(recon, x) + 0.25 * commit
    loss.backward()
    g = next(m.encoder.parameters()).grad
    assert g is not None and g.abs().sum() > 0


@pytest.mark.slow
def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = VQVAE()
    # structured frames (colored squares), not noise — must become reconstructable
    x = torch.zeros(4, 3, 64, 64)
    for i in range(4):
        x[i, i % 3, 8 * i : 8 * i + 24, 8 * i : 8 * i + 24] = 1.0
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(300):
        recon, commit, _ = m(x)
        loss = torch.nn.functional.mse_loss(recon, x) + 0.25 * commit
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.nn.functional.mse_loss(m(x)[0], x).item() < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_vqvae.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement the VQ-VAE in model.py**

```python
"""nano-world-model: a VQ-VAE tokenizer and a GPT dynamics model.

The tokenizer turns each 64x64 RGB frame into a small grid of discrete codes
("the frame as 64 words"). The world model is a plain GPT over sequences of
[action, frame tokens, action, frame tokens, ...] — nanoGPT whose text is a game.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_VOCAB = 512      # tokenizer codebook size
NUM_ACTIONS = 15       # procgen discrete action space
VOCAB_SIZE = FRAME_VOCAB + NUM_ACTIONS  # world-model vocab; action a -> token 512+a


# ----------------------------- tokenizer ------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(), nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizerEMA(nn.Module):
    """Classic VQ with EMA codebook updates (van den Oord et al. 2017, appendix).

    EMA keeps a running estimate of each code's usage and mean assigned vector,
    which is more stable than learning the codebook by gradient. Codes that fall
    out of use get re-seeded from live encoder outputs (dead-code reinit).
    """

    def __init__(self, num_codes, dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes, self.dim, self.decay, self.eps = num_codes, dim, decay, eps
        embed = torch.randn(num_codes, dim) * 0.1
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", embed.clone())

    def forward(self, z):  # z: [B, D, H, W]
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)                    # [N, D]
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1))                          # [N, K]
        idx = dist.argmin(1)                                            # [N]
        quant = self.codebook[idx].view(B, H, W, D).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.num_codes).float()         # [N, K]
                count = onehot.sum(0)
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                smoothed = (self.ema_count + self.eps) / (n + self.num_codes * self.eps) * n
                self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))
                # dead-code reinit: any code with (almost) no usage steals a live vector
                dead = self.ema_count < 1.0
                if dead.any() and flat.shape[0] >= int(dead.sum()):
                    take = flat[torch.randperm(flat.shape[0])[: int(dead.sum())]]
                    self.codebook[dead] = take
                    self.ema_sum[dead] = take
                    self.ema_count[dead] = 1.0

        commit = F.mse_loss(z, quant.detach())
        quant = z + (quant - z).detach()   # straight-through: gradients skip argmin
        return quant, commit, idx.view(B, H * W)


class VQVAE(nn.Module):
    """Frames <-> tokens. grid=8 -> 64 tokens/frame (3 stride-2 convs), grid=16 -> 256."""

    def __init__(self, codebook_size=FRAME_VOCAB, emb_dim=64, grid=8):
        super().__init__()
        assert grid in (8, 16)
        self.grid, self.tokens_per_frame = grid, grid * grid
        downs = 3 if grid == 8 else 2
        chs = [64, 128, 256][:downs]
        enc, in_ch = [], 3
        for ch in chs:
            enc += [nn.Conv2d(in_ch, ch, 4, stride=2, padding=1), nn.ReLU()]
            in_ch = ch
        enc += [ResBlock(in_ch), ResBlock(in_ch), nn.Conv2d(in_ch, emb_dim, 1)]
        self.encoder = nn.Sequential(*enc)
        self.vq = VectorQuantizerEMA(codebook_size, emb_dim)
        dec = [nn.Conv2d(emb_dim, in_ch, 1), ResBlock(in_ch), ResBlock(in_ch)]
        for ch in reversed([3] + chs[:-1]):
            dec += [nn.ReLU(), nn.ConvTranspose2d(in_ch, ch, 4, stride=2, padding=1)]
            in_ch = ch
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        quant, commit, idx = self.vq(z)
        recon = torch.sigmoid(self.decoder(quant))
        return recon, commit, idx

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        _, _, idx = self.vq(z)
        return idx

    @torch.no_grad()
    def decode(self, idx):
        B = idx.shape[0]
        emb = self.vq.codebook[idx.view(-1)].view(B, self.grid, self.grid, -1)
        return torch.sigmoid(self.decoder(emb.permute(0, 3, 1, 2)))
```

Note: the decoder channel walk `reversed([3] + chs[:-1])` yields e.g. `[128, 64, 3]` for grid=8 — mirror of the encoder. The final ConvTranspose outputs 3 channels; sigmoid maps to `[0,1]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vqvae.py -v`
Expected: 6 passed (slow test ~30–60 s on CPU)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_vqvae.py
git commit -m "feat: VQ-VAE tokenizer with EMA codebook and dead-code reinit"
```

---

### Task 3: train_tokenizer.py

**Files:**
- Create: `train_tokenizer.py`, `eval.py` (visualization helpers only at this stage)
- Test: `tests/test_train_tokenizer.py`

**Interfaces:**
- Produces: CLI `python train_tokenizer.py --data DIR --out DIR [--steps N] [--bs N] [--lr F] [--grid 8|16] [--overfit] [--device D]`.
- Produces: checkpoint format `out/tokenizer.pt` = `{"model": state_dict, "grid": int, "step": int}` — consumed by Tasks 6–13.
- Produces: `eval.save_grid(frames_uint8 [N,64,64,3], path, ncol)` and `eval.to_uint8(t [B,3,64,64] float) -> np [B,64,64,3] u8` — reused by all later visualization.
- Produces: `eval.codebook_stats(idx_batches) -> (usage [512] np.int64, perplexity float)`.

- [ ] **Step 1: Write the failing test**

`tests/test_train_tokenizer.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_tokenizer.py -v`
Expected: FAIL (train_tokenizer.py does not exist; returncode != 0)

- [ ] **Step 3: Implement eval.py visualization helpers**

`eval.py` (initial content; dynamics subcommands are added in Task 9):
```python
"""Evaluation and visualization for nano-world-model."""
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
    """frames: uint8 [N,64,64,3] -> one PNG grid."""
    n = len(frames)
    nrow = (n + ncol - 1) // ncol
    H = W = 64 * scale
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
```

- [ ] **Step 4: Implement train_tokenizer.py**

```python
"""Stage 1: train the VQ-VAE tokenizer. ~30k steps fits in <1h on a T4."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import data as data_mod
from eval import save_grid, to_uint8, codebook_stats
from model import VQVAE

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--grid", type=int, default=8, choices=[8, 16])
p.add_argument("--beta", type=float, default=0.25)   # commitment weight
p.add_argument("--overfit", action="store_true", help="train on 256 fixed frames")
p.add_argument("--log-every", type=int, default=100)
p.add_argument("--save-every", type=int, default=2000)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
ds = data_mod.FrameDataset(args.data, "train")
if args.overfit:
    ds = Subset(ds, range(min(256, len(ds))))
dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2,
                pin_memory=True, drop_last=True, persistent_workers=True)
val = data_mod.FrameDataset(args.data, "val")
val_batch = torch.stack([val[i] for i in range(min(32, len(val)))]).to(args.device)

model = VQVAE(grid=args.grid).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler(enabled=args.device == "cuda")

step, t0, it = 0, time.time(), iter(dl)
while step < args.steps:
    try:
        x = next(it)
    except StopIteration:
        it = iter(dl); x = next(it)
    x = x.to(args.device, non_blocking=True)
    with torch.amp.autocast(args.device, enabled=args.device == "cuda"):
        recon, commit, idx = model(x)
        loss = F.mse_loss(recon, x) + args.beta * commit
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    step += 1
    if step % args.log_every == 0:
        _, ppl = codebook_stats([idx])
        print(f"step {step:6d}  loss {loss.item():.4f}  ppl {ppl:6.1f}  "
              f"{args.log_every / (time.time() - t0):.1f} it/s", flush=True)
        t0 = time.time()
    if step % args.save_every == 0 or step == args.steps:
        model.eval()
        with torch.no_grad():
            r, _, _ = model(val_batch)
        pair = np.concatenate([to_uint8(val_batch), to_uint8(r)], axis=2)  # side-by-side
        save_grid(pair.reshape(-1, 64, 128, 3)[:32], out / f"recon_{step:06d}.png", ncol=4)
        torch.save({"model": model.state_dict(), "grid": args.grid, "step": step},
                   out / "tokenizer.pt")
        model.train()
print("done")
```

Note: `pair` concatenates original|reconstruction horizontally per frame (64×128 tiles); `save_grid` handles non-square tiles because it only resizes, so pass tiles as-is — adjust `save_grid` call: it expects 64×64; passing 64×128 works since PIL resize scales both dims (`W` computed from tile). Update `save_grid` to derive tile size from input:

In `eval.py::save_grid` replace the fixed `H = W = 64 * scale` with:
```python
    th, tw = frames[0].shape[:2]
    H, W = th * scale, tw * scale
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_tokenizer.py -v`
Expected: 1 passed (~60–90 s CPU)

- [ ] **Step 6: Run the full local test suite**

Run: `.venv/bin/pytest tests/ -v -m "not slow"`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add train_tokenizer.py eval.py tests/test_train_tokenizer.py
git commit -m "feat: tokenizer training script with recon grids and codebook stats"
```

---

### Task 4: [SERVER] gen_dgrl download + verification gate

**Files:**
- Create: `scripts/inspect_gen_dgrl.py`, `docs/notes/gen-dgrl-findings.md`

**Interfaces:**
- Consumes: `$WM_SERVER`, server conda/venv.
- Produces: raw gen_dgrl Chaser data on the server under `~/nwm/raw_gen_dgrl/`, plus a written findings note that Task 5 depends on (file layout, dtypes, action convention, episode stats, GO/FALLBACK decision).

This task is investigation: the gen_dgrl release layout must be **discovered and recorded**, not assumed. Decision criteria are fixed below.

- [ ] **Step 1: Set up server workspace**

```bash
ssh $WM_SERVER 'mkdir -p ~/nwm && cd ~/nwm && python3 -m venv venv && \
  venv/bin/pip install torch numpy huggingface_hub pillow'
ssh $WM_SERVER 'nvidia-smi; df -h ~ | tail -1'
```
Expected: GPU listed; ≥60 GB free. Record GPU model in the findings note.

- [ ] **Step 2: Locate and download the Chaser data**

```bash
ssh $WM_SERVER 'cd ~/nwm && git clone https://github.com/facebookresearch/gen_dgrl.git'
ssh $WM_SERVER 'grep -ri -A5 "download" ~/nwm/gen_dgrl/README.md | head -80'
```
Follow the repo's documented download procedure for **chaser** only (the release provides per-game expert (1M) and suboptimal/mixed variants; download BOTH variants for chaser if separately available). Put raw files under `~/nwm/raw_gen_dgrl/`. If the documented URLs 404, check the repo's issues for mirrors before declaring FALLBACK.

- [ ] **Step 3: Write scripts/inspect_gen_dgrl.py**

The script must load whatever per-episode/per-shard files the release uses (discover format in Step 2 — likely `.npy`/`.hdf5` per-episode trajectories) and print/save:

```python
"""Inspect raw gen_dgrl chaser data. Usage: python scripts/inspect_gen_dgrl.py RAW_DIR OUT_DIR
Prints dataset stats and saves a 64-frame contact sheet + action histogram.
Adapt the `load_episodes` function to the actual on-disk layout discovered in Task 4
and RECORD that layout in docs/notes/gen-dgrl-findings.md.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import save_grid


def load_episodes(raw_dir):
    """Yield dicts with keys: frames [T,64,64,3] u8, actions [T] int, dones/T implied.
    IMPLEMENT AGAINST THE REAL LAYOUT and document it in the findings note."""
    raise NotImplementedError("fill in against the discovered gen_dgrl layout")


raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lens, act = [], Counter()
sheet = []
for i, ep in enumerate(load_episodes(raw)):
    f, a = ep["frames"], ep["actions"]
    assert f.dtype == np.uint8 and f.shape[1:] == (64, 64, 3), f.shape
    assert a.min() >= 0 and a.max() < 15, (a.min(), a.max())
    lens.append(len(f)); act.update(a.tolist())
    if i % 50 == 0 and len(sheet) < 64:
        sheet.append(f[len(f) // 2])
    if i >= 2000:
        break
print(f"episodes: {len(lens)}  frames: {sum(lens)}  len min/med/max: "
      f"{min(lens)}/{int(np.median(lens))}/{max(lens)}")
tot = sum(act.values())
for k in sorted(act):
    print(f"action {k:2d}: {act[k] / tot:6.2%}")
save_grid(np.stack(sheet), out / "contact_sheet.png")
```

Note the `NotImplementedError`: filling in `load_episodes` **is Step 4's work product**, done against the real files — commit the working version.

- [ ] **Step 4: Run inspection on the server, record findings**

```bash
scp scripts/inspect_gen_dgrl.py $WM_SERVER:~/nwm/
ssh $WM_SERVER 'cd ~/nwm && venv/bin/python inspect_gen_dgrl.py raw_gen_dgrl inspect_out'
scp $WM_SERVER:~/nwm/inspect_out/contact_sheet.png docs/notes/
```

Write `docs/notes/gen-dgrl-findings.md` recording: exact download URLs used, on-disk layout, dtype/shape confirmation, episode count and length stats, action histogram, action convention (verify `actions[t]` produces `frames[t+1]` by checking a frame pair around a distinctive movement), which variants were obtained, contact-sheet observations.

**GO criteria (all must hold):** frames are 64×64×3 uint8; actions in [0,15); ≥ 600k total usable frames; episode boundaries recoverable; ≥ 8 distinct actions each ≥ 1% frequency, no action > 50%; contact sheet shows legible mazes/sprites.
**FALLBACK:** if any fail → STOP, report to Kevin, switch to the PPO-checkpoint collection plan (separate planning session).

- [ ] **Step 5: Commit**

```bash
git add scripts/inspect_gen_dgrl.py docs/notes/gen-dgrl-findings.md docs/notes/contact_sheet.png
git commit -m "feat: gen_dgrl inspection script and verification findings"
```

---

### Task 5: [SERVER] Curation → HuggingFace dataset

**Files:**
- Create: `scripts/curate_dataset.py`, `scripts/hf_dataset_card.md`
- Modify: `data.py` (set `DEFAULT_DATASET_REPO`)

**Interfaces:**
- Consumes: raw data + layout knowledge from Task 4 (`load_episodes` from `scripts/inspect_gen_dgrl.py`).
- Produces: HF dataset repo `$HF_NAMESPACE/nano-world-model-chaser` (private) in the exact layout `data.py` expects (Task 1). Splits: train ≈ 420k frames, val ≈ 20k frames, test_episodes = 60 complete episodes. Episode-level split — no episode contributes to two splits.

- [ ] **Step 1: Write scripts/curate_dataset.py**

```python
"""Curate raw gen_dgrl chaser episodes into nano-world-model's .bin format.

Usage: python scripts/curate_dataset.py RAW_DIR OUT_DIR [--train-frames 420000] [--val-frames 20000] [--test-eps 60]
Mixes expert and suboptimal variants ~50/50 (interleaved) when both exist.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_gen_dgrl import load_episodes  # the version implemented in Task 4

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--train-frames", type=int, default=420_000)
ap.add_argument("--val-frames", type=int, default=20_000)
ap.add_argument("--test-eps", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

eps = list(load_episodes(Path(a.raw)))
rng = np.random.default_rng(a.seed)
rng.shuffle(eps)

out = Path(a.out)
test, rest = eps[: a.test_eps], eps[a.test_eps:]
(out / "test_episodes").mkdir(parents=True, exist_ok=True)
for i, ep in enumerate(test):
    np.savez_compressed(out / "test_episodes" / f"ep_{i:03d}.npz",
                        frames=ep["frames"], actions=ep["actions"].astype(np.uint8))

def write_split(name, budget, pool):
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    ff = open(d / "frames.bin", "wb"); af = open(d / "actions.bin", "wb")
    df = open(d / "dones.bin", "wb")
    n = 0
    while pool and n < budget:
        ep = pool.pop()
        f, act = ep["frames"], ep["actions"].astype(np.uint8)
        dones = np.zeros(len(f), np.uint8); dones[-1] = 1
        ff.write(f.tobytes()); af.write(act.tobytes()); df.write(dones.tobytes())
        n += len(f)
    for fh in (ff, af, df): fh.close()
    (d / "meta.json").write_text(json.dumps({
        "num_frames": n, "frame_shape": [64, 64, 3], "num_actions": 15,
        "action_convention": "actions[t] is taken at frames[t] and produces frames[t+1]",
    }))
    print(f"{name}: {n} frames")
    return pool

pool = write_split("val", a.val_frames, rest)
write_split("train", a.train_frames, pool)
```

If both expert and suboptimal variants exist, `load_episodes` must interleave them (alternate yielding) so the shuffle above mixes skill levels; record the actual mix in the findings note.

- [ ] **Step 2: Run curation on the server**

```bash
scp scripts/curate_dataset.py $WM_SERVER:~/nwm/
ssh $WM_SERVER 'cd ~/nwm && venv/bin/python curate_dataset.py raw_gen_dgrl hf_upload'
```
Expected output: `val: ~20000 frames`, `train: ~420000 frames`; `ls hf_upload/test_episodes | wc -l` → 60.

- [ ] **Step 3: Write the dataset card**

`scripts/hf_dataset_card.md` — becomes the HF repo README:
```markdown
---
license: cc-by-nc-4.0
---
# nano-world-model: procgen Chaser frames + actions

Curated subset of the offline procgen dataset released with
"The Generalization Gap in Offline Reinforcement Learning" (Mediratta et al.,
ICLR 2024, https://github.com/facebookresearch/gen_dgrl), CC-BY-NC 4.0.
Frames were collected by PPO agents playing procgen **Chaser** at 64x64.

Format (memmap-friendly flat binaries):
- `train/`, `val/`: `frames.bin` (uint8, N x 64 x 64 x 3), `actions.bin` (uint8, N),
  `dones.bin` (uint8, N), `meta.json`. `actions[t]` is taken at `frames[t]` and
  produces `frames[t+1]`; `dones[t]=1` marks an episode's final frame.
- `test_episodes/ep_###.npz`: 60 complete held-out episodes (`frames`, `actions`)
  for open-loop evaluation and dream priming.

Made for [nano-world-model](https://github.com/REPO_URL_SET_AT_RELEASE) — an
educational, minimal action-conditioned world model. Non-commercial use only.
```

- [ ] **Step 4: Create the HF repo and upload**

```bash
ssh $WM_SERVER 'cd ~/nwm && venv/bin/python - << "EOF"
from huggingface_hub import HfApi
import os
api = HfApi()
ns = os.environ["HF_NAMESPACE"]
api.create_repo(f"{ns}/nano-world-model-chaser", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="hf_upload", repo_id=f"{ns}/nano-world-model-chaser", repo_type="dataset")
EOF'
```
Then upload `scripts/hf_dataset_card.md` as the dataset README via `api.upload_file(path_or_fileobj=..., path_in_repo="README.md", ...)`.

- [ ] **Step 5: Set DEFAULT_DATASET_REPO and round-trip test**

In `data.py`, change the empty default:
```python
DEFAULT_DATASET_REPO = os.environ.get("NWM_DATASET_REPO", "<HF_NAMESPACE>/nano-world-model-chaser")
```
(with the real namespace literal). Round-trip on the server:
```bash
ssh $WM_SERVER 'cd ~/nwm && venv/bin/python -c "
import data
p = data.download(root=\"data/chaser\")
f, a, d = data.load_split(p, \"train\")
print(f.shape, a.shape, int(d.sum()), \"episodes\")
print(len(data.list_episodes(p)), \"test episodes\")
"'
```
Expected: `(~420000, 64, 64, 3)`, matching actions/dones, 60 test episodes.

- [ ] **Step 6: Commit**

```bash
git add scripts/curate_dataset.py scripts/hf_dataset_card.md data.py
git commit -m "feat: dataset curation and HF hosting (gen_dgrl chaser subset)"
```

---

### Task 6: [SERVER] Tokenizer training + review panels → **HARD PAUSE**

**Files:**
- Create: `scripts/make_tokenizer_panels.py`

**Interfaces:**
- Consumes: `train_tokenizer.py` (Task 3), dataset on server (Task 5), `eval.py` helpers.
- Produces: `out/tok8/tokenizer.pt` on the server; a review PDF for Kevin. **World-model training may not start until Kevin approves.**

- [ ] **Step 1: Sync repo to server and launch training**

```bash
rsync -av --exclude .venv --exclude data --exclude .git . $WM_SERVER:~/nwm/repo/
ssh $WM_SERVER 'cd ~/nwm/repo && ln -sfn ~/nwm/data data && tmux new -d -s tok8 \
  "~/nwm/venv/bin/python train_tokenizer.py --data data/chaser --out out/tok8 \
   --steps 30000 --bs 256 2>&1 | tee out/tok8.log"'
```
Monitor: `ssh $WM_SERVER 'tail -5 ~/nwm/repo/out/tok8.log'`. Expected: loss falling below ~0.003 and perplexity > 200 by late training (indicative, not gating).

- [ ] **Step 2: Write scripts/make_tokenizer_panels.py**

```python
"""Build the tokenizer review panels (spec: review gate deliverable).
Usage: python scripts/make_tokenizer_panels.py CKPT DATA_DIR OUT_DIR
Panels: (a) random orig|recon grid, (b) worst-32 hard cases by MSE,
(c) codebook usage histogram, (d) mean per-pixel error heatmap.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data as data_mod
from eval import save_grid, to_uint8, codebook_stats
from model import VQVAE

ckpt, root, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ckpt, map_location=dev)
m = VQVAE(grid=ck["grid"]).to(dev).eval()
m.load_state_dict(ck["model"])

val = data_mod.FrameDataset(root, "val")
N = min(5000, len(val))
mses, idxs = [], []
with torch.no_grad():
    for s in range(0, N, 256):
        x = torch.stack([val[i] for i in range(s, min(s + 256, N))]).to(dev)
        r, _, idx = m(x)
        mses.append(((r - x) ** 2).mean(dim=(1, 2, 3)).cpu())
        idxs.append(idx)
mse = torch.cat(mses)

def pair_grid(ids, path):
    x = torch.stack([val[int(i)] for i in ids]).to(dev)
    with torch.no_grad():
        r, _, _ = m(x)
    pair = np.concatenate([to_uint8(x), to_uint8(r)], axis=2)
    save_grid(pair, path, ncol=4)

pair_grid(np.random.default_rng(0).choice(N, 32, replace=False), out / "panel_random.png")
pair_grid(mse.topk(32).indices.numpy(), out / "panel_hard.png")

usage, ppl = codebook_stats(idxs)
plt.figure(figsize=(10, 3))
plt.bar(range(512), np.sort(usage)[::-1], width=1.0)
plt.title(f"codebook usage (sorted) — perplexity {ppl:.0f}, "
          f"{(usage > 0).sum()}/512 codes used")
plt.tight_layout(); plt.savefig(out / "panel_codebook.png", dpi=150); plt.close()

with torch.no_grad():
    x = torch.stack([val[i] for i in range(min(2000, N))]).to(dev)
    err = torch.zeros(64, 64)
    for s in range(0, len(x), 256):
        xb = x[s:s + 256]
        r, _, _ = m(xb)
        err += (r - xb).abs().mean(dim=(0, 1)).cpu() * len(xb)
    err /= len(x)
plt.figure(figsize=(4, 4))
plt.imshow(err.numpy(), cmap="magma"); plt.colorbar()
plt.title("mean |error| per pixel")
plt.tight_layout(); plt.savefig(out / "panel_pixel_error.png", dpi=150); plt.close()
print(f"val MSE mean {mse.mean():.5f}  p95 {mse.quantile(0.95):.5f}  perplexity {ppl:.0f}")
```

- [ ] **Step 3: Generate panels, assemble review PDF, deliver to Kevin**

```bash
ssh $WM_SERVER 'cd ~/nwm/repo && ~/nwm/venv/bin/python scripts/make_tokenizer_panels.py \
  out/tok8/tokenizer.pt data/chaser panels/tok8'
rsync -av $WM_SERVER:~/nwm/repo/panels/tok8/ panels/tok8/
```
Assemble the four PNGs + the printed stats into a one-page PDF (pandoc/Chrome headless as done for the spec) and send to Kevin via SendUserFile.

**Borderline rule:** if sprites (orbs/enemies/agent) are visibly mangled in `panel_hard.png`, ALSO launch a `--grid 16` run (`out/tok16`) and include a side-by-side comparison in the PDF, per spec.

- [ ] **Step 4: Commit, then STOP**

```bash
git add scripts/make_tokenizer_panels.py
git commit -m "feat: tokenizer review panel generation"
```

**HARD GATE: report results to Kevin and WAIT for his qualitative sign-off (including grid=8 vs grid=16 decision) before any Task 10 training run.** Tasks 7–9 (CPU-only coding) may proceed while waiting.

---

### Task 7: WorldModel GPT (model.py part 2)

**Files:**
- Modify: `model.py` (append)
- Test: `tests/test_wm.py`

**Interfaces:**
- Produces: `GPTConfig(vocab_size=527, block_size=1040, n_layer=8, n_head=6, n_embd=384, dropout=0.0)` dataclass.
- Produces: `WorldModel(cfg)` with:
  - `forward(idx [B,S] long, targets=None [B,S] long with -1 ignored) -> (logits [B,S,vocab], loss|None)`
  - `forward_cached(idx_new [B,s], past) -> (logits [B,s,vocab], past)` — incremental KV-cached forward; `past=None` starts a fresh cache.
  - `generate_frame(ctx [1,S] long, action int, tokens_per_frame=64, temperature=1.0, top_k=50) -> LongTensor [1, tokens_per_frame]` — appends the action token, prefills the cache, then samples frame tokens one at a time **restricted to the frame vocab** (action logits masked to -inf).
- Produces (in `train_wm.py`, but tested here — see Task 8 interfaces): `interleave(tokens [T,K] , actions [T]) -> np.int64 [T*(K+1)]` and `make_targets(seq) -> np.int64` (shifted, action positions = -1).

- [ ] **Step 1: Write the failing tests**

`tests/test_wm.py`:
```python
import numpy as np
import pytest
import torch

from model import FRAME_VOCAB, VOCAB_SIZE, GPTConfig, WorldModel


def tiny_cfg(**kw):
    d = dict(vocab_size=VOCAB_SIZE, block_size=5 * 5, n_layer=2, n_head=2,
             n_embd=32, dropout=0.0)
    d.update(kw)
    return GPTConfig(**d)


def test_forward_shapes_and_loss():
    m = WorldModel(tiny_cfg())
    idx = torch.randint(0, VOCAB_SIZE, (2, 20))
    tgt = idx.roll(-1, dims=1).clone()
    tgt[tgt >= FRAME_VOCAB] = -1
    logits, loss = m(idx, tgt)
    assert logits.shape == (2, 20, VOCAB_SIZE)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_causality():
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, VOCAB_SIZE, (1, 20))
    with torch.no_grad():
        l1, _ = m(idx)
        idx2 = idx.clone(); idx2[0, 10] = (idx2[0, 10] + 1) % VOCAB_SIZE
        l2, _ = m(idx2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5)
    assert not torch.allclose(l1[0, 10:], l2[0, 10:], atol=1e-5)


def test_loss_ignores_action_positions():
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, FRAME_VOCAB, (1, 10))
    tgt = torch.full((1, 10), -1)
    tgt[0, 3] = 7
    logits, loss = m(idx, tgt)
    per_tok = torch.nn.functional.cross_entropy(
        logits[0], tgt[0].clamp(min=0), reduction="none")
    assert torch.isclose(loss, per_tok[3], atol=1e-5)


def test_cached_forward_matches_full():
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, VOCAB_SIZE, (1, 12))
    with torch.no_grad():
        full, _ = m(idx)
        l1, past = m.forward_cached(idx[:, :8], None)
        l2, past = m.forward_cached(idx[:, 8:], past)
    assert torch.allclose(full[:, :8], l1, atol=1e-4)
    assert torch.allclose(full[:, 8:], l2, atol=1e-4)


def test_generate_frame_range():
    m = WorldModel(tiny_cfg(block_size=200)).eval()
    ctx = torch.randint(0, FRAME_VOCAB, (1, 30))
    out = m.generate_frame(ctx, action=4, tokens_per_frame=16)
    assert out.shape == (1, 16)
    assert out.max() < FRAME_VOCAB  # never samples an action token


def test_interleave_and_targets():
    from train_wm import interleave, make_targets
    tokens = np.arange(8, dtype=np.int64).reshape(2, 4)     # 2 frames, 4 tokens
    actions = np.array([3, 7])
    seq = interleave(tokens, actions)
    assert seq.tolist() == [512 + 3, 0, 1, 2, 3, 512 + 7, 4, 5, 6, 7]
    tgt = make_targets(seq)
    # next-token targets, with positions whose TARGET is an action masked out
    assert tgt.tolist() == [0, 1, 2, 3, -1, 4, 5, 6, 7]
    assert len(tgt) == len(seq) - 1


@pytest.mark.slow
def test_action_conditioning_on_toy_world():
    """A 4x4 gridworld: frame = 16 tokens (agent cell=1, rest=0). Actions
    1/7/5/3 move the agent left/right/up/down (procgen movement subset).
    A tiny WM must learn action-conditioned dynamics almost perfectly."""
    from train_wm import interleave, make_targets
    torch.manual_seed(0); rng = np.random.default_rng(0)
    MOVES = {1: (0, -1), 7: (0, 1), 5: (-1, 0), 3: (1, 0)}

    def episode(T=9):
        r, c = rng.integers(0, 4, 2)
        toks, acts = [], []
        for _ in range(T):
            f = np.zeros(16, np.int64); f[r * 4 + c] = 1
            a = int(rng.choice(list(MOVES)))
            toks.append(f); acts.append(a)
            dr, dc = MOVES[a]
            r, c = np.clip(r + dr, 0, 3), np.clip(c + dc, 0, 3)
        return np.stack(toks), np.array(acts)

    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=9 * 17, n_layer=2,
                    n_head=2, n_embd=64, dropout=0.0)
    m = WorldModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(400):
        seqs = [interleave(*episode()) for _ in range(16)]
        x = torch.tensor(np.stack([s[:-1] for s in seqs]))
        y = torch.tensor(np.stack([make_targets(s) for s in seqs]))
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    # rollout: prime one frame, drive 5 chosen actions, check the agent obeys
    m.eval()
    toks, _ = episode(1)
    ctx = torch.tensor(np.concatenate([[512 + 4], toks[0]]))[None]  # noop-primed
    r, c, ok = *(divmod(int(toks[0].argmax()), 4)), 0
    for a in [7, 7, 3, 1, 5]:
        out = m.generate_frame(ctx, action=a, tokens_per_frame=16, temperature=0.01)
        dr, dc = MOVES[a]
        r, c = int(np.clip(r + dr, 0, 3)), int(np.clip(c + dc, 0, 3))
        ok += int(out[0].argmax().item() == r * 4 + c)
        ctx = torch.cat([ctx, torch.tensor([[512 + a]]), out], dim=1)
    assert ok >= 4, f"only {ok}/5 moves obeyed the action"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_wm.py -v -m "not slow"`
Expected: FAIL with `ImportError` (GPTConfig/WorldModel not defined)

- [ ] **Step 3: Append the GPT to model.py**

```python
# ----------------------------- world model ----------------------------------
# A nanoGPT. The only differences from language modeling: the vocabulary mixes
# frame codes (0..511) with action tokens (512..526), and the loss skips
# positions whose target is an action (the model predicts the world, not the
# player's mind).

@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 1040          # 16 frames x (1 action + 64 tokens)
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x, past=None):
        B, S, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=past is None or past[0].shape[2] == 0,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, past=None):
        a, kv = self.attn(self.ln1(x), past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, kv


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, S = idx.shape
        assert S <= self.cfg.block_size
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(S, device=idx.device)))
        for blk in self.blocks:
            x, _ = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def forward_cached(self, idx_new, past):
        """Incremental forward. past = (pos, [per-layer (k,v)]) or None."""
        pos0, kvs = past if past is not None else (0, [None] * self.cfg.n_layer)
        B, s = idx_new.shape
        pos = torch.arange(pos0, pos0 + s, device=idx_new.device)
        x = self.wte(idx_new) + self.wpe(pos)
        new_kvs = []
        for blk, kv in zip(self.blocks, kvs):
            x, nkv = blk(x, past=kv if kv is not None else None)
            new_kvs.append(nkv)
        logits = self.head(self.ln_f(x))
        return logits, (pos0 + s, new_kvs)

    @torch.no_grad()
    def generate_frame(self, ctx, action, tokens_per_frame=64,
                       temperature=1.0, top_k=50):
        """Append an action token, then sample one full frame, token by token.
        This is nanoGPT's generate(): the chain rule, 64 links long."""
        dev = ctx.device
        seq = torch.cat([ctx, torch.tensor([[FRAME_VOCAB + action]], device=dev)], 1)
        logits, past = self.forward_cached(seq, None)      # prefill
        out = []
        for _ in range(tokens_per_frame):
            lg = logits[:, -1, :FRAME_VOCAB] / max(temperature, 1e-6)  # frame vocab only
            if top_k is not None:
                kth = torch.topk(lg, top_k).values[..., -1, None]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            tok = torch.multinomial(F.softmax(lg, dim=-1), 1)
            out.append(tok)
            logits, past = self.forward_cached(tok, past)
        return torch.cat(out, dim=1)
```

Caveat honored by tests: `forward_cached` uses `is_causal` only on the prefill (square attention); incremental single-token steps attend to the whole cache (`is_causal=False` path via `past[0].shape[2] != 0`). The prefill call passes `past=None` → `is_causal=True`. Subsequent calls pass grown caches → non-causal flag, which is correct because queries are strictly newest tokens.

- [ ] **Step 4: Create a minimal train_wm.py stub containing only the two pure functions** (full script is Task 8; the functions are tested now):

```python
"""Stage 2: train the world-model GPT on tokenized frames. (Training loop: Task 8.)"""
import numpy as np

from model import FRAME_VOCAB


def interleave(tokens, actions):
    """tokens [T,K] int, actions [T] int -> [T*(K+1)] int64: a0 z0.. a1 z1.."""
    T, K = tokens.shape
    seq = np.empty((T, K + 1), dtype=np.int64)
    seq[:, 0] = np.asarray(actions, dtype=np.int64) + FRAME_VOCAB
    seq[:, 1:] = tokens
    return seq.reshape(-1)


def make_targets(seq):
    """Next-token targets for seq[:-1]; positions predicting an action -> -1."""
    tgt = seq[1:].astype(np.int64).copy()
    tgt[tgt >= FRAME_VOCAB] = -1
    return tgt
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_wm.py -v -m "not slow"` → 6 passed.
Run: `.venv/bin/pytest tests/test_wm.py::test_action_conditioning_on_toy_world -v` → 1 passed (~1–3 min CPU). This test is the pre-GPU proof that interleaving, masking, caching, and generation compose correctly.

- [ ] **Step 6: Commit**

```bash
git add model.py train_wm.py tests/test_wm.py
git commit -m "feat: world-model GPT with KV cache and action-masked loss"
```

---

### Task 8: train_wm.py (token pre-compute + training loop)

**Files:**
- Modify: `train_wm.py` (extend the Task 7 stub)
- Test: `tests/test_train_wm.py`

**Interfaces:**
- Consumes: `tokenizer.pt` checkpoint, dataset dir, `data.valid_starts`.
- Produces: CLI `python train_wm.py --data DIR --tokenizer CKPT --out DIR [--prepare] [--steps N] [--bs N] [--seq-len 16] [--overfit] [--device D] [training hparams]`.
- Produces: token cache `<data>/<split>/tokens.bin` (uint16 memmap `[N, tokens_per_frame]`) via `--prepare`.
- Produces: checkpoint `out/world_model.pt` = `{"model": state_dict, "config": asdict(GPTConfig), "tokens_per_frame": int, "step": int, "val_loss": float}` — consumed by Tasks 9–13.

- [ ] **Step 1: Write the failing test**

`tests/test_train_wm.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_wm.py -v`
Expected: FAIL (train_wm.py has no CLI yet)

- [ ] **Step 3: Extend train_wm.py**

Append below the Task 7 functions:

```python
if __name__ == "__main__":
    import argparse
    import json
    import math
    import time
    from dataclasses import asdict
    from pathlib import Path

    import torch
    from model import VQVAE, GPTConfig, WorldModel
    import data as data_mod

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prepare", action="store_true", help="tokenize dataset, then exit")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--overfit", action="store_true", help="train on 64 fixed windows")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = args.device
    ck = torch.load(args.tokenizer, map_location=dev)
    tok = VQVAE(grid=ck["grid"]).to(dev).eval()
    tok.load_state_dict(ck["model"])
    K = tok.tokens_per_frame

    if args.prepare:
        for split in ("train", "val"):
            frames, _, _ = data_mod.load_split(args.data, split)
            out_path = Path(args.data) / split / "tokens.bin"
            mm = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(len(frames), K))
            for s in range(0, len(frames), 512):
                x = torch.from_numpy(np.ascontiguousarray(frames[s:s + 512]))
                x = x.permute(0, 3, 1, 2).float().div_(255).to(dev)
                mm[s:s + len(x)] = tok.encode(x).cpu().numpy().astype(np.uint16)
                if s % 51200 == 0:
                    print(f"{split}: {s}/{len(frames)}", flush=True)
            mm.flush()
        raise SystemExit(0)

    def load_windows(split):
        frames, actions, dones = data_mod.load_split(args.data, split)
        tokens = np.memmap(Path(args.data) / split / "tokens.bin", dtype=np.uint16,
                           mode="r", shape=(len(frames), K))
        starts = data_mod.valid_starts(dones, args.seq_len)
        return tokens, np.asarray(actions), starts

    tr_tokens, tr_actions, tr_starts = load_windows("train")
    va_tokens, va_actions, va_starts = load_windows("val")
    if args.overfit:
        tr_starts = tr_starts[:64]
    rng = np.random.default_rng(0)
    va_fixed = va_starts[rng.choice(len(va_starts), min(256, len(va_starts)), replace=False)]

    def batch(tokens, actions, starts, idx):
        xs, ys = [], []
        for s in idx:
            seq = interleave(tokens[s:s + args.seq_len].astype(np.int64),
                             actions[s:s + args.seq_len])
            xs.append(seq[:-1]); ys.append(make_targets(seq))
        return (torch.tensor(np.stack(xs), device=dev),
                torch.tensor(np.stack(ys), device=dev))

    cfg = GPTConfig(block_size=args.seq_len * (K + 1), n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = WorldModel(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))

    best, t0 = float("inf"), time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(tr_tokens, tr_actions, tr_starts,
                     rng.choice(len(tr_starts), args.bs))
        with torch.amp.autocast(dev, enabled=dev == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % args.log_every == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{args.log_every / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval(); vl, n = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(va_fixed), args.bs):
                    x, y = batch(va_tokens, va_actions, va_fixed, va_fixed[s:s + args.bs] * 0
                                 + va_fixed[s:s + args.bs])  # indices are starts already
                    _, l = model(x, y)
                    vl += l.item() * len(x); n += len(x)
            vl /= max(n, 1)
            print(f"step {step}: val loss {vl:.4f}", flush=True)
            if vl < best or step == args.steps:
                best = min(best, vl)
                torch.save({"model": model.state_dict(), "config": asdict(cfg),
                            "tokens_per_frame": K, "step": step, "val_loss": vl},
                           out / "world_model.pt")
            model.train()
    print("done")
```

Fix the sloppy val-batch line before running — `batch()` takes start indices directly:
```python
                    x, y = batch(va_tokens, va_actions, None, va_fixed[s:s + args.bs])
```
and change `batch`'s signature to `batch(tokens, actions, idx)` (drop the unused `starts` param) with call sites `batch(tr_tokens, tr_actions, rng.choice(tr_starts, args.bs))` — note: choose FROM `tr_starts` values, i.e. `rng.choice(tr_starts, size=args.bs)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_wm.py -v`
Expected: 1 passed (~2–3 min CPU)

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/pytest tests/ -m "not slow" -v` → all pass.
```bash
git add train_wm.py tests/test_train_wm.py
git commit -m "feat: world-model training with token pre-compute"
```

---

### Task 9: eval.py dynamics subcommands

**Files:**
- Modify: `eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `tokenizer.pt`, `world_model.pt`, dataset dir with `test_episodes/`.
- Produces: CLI subcommands:
  - `python eval.py rollout --tokenizer T --wm W --data D --out DIR [--episode I] [--prime 4] [--horizon 40] [--temperature 1.0]` → `rollout_epI.gif` (top row real, bottom row dream, red separator after priming) + `rollout_epI_mse.png`.
  - `python eval.py drift --tokenizer T --wm W --data D --out DIR [--episodes 20]` → `drift.png` (mean per-step pixel MSE, dream vs real) + `drift.json`.
  - `python eval.py loss-heatmap --tokenizer T --wm W --data D --out DIR` → `loss_heatmap.png` (mean CE per frame-token position, reshaped to the token grid).
- Produces: `eval.open_loop_rollout(tok, wm, frames, actions, prime, horizon, temperature, device) -> np.uint8 [T,64,64,3]` (reused by play/notebook).

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
import subprocess
import sys

import numpy as np

from tests.test_data import make_split


def test_dynamics_evals_end_to_end(tmp_path):
    make_split(tmp_path / "d", "train", n=120)
    make_split(tmp_path / "d", "val", n=60, seed=1)
    ep_dir = tmp_path / "d" / "test_episodes"; ep_dir.mkdir()
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
    assert (tmp_path / "ev" / "drift.png").exists()
    assert (tmp_path / "ev" / "loss_heatmap.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL (eval.py has no CLI)

- [ ] **Step 3: Extend eval.py**

Append:

```python
def load_models(tok_path, wm_path, device):
    from dataclasses import fields
    from model import GPTConfig, WorldModel
    ck = torch.load(tok_path, map_location=device)
    tok = VQVAE(grid=ck["grid"]).to(device).eval()
    tok.load_state_dict(ck["model"])
    wm = None
    if wm_path:
        wck = torch.load(wm_path, map_location=device)
        cfg = GPTConfig(**wck["config"])
        wm = WorldModel(cfg).to(device).eval()
        wm.load_state_dict(wck["model"])
    return tok, wm


@torch.no_grad()
def open_loop_rollout(tok, wm, frames, actions, prime, horizon, temperature, device,
                      seq_len=16, top_k=50):
    """Prime with real frames, then dream forward following the recorded actions.
    Returns uint8 [prime+horizon, 64, 64, 3] (primed frames included verbatim)."""
    from train_wm import interleave
    K = tok.tokens_per_frame
    x = torch.from_numpy(np.ascontiguousarray(frames[:prime]))
    x = x.permute(0, 3, 1, 2).float().div(255).to(device)
    toks = tok.encode(x).cpu().numpy().astype(np.int64)          # [prime, K]
    seq = interleave(toks, actions[:prime]).tolist()
    out = [frames[i] for i in range(prime)]
    cur = toks[-1:]
    for t in range(prime, prime + horizon):
        # keep the context to the most recent (seq_len - 1) frames
        max_ctx = (seq_len - 1) * (K + 1)
        ctx = torch.tensor(seq[-max_ctx:], device=device)[None]
        nxt = wm.generate_frame(ctx, int(actions[t]), tokens_per_frame=K,
                                temperature=temperature, top_k=top_k)
        seq += [512 + int(actions[t])] + nxt[0].tolist()
        img = to_uint8(tok.decode(nxt))[0]
        out.append(img)
    return np.stack(out)


def _add_common(sp):
    sp.add_argument("--tokenizer", required=True)
    sp.add_argument("--wm")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out", default="panels")
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sp.add_argument("--seq-len", type=int, default=16)
    sp.add_argument("--prime", type=int, default=4)
    sp.add_argument("--horizon", type=int, default=40)
    sp.add_argument("--temperature", type=float, default=1.0)


def main():
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train_wm import interleave, make_targets

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rollout", "drift", "loss-heatmap"):
        _add_common(sub.add_parser(name))
    sub.choices["rollout"].add_argument("--episode", type=int, default=0)
    sub.choices["drift"].add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok, wm = load_models(a.tokenizer, a.wm, a.device)
    eps = data_mod.list_episodes(a.data)

    if a.cmd == "rollout":
        ep = data_mod.load_episode(eps[a.episode])
        T = min(a.prime + a.horizon, len(ep["frames"]))
        dream = open_loop_rollout(tok, wm, ep["frames"], ep["actions"], a.prime,
                                  T - a.prime, a.temperature, a.device, a.seq_len)
        real = ep["frames"][:T]
        rows = []
        for t in range(T):
            top, bot = real[t], dream[t]
            sep = np.zeros((2, 64, 3), np.uint8)
            sep[:] = (255, 0, 0) if t >= a.prime else (0, 255, 0)
            rows.append(np.concatenate([top, sep, bot], axis=0))
        frames_out = [Image.fromarray(r).resize((256, 520), Image.NEAREST)
                      for r in rows]
        frames_out[0].save(out / f"rollout_ep{a.episode}.gif", save_all=True,
                           append_images=frames_out[1:], duration=120, loop=0)
        mse = ((real.astype(np.float32) - dream.astype(np.float32)) ** 2).mean((1, 2, 3))
        plt.figure(); plt.plot(mse); plt.axvline(a.prime, ls="--", c="r")
        plt.xlabel("frame"); plt.ylabel("pixel MSE"); plt.title("open-loop drift")
        plt.savefig(out / f"rollout_ep{a.episode}_mse.png", dpi=150); plt.close()

    elif a.cmd == "drift":
        curves = []
        for i in range(min(a.episodes, len(eps))):
            ep = data_mod.load_episode(eps[i])
            T = min(a.prime + a.horizon, len(ep["frames"]))
            if T <= a.prime:
                continue
            dream = open_loop_rollout(tok, wm, ep["frames"], ep["actions"], a.prime,
                                      T - a.prime, a.temperature, a.device, a.seq_len)
            real = ep["frames"][:T].astype(np.float32)
            curves.append(((real - dream.astype(np.float32)) ** 2).mean((1, 2, 3))[:T])
        L = min(len(c) for c in curves)
        mean = np.stack([c[:L] for c in curves]).mean(0)
        plt.figure(); plt.plot(mean); plt.axvline(a.prime, ls="--", c="r")
        plt.xlabel("frame"); plt.ylabel("pixel MSE")
        plt.title(f"open-loop drift, mean of {len(curves)} episodes")
        plt.savefig(out / "drift.png", dpi=150); plt.close()
        (out / "drift.json").write_text(json.dumps({"mean_mse": mean.tolist()}))

    elif a.cmd == "loss-heatmap":
        frames, actions, dones = data_mod.load_split(a.data, "val")
        K = tok.tokens_per_frame
        tokens = np.memmap(Path(a.data) / "val" / "tokens.bin", dtype=np.uint16,
                           mode="r", shape=(len(frames), K))
        starts = data_mod.valid_starts(dones, a.seq_len)[:128]
        ce = torch.zeros(K, device=a.device); n = 0
        for s in starts:
            seq = interleave(tokens[s:s + a.seq_len].astype(np.int64),
                             np.asarray(actions[s:s + a.seq_len]))
            x = torch.tensor(seq[:-1], device=a.device)[None]
            y = torch.tensor(make_targets(seq), device=a.device)[None]
            with torch.no_grad():
                logits, _ = wm(x, None)
            lt = torch.nn.functional.cross_entropy(
                logits[0], y[0].clamp(min=0), reduction="none")
            mask = (y[0] >= 0).float()
            # positions of frame-token predictions within each frame block
            lt = (lt * mask).view(a.seq_len, K + 1)[1:, 1:]   # skip first frame & action col
            ce += lt.mean(0); n += 1
        grid = int(K ** 0.5)
        hm = (ce / n).view(grid, grid).cpu().numpy()
        plt.figure(figsize=(4, 4)); plt.imshow(hm, cmap="magma"); plt.colorbar()
        plt.title("mean CE per token position")
        plt.savefig(out / "loss_heatmap.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
```

Careful with the loss-heatmap reshape: `seq[:-1]` has length `L*(K+1)-1`; per-position CE must be realigned before `.view(seq_len, K+1)` — prepend one zero: `lt = torch.cat([torch.zeros(1, device=a.device), lt * mask]).view(a.seq_len, K + 1)[1:, 1:]`. Same for `mask` alignment. Implement it that way.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: 1 passed (~2–4 min CPU)

- [ ] **Step 5: Commit**

```bash
git add eval.py tests/test_eval.py
git commit -m "feat: rollout, drift, and loss-heatmap evaluation"
```

---

### Task 10: [SERVER — AFTER KEVIN'S TOKENIZER SIGN-OFF] World-model training + eval artifacts

**Files:**
- Create: `docs/notes/wm-training-notes.md`

- [ ] **Step 1: Confirm gate** — Kevin has approved the tokenizer panels (Task 6) and the grid choice. If grid=16 was chosen, pass `--seq-len 8` to keep block_size = 8×257 = 2056 manageable, and note VRAM impact.

- [ ] **Step 2: Pre-compute tokens and launch training**

```bash
rsync -av --exclude .venv --exclude data --exclude .git . $WM_SERVER:~/nwm/repo/
ssh $WM_SERVER 'cd ~/nwm/repo && ~/nwm/venv/bin/python train_wm.py --data data/chaser \
  --tokenizer out/tok8/tokenizer.pt --out out/wm --prepare'
ssh $WM_SERVER 'cd ~/nwm/repo && tmux new -d -s wm \
  "~/nwm/venv/bin/python train_wm.py --data data/chaser --tokenizer out/tok8/tokenizer.pt \
   --out out/wm --steps 40000 --bs 24 2>&1 | tee out/wm.log"'
```
Monitor val loss; expect steady decrease and final teacher-forced val CE roughly in the 0.2–0.8 range (Chaser frames are mostly copy-able; exact value recorded, not gated).

- [ ] **Step 3: Generate eval artifacts**

```bash
ssh $WM_SERVER 'cd ~/nwm/repo && for c in rollout drift loss-heatmap; do \
  ~/nwm/venv/bin/python eval.py $c --tokenizer out/tok8/tokenizer.pt \
  --wm out/wm/world_model.pt --data data/chaser --out panels/wm; done'
rsync -av $WM_SERVER:~/nwm/repo/panels/wm/ panels/wm/
```
Also run `eval.py rollout` for 3–4 different episodes and at temperatures 0.5 / 1.0.

- [ ] **Step 4: Quality bar + share with Kevin (soft checkpoint)**

Acceptance to proceed: dreams stay maze-coherent for ≥ 20 open-loop steps; the agent sprite responds to actions; drift curve grows smoothly (no immediate explosion). Record hparams, losses, and observations in `docs/notes/wm-training-notes.md`; send the GIFs + curves to Kevin (non-blocking FYI unless quality is bad — if bad, STOP and debug with systematic-debugging before proceeding).

- [ ] **Step 5: Commit**

```bash
git add docs/notes/wm-training-notes.md
git commit -m "docs: world-model training results"
```

---

### Task 11: play.py (interactive dream)

**Files:**
- Create: `play.py`
- Test: manual (interactive); logic reuses `eval.open_loop_rollout` machinery already under test.

**Interfaces:**
- Consumes: `tokenizer.pt`, `world_model.pt`, dataset `test_episodes/` (for priming).
- Produces: `python play.py --tokenizer T --wm W --data D [--device D] [--temperature 1.0] [--seq-len 16]` — pygame window, arrow keys, R=reset, -/+ temperature, ESC quit.

- [ ] **Step 1: Implement play.py**

```python
"""Play inside the dream: the world model hallucinates each next frame from
your keystrokes. Arrow keys move (procgen action combos), R resets to a fresh
real-frame priming, -/+ adjust sampling temperature, ESC quits."""
import argparse
import time

import numpy as np
import pygame
import torch

import data as data_mod
from eval import load_models, to_uint8
from model import FRAME_VOCAB
from train_wm import interleave

# procgen's 15 discrete actions are (LEFT/RIGHT) x (DOWN/UP) combos + special keys.
# index: 0 (L,D) 1 (L) 2 (L,U) 3 (D) 4 noop 5 (U) 6 (R,D) 7 (R) 8 (R,U) 9-14 specials
def action_from_keys(keys):
    lr = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])   # -1, 0, 1
    ud = (keys[pygame.K_UP] - keys[pygame.K_DOWN])
    return {(-1, -1): 0, (-1, 0): 1, (-1, 1): 2, (0, -1): 3, (0, 0): 4,
            (0, 1): 5, (1, -1): 6, (1, 0): 7, (1, 1): 8}[(lr, ud)]

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--wm", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--seq-len", type=int, default=16)
ap.add_argument("--prime", type=int, default=4)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
a = ap.parse_args()

tok, wm = load_models(a.tokenizer, a.wm, a.device)
K = tok.tokens_per_frame
eps = data_mod.list_episodes(a.data)
rng = np.random.default_rng()

def reset():
    ep = data_mod.load_episode(eps[rng.integers(len(eps))])
    x = torch.from_numpy(np.ascontiguousarray(ep["frames"][:a.prime]))
    x = x.permute(0, 3, 1, 2).float().div(255).to(a.device)
    toks = tok.encode(x).cpu().numpy().astype(np.int64)
    seq = interleave(toks, ep["actions"][:a.prime]).tolist()
    return seq, ep["frames"][a.prime - 1]

pygame.init()
screen = pygame.display.set_mode((512, 512))
seq, frame = reset()
temp, running = a.temperature, True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_r:
                seq, frame = reset()
            elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                temp = max(0.1, temp - 0.1)
            elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                temp = min(2.0, temp + 0.1)
    act = action_from_keys(pygame.key.get_pressed())
    t0 = time.time()
    max_ctx = (a.seq_len - 1) * (K + 1)
    ctx = torch.tensor(seq[-max_ctx:], device=a.device)[None]
    with torch.no_grad():
        nxt = wm.generate_frame(ctx, act, tokens_per_frame=K, temperature=temp)
    seq += [FRAME_VOCAB + act] + nxt[0].tolist()
    if len(seq) > 4 * a.seq_len * (K + 1):        # keep memory bounded
        seq = seq[-2 * max_ctx:]
    frame = to_uint8(tok.decode(nxt))[0]
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(pygame.transform.scale(surf, (512, 512)), (0, 0))
    pygame.display.flip()
    pygame.display.set_caption(
        f"nano-world-model — {1000 * (time.time() - t0):.0f} ms/frame  temp {temp:.1f}")
pygame.quit()
```

- [ ] **Step 2: Manual test on the Mac**

```bash
rsync -av $WM_SERVER:~/nwm/repo/out/tok8/tokenizer.pt out/tok8/
rsync -av $WM_SERVER:~/nwm/repo/out/wm/world_model.pt out/wm/
.venv/bin/python play.py --tokenizer out/tok8/tokenizer.pt --wm out/wm/world_model.pt --data data/chaser
```
(Data locally via `data.download()` first.) Verify: window opens, agent responds to arrows, R re-primes, temperature keys work, ms/frame displayed (expect ~100–500 ms on MPS/CPU — acceptable; note actual number).

- [ ] **Step 3: Commit**

```bash
git add play.py
git commit -m "feat: interactive play-in-the-dream demo"
```

---

### Task 12: walkthrough.ipynb (Colab)

**Files:**
- Create: `walkthrough.ipynb`

**Interfaces:**
- Consumes: everything published; HF checkpoints from Task 13 Step 1 (create checkpoints first, then finish notebook — or use rsync'd local ckpts and swap the cell to HF paths in Task 13).

- [ ] **Step 1: Build the notebook** with exactly these cells:

1. **Markdown intro** — what a world model is, the two components, link to README.
2. `!git clone <repo> && %cd nano-world-model && !pip -q install -r requirements.txt`
3. Data: `import data; root = data.download()` + show 8 random frames with `eval.save_grid` → display inline.
4. Markdown: tokenizer explanation (frames→tokens; the "why discrete?" FAQ condensed).
5. **Fast path**: download published checkpoints via `huggingface_hub.hf_hub_download(f"{NS}/nano-world-model", "tokenizer.pt")` (+ `world_model.pt`).
6. Tokenizer sanity: encode/decode 4 val frames, show orig|recon pairs inline.
7. Markdown: dynamics model explanation (interleaved sequence diagram, AR sampling, the four-question FAQ condensed).
8. Rollout: `eval.open_loop_rollout(...)` on a test episode → display GIF inline.
9. **Slow path (optional, collapsed section)**: `!python train_tokenizer.py --steps 20000 --bs 128 ...` and `!python train_wm.py --prepare && !python train_wm.py --steps 30000 --bs 16 ...` with T4 time estimates printed.
10. **Interactive play**: ipywidgets — an `Image` widget + five `Button`s (←, ↑, ↓, →, noop) + reset button + temperature slider; button callback runs one `generate_frame` step and updates the image (PNG bytes via PIL). Uses the same seq/context management as `play.py` (~40 lines, self-contained in the cell).
11. Markdown: exercises (one-shot decoding ablation, temperature sweep, grid=16, drift curve reading) + pointers to eval.py.

- [ ] **Step 2: Test locally**

Run: `.venv/bin/pip install jupyter && .venv/bin/jupyter nbconvert --to notebook --execute walkthrough.ipynb --output /tmp/wt_test.ipynb` with the slow-path cells tagged `skip-execution` (nbconvert honors cell tags) or guarded by `if False`. Expected: executes cleanly end-to-end with the fast path.

- [ ] **Step 3: Colab verification (manual)** — upload to Colab on a T4, run fast path top to bottom; confirm the button UI is responsive (~0.5–2 s/step on T4 is fine). Fix anything Colab-specific (pygame is NOT imported by the notebook path; verify `pip install` cell stays under 3 min).

- [ ] **Step 4: Commit**

```bash
git add walkthrough.ipynb
git commit -m "feat: Colab walkthrough notebook with button-driven dream play"
```

---

### Task 13: Publish checkpoints + README + release checklist

**Files:**
- Create: `README.md`, `LICENSE` (MIT), `scripts/publish_checkpoints.py`

- [ ] **Step 1: Publish checkpoints**

`scripts/publish_checkpoints.py`:
```python
"""Upload trained checkpoints to the HF model repo. Usage: python scripts/publish_checkpoints.py TOK_CKPT WM_CKPT"""
import os
import sys

from huggingface_hub import HfApi

ns = os.environ["HF_NAMESPACE"]
api = HfApi()
repo = f"{ns}/nano-world-model"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo="tokenizer.pt", repo_id=repo)
api.upload_file(path_or_fileobj=sys.argv[2], path_in_repo="world_model.pt", repo_id=repo)
print(f"published to {repo}")
```
Run on server against `out/tok8/tokenizer.pt` and `out/wm/world_model.pt`. Update the notebook fast-path cell to these repo paths.

- [ ] **Step 2: Write README.md** with this exact structure (prose written fresh, drawing on the spec and the brainstorming FAQ):

1. Title + one-line pitch + hero GIF (a `rollout_ep*.gif` + a screen capture of play.py).
2. **What is a world model?** (3 paragraphs, action-conditioned prediction, why it matters for agents/RL/robotics.)
3. **The idea in one diagram**: frames → VQ tokens → GPT over `[a, z…]` → sampled tokens → decoded frames.
4. **Quickstart** (copy-paste): install, `data.download()`, download checkpoints, `play.py` command, Colab badge link to walkthrough.ipynb.
5. **Train it yourself**: the two training commands with T4 wall-clock numbers (fill with actuals from Tasks 6/10).
6. **How it works**: tokenizer section, dynamics section, sampling section — each pointing at the ~30 relevant lines of source.
7. **FAQ — the four questions** (from the spec's conceptual spine, ~150 words each): why discrete tokens; why sample; why autoregressive within a frame (beliefs vs outcomes); why 64 tokens not 1 (the exponential tradeoff table).
8. **What to look at**: eval artifacts explained (drift curve, loss heatmap "most tokens copy, sprites are hard", codebook usage).
9. **Exercises**: one-shot ablation, temperature, grid=16, MaskGIT reading pointer.
10. **Limitations & extensions**: LAM/IDM, MaskGIT decoding, diffusion head, own data collection (from spec Future Extensions).
11. **Acknowledgments & licenses**: MIT for code; gen_dgrl CC-BY-NC attribution for data; IRIS/Genie/DIAMOND/nanoGPT citations.

- [ ] **Step 3: Final end-to-end verification (fresh environment)**

```bash
ssh $WM_SERVER 'cd /tmp && rm -rf nwm_check && mkdir nwm_check && cd nwm_check && \
  python3 -m venv v && git clone ~/nwm/repo repo && cd repo && \
  ../v/bin/pip -q install -r requirements.txt && \
  ../v/bin/python -c "import data; data.download(root=\"d\")" && \
  ../v/bin/python train_tokenizer.py --data d --out o --steps 50 --bs 32 --save-every 50 && \
  ../v/bin/python -c "print(\"fresh-env OK\")"'
```
Expected: `fresh-env OK`. Also re-run the full local suite: `.venv/bin/pytest tests/ -v` (including slow) → all pass.

- [ ] **Step 4: Commit + release gate**

```bash
git add README.md LICENSE scripts/publish_checkpoints.py
git commit -m "feat: README, license, checkpoint publishing"
```

**RELEASE GATE (Kevin decides):** GitHub publication (repo name, org), flipping HF dataset + model repos public, README hero GIF selection. Do not publish anywhere without explicit instruction.

---

## Self-Review Notes

- **Spec coverage:** tokenizer (T2/T3/T6), review gate (T6 HARD PAUSE), WM + masked loss + KV cache (T7), training (T8/T10), all five pedagogical eval artifacts (T6 panels, T9 rollout/drift/heatmap), play.py + notebook button UI (T11/T12), pretrained checkpoints (T13), FAQ in README (T13), data verification gate + curation + HF + license (T4/T5), overfit modes (T3/T8), fresh-env check (T13). Future extensions intentionally absent (spec defers them).
- **Types:** checkpoint dicts, `interleave`/`make_targets`, `valid_starts`, `open_loop_rollout`, and CLI flags are used consistently across tasks; `tokens.bin` is uint16 everywhere; action token id is `FRAME_VOCAB + a` everywhere.
- **Known judgment calls:** `load_episodes` in Task 4 is deliberately investigation-shaped (external-world unknown), with GO/FALLBACK criteria fixed in advance. Task 8 includes an explicit in-plan correction of the val-batch call signature — executor must apply it.

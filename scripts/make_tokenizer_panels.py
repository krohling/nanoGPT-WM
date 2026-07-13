"""Build the tokenizer review panels (spec: the tokenizer -> WM review gate).

Usage: python scripts/make_tokenizer_panels.py CKPT DATA_DIR OUT_DIR

Panels: (a) random orig|recon pairs, (b) worst-32 hard cases by MSE,
(c) codebook usage histogram, (d) mean per-pixel error heatmap.
Prints summary stats for the review note.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data as data_mod  # noqa: E402
from eval import save_grid, to_uint8, codebook_stats  # noqa: E402
from model import VQVAE  # noqa: E402

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
        mses.append(((r.clamp(0, 1) - x) ** 2).mean(dim=(1, 2, 3)).cpu())
        idxs.append(idx)
mse = torch.cat(mses)


def pair_grid(ids, path):
    x = torch.stack([val[int(i)] for i in ids]).to(dev)
    with torch.no_grad():
        r, _, _ = m(x)
    pair = np.concatenate([to_uint8(x), to_uint8(r)], axis=2)  # orig | recon
    save_grid(pair, path, ncol=4)


pair_grid(np.random.default_rng(0).choice(N, 32, replace=False),
          out / "panel_random.png")
pair_grid(mse.topk(32).indices.numpy(), out / "panel_hard.png")

usage, ppl = codebook_stats(idxs)
plt.figure(figsize=(10, 3))
plt.bar(range(len(usage)), np.sort(usage)[::-1], width=1.0)
plt.title(f"codebook usage (sorted) — perplexity {ppl:.0f}, "
          f"{(usage > 0).sum()}/{len(usage)} codes used")
plt.tight_layout()
plt.savefig(out / "panel_codebook.png", dpi=150)
plt.close()

with torch.no_grad():
    err = torch.zeros(64, 64)
    n = min(2000, N)
    for s in range(0, n, 256):
        x = torch.stack([val[i] for i in range(s, min(s + 256, n))]).to(dev)
        r, _, _ = m(x)
        err += (r.clamp(0, 1) - x).abs().mean(dim=(0, 1)).cpu() * len(x)
    err /= n
plt.figure(figsize=(4, 4))
plt.imshow(err.numpy(), cmap="magma")
plt.colorbar()
plt.title("mean |error| per pixel")
plt.tight_layout()
plt.savefig(out / "panel_pixel_error.png", dpi=150)
plt.close()
print(f"val MSE mean {mse.mean():.5f}  p95 {mse.quantile(0.95):.5f}  "
      f"worst {mse.max():.5f}  perplexity {ppl:.0f}  "
      f"codes used {(usage > 0).sum()}/{len(usage)}")

"""Stage 1: train the VQ-VAE tokenizer. ~30k steps fits in <1h on a T4."""
import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
# nanoGPT idiom: no DataLoader, just random reads from the memmap
frames, _, _ = data_mod.load_split(args.data, "train")
n_frames = min(256, len(frames)) if args.overfit else len(frames)
rng = np.random.default_rng(0)


def get_batch():
    ix = rng.integers(0, n_frames, args.bs)
    x = torch.from_numpy(frames[ix].astype(np.float32) / 255.0)
    return x.permute(0, 3, 1, 2).to(args.device, non_blocking=True)


val = data_mod.FrameDataset(args.data, "val")
val_batch = torch.stack([val[i] for i in range(min(32, len(val)))]).to(args.device)

model = VQVAE(grid=args.grid).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
use_amp = args.device == "cuda"
scaler = torch.amp.GradScaler(args.device, enabled=use_amp)
autocast = (lambda: torch.amp.autocast("cuda")) if use_amp else contextlib.nullcontext

step, t0 = 0, time.time()
while step < args.steps:
    x = get_batch()
    with autocast():
        recon, commit, idx = model(x)
        loss = F.mse_loss(recon, x) + args.beta * commit
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
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
        # each tile: original | reconstruction, side by side (64x128)
        pair = np.concatenate([to_uint8(val_batch), to_uint8(r)], axis=2)
        save_grid(pair[:32], out / f"recon_{step:06d}.png", ncol=4)
        torch.save({"model": model.state_dict(), "grid": args.grid, "step": step},
                   out / "tokenizer.pt")
        model.train()
print("done")

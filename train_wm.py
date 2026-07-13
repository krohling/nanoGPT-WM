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


if __name__ == "__main__":
    import argparse
    import contextlib
    import math
    import time
    from dataclasses import asdict
    from pathlib import Path

    import torch

    import data as data_mod
    from model import VQVAE, GPTConfig, WorldModel

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
        # tokenize every frame once; the GPT then trains on integers only
        for split in ("train", "val"):
            frames, _, _ = data_mod.load_split(args.data, split)
            out_path = Path(args.data) / split / "tokens.bin"
            mm = np.memmap(out_path, dtype=np.uint16, mode="w+",
                           shape=(len(frames), K))
            for s in range(0, len(frames), 512):
                x = torch.from_numpy(frames[s:s + 512].astype(np.float32) / 255.0)
                x = x.permute(0, 3, 1, 2).to(dev)
                mm[s:s + len(x)] = tok.encode(x).cpu().numpy().astype(np.uint16)
                if s % 51200 == 0:
                    print(f"{split}: {s}/{len(frames)}", flush=True)
            mm.flush()
        print("prepared")
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
    va_fixed = va_starts[rng.choice(len(va_starts), min(256, len(va_starts)),
                                    replace=False)]

    def batch(tokens, actions, idx):
        xs, ys = [], []
        for s in idx:
            seq = interleave(tokens[s:s + args.seq_len].astype(np.int64),
                             actions[s:s + args.seq_len])
            xs.append(seq[:-1])
            ys.append(make_targets(seq))
        return (torch.tensor(np.stack(xs), device=dev),
                torch.tensor(np.stack(ys), device=dev))

    cfg = GPTConfig(block_size=args.seq_len * (K + 1) - 1, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = WorldModel(cfg).to(dev)
    print(f"params: {sum(pm.numel() for pm in model.parameters()) / 1e6:.1f}M  "
          f"block_size: {cfg.block_size}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(dev, enabled=use_amp)
    autocast = (lambda: torch.amp.autocast("cuda")) if use_amp else contextlib.nullcontext
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))

    best, t0 = float("inf"), time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(tr_tokens, tr_actions, rng.choice(tr_starts, size=args.bs))
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % args.log_every == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{args.log_every / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(va_fixed), args.bs):
                    x, y = batch(va_tokens, va_actions, va_fixed[s:s + args.bs])
                    _, l = model(x, y)
                    vl += l.item() * len(x)
                    n += len(x)
            vl /= max(n, 1)
            print(f"step {step}: val loss {vl:.4f}", flush=True)
            ckpt = {"model": model.state_dict(), "config": asdict(cfg),
                    "tokens_per_frame": K, "step": step, "val_loss": vl}
            if vl < best:      # world_model.pt is always the BEST val checkpoint
                best = vl
                torch.save(ckpt, out / "world_model.pt")
            torch.save(ckpt, out / "world_model_last.pt")
            model.train()
    print("done")

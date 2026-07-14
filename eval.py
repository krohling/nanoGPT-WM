"""Evaluation and visualization for nano-world-model.

Helpers (save_grid, to_uint8, codebook_stats) are imported by the train
scripts; the CLI subcommands for dynamics evaluation live in main() below.
"""
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
    """frames: uint8 [N,H,W,3] (any tile size) -> one PNG grid."""
    n = len(frames)
    nrow = (n + ncol - 1) // ncol
    th, tw = frames[0].shape[:2]
    H, W = th * scale, tw * scale
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


# --------------------------- dynamics evaluation ----------------------------

def load_models(tok_path, wm_path, device):
    from model import GPTConfig, WorldModel
    ck = torch.load(tok_path, map_location=device)
    tok = VQVAE(grid=ck["grid"]).to(device).eval()
    tok.load_state_dict(ck["model"])
    wm = None
    if wm_path:
        wck = torch.load(wm_path, map_location=device)
        wm = WorldModel(GPTConfig(**wck["config"])).to(device).eval()
        wm.load_state_dict(wck["model"])
    return tok, wm


def _crop_to_frames(seq, K, max_frames):
    """Drop the oldest (frame, action) chunks so at most max_frames frames
    remain. The sequence must keep starting on a frame boundary — absolute
    positions are only meaningful to the model if position 0 is a frame start."""
    n_frames = (len(seq) + 1) // (K + 1)
    if n_frames > max_frames:
        seq = seq[(n_frames - max_frames) * (K + 1):]
    return seq


@torch.no_grad()
def open_loop_rollout(tok, wm, frames, actions, prime, horizon, temperature, device,
                      seq_len=16, top_k=50):
    """Prime with real frames, then dream forward following the recorded actions.

    frames/actions follow the dataset convention: actions[t] is taken at
    frames[t] and produces frames[t+1]. Dreamed frame t is therefore
    conditioned on actions[t-1]. Returns uint8 [prime+horizon, 64, 64, 3]
    (primed frames included verbatim).
    """
    from train_wm import interleave
    K = tok.tokens_per_frame
    x = torch.from_numpy(frames[:prime].astype(np.float32) / 255.0)
    x = x.permute(0, 3, 1, 2).to(device)
    toks = tok.encode(x).cpu().numpy().astype(np.int64)          # [prime, K]
    seq = interleave(toks, np.asarray(actions[:prime])).tolist()  # ends with a frame
    out = [frames[i] for i in range(prime)]
    for t in range(prime, prime + horizon):
        seq = _crop_to_frames(seq, K, seq_len - 1)
        ctx = torch.tensor(seq, device=device)[None]
        a = int(actions[t - 1])
        nxt = wm.generate_frame(ctx, a, tokens_per_frame=K,
                                temperature=temperature, top_k=top_k)
        seq += [FRAME_VOCAB + a] + nxt[0].tolist()
        out.append(to_uint8(tok.decode(nxt))[0])
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, wm = load_models(a.tokenizer, a.wm, a.device)
    eps = data_mod.list_episodes(a.data)

    if a.cmd == "rollout":
        ep = data_mod.load_episode(eps[a.episode])
        T = min(a.prime + a.horizon, len(ep["frames"]))
        dream = open_loop_rollout(tok, wm, ep["frames"], ep["actions"], a.prime,
                                  T - a.prime, a.temperature, a.device, a.seq_len)
        real = ep["frames"][:T]
        tiles = []
        for t in range(T):
            sep = np.zeros((2, 64, 3), np.uint8)
            sep[:] = (0, 255, 0) if t < a.prime else (255, 0, 0)
            tiles.append(np.concatenate([real[t], sep, dream[t]], axis=0))
        imgs = [Image.fromarray(r).resize((256, 520), Image.NEAREST) for r in tiles]
        imgs[0].save(out / f"rollout_ep{a.episode}.gif", save_all=True,
                     append_images=imgs[1:], duration=120, loop=0)
        mse = ((real.astype(np.float32) - dream.astype(np.float32)) ** 2).mean((1, 2, 3))
        plt.figure()
        plt.plot(mse)
        plt.axvline(a.prime - 0.5, ls="--", c="r")
        plt.xlabel("frame")
        plt.ylabel("pixel MSE")
        plt.title("open-loop drift (real vs dream, same actions)")
        plt.savefig(out / f"rollout_ep{a.episode}_mse.png", dpi=150)
        plt.close()

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
            curves.append(((real - dream.astype(np.float32)) ** 2).mean((1, 2, 3)))
        L = min(len(c) for c in curves)
        mean = np.stack([c[:L] for c in curves]).mean(0)
        plt.figure()
        plt.plot(mean)
        plt.axvline(a.prime - 0.5, ls="--", c="r")
        plt.xlabel("frame")
        plt.ylabel("pixel MSE")
        plt.title(f"open-loop drift, mean of {len(curves)} episodes")
        plt.savefig(out / "drift.png", dpi=150)
        plt.close()
        (out / "drift.json").write_text(json.dumps({"mean_mse": mean.tolist()}))

    elif a.cmd == "loss-heatmap":
        frames, actions, dones = data_mod.load_split(a.data, "val")
        K = tok.tokens_per_frame
        tokens = np.memmap(Path(a.data) / "val" / "tokens.bin", dtype=np.uint16,
                           mode="r", shape=(len(frames), K))
        starts = data_mod.valid_starts(dones, a.seq_len)[:128]
        ce = torch.zeros(K, device=a.device)
        n = 0
        for s in starts:
            seq = interleave(tokens[s:s + a.seq_len].astype(np.int64),
                             np.asarray(actions[s:s + a.seq_len]))
            x = torch.tensor(seq[:-1], device=a.device)[None]
            y = torch.tensor(make_targets(seq), device=a.device)[None]
            with torch.no_grad():
                # nanoGPT returns full-sequence logits only when targets are
                # given (else just the last position) — pass y, ignore its loss
                logits, _ = wm(x, y)
            lt = torch.nn.functional.cross_entropy(
                logits[0], y[0].clamp(min=0), reduction="none") * (y[0] >= 0)
            # realign so row t holds the CE of frame t's K tokens: seq position
            # j is predicted by lt[j-1]; pad front (position 0 has no predictor)
            # and back (the dropped trailing slot) to length L*(K+1)
            pad = torch.zeros(1, device=a.device)
            full = torch.cat([pad, lt, pad]).view(a.seq_len, K + 1)
            ce += full[1:, :K].mean(0)   # skip frame 0 (mostly unpredictable)
            n += 1
        grid = int(K ** 0.5)
        hm = (ce / n).view(grid, grid).cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(hm, cmap="magma")
        plt.colorbar()
        plt.title("mean CE per token position")
        plt.savefig(out / "loss_heatmap.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()

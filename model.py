"""nano-world-model: a VQ-VAE tokenizer and a GPT dynamics model.

The tokenizer turns each 64x64 RGB frame into a small grid of discrete codes
("the frame as 64 words"). The dynamics model is nanoGPT — literally: the
transformer in nanogpt.py is vendored verbatim from Karpathy's repo and used
unmodified. Everything world-model-specific lives here: what the vocabulary
means (512 frame codes + 15 action tokens), the sampling loop that restricts
frame generation to frame tokens, and an optional KV-cached fast path for
real-time play.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt import GPT, GPTConfig  # GPTConfig re-exported for train/eval/tests

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

    def __init__(self, num_codes, dim, decay=0.99, eps=1e-5, stale_after=25):
        super().__init__()
        self.num_codes, self.dim, self.decay, self.eps = num_codes, dim, decay, eps
        self.stale_after = stale_after
        self.register_buffer("codebook", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", self.codebook.clone())
        # data-dependent init: the random codebook above is replaced by real
        # encoder outputs on the first training batch (prevents collapse where
        # every vector maps to whichever random code sits nearest the cluster)
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("unused_steps", torch.zeros(num_codes))

    @torch.no_grad()
    def _reseed(self, flat, mask):
        """Re-seed masked codes from live encoder outputs (with replacement)."""
        n = int(mask.sum())
        take = flat[torch.randint(0, flat.shape[0], (n,), device=flat.device)]
        take = take + torch.randn_like(take) * (flat.std() * 0.02)  # break duplicates
        self.codebook[mask] = take
        self.ema_sum[mask] = take
        self.ema_count[mask] = 1.0
        self.unused_steps[mask] = 0

    def forward(self, z):  # z: [B, D, H, W]
        B, D, H, W = z.shape
        # fp32 throughout the VQ bookkeeping: under CUDA autocast z arrives as
        # fp16, but codebook/EMA buffers are fp32 (and should stay fp32)
        z = z.float()
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)                     # [N, D]
        if self.training and not bool(self.initialized):
            self._reseed(flat.detach(), torch.ones(self.num_codes, dtype=torch.bool,
                                                   device=flat.device))
            self.initialized.fill_(True)
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
                # stale-code re-seeding: codes unused for `stale_after` steps get
                # moved onto live encoder outputs, keeping the whole codebook alive
                self.unused_steps[count > 0] = 0
                self.unused_steps[count == 0] += 1
                stale = self.unused_steps > self.stale_after
                if stale.any():
                    self._reseed(flat.detach(), stale)

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
        # NOTE: no sigmoid — MSE on a linear output keeps gradients alive on
        # saturated (pure black/white) pixels; clamp to [0,1] only for display.
        recon = self.decoder(quant)
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
        return self.decoder(emb.permute(0, 3, 1, 2)).clamp(0, 1)


# ----------------------------- world model ----------------------------------
# The dynamics model IS nanoGPT (see nanogpt.py — vendored, unmodified).
# WorldModel is a thin wrapper that owns a GPT and adds the one thing a world
# model needs beyond language modeling: generate_frame, which appends an
# action token and samples the 64 tokens of the frame that action causes,
# restricting the softmax to the frame vocabulary.
#
# Everything else is convention, not architecture:
#   * vocab 0..511  = tokenizer codes, 512..526 = the 15 procgen actions
#   * training targets put -1 at positions whose target is an action token —
#     nanoGPT's own cross_entropy(ignore_index=-1) skips them (the model
#     predicts the world, not the player's mind)


class WorldModel(nn.Module):
    """Verbatim nanoGPT + a frame-sampling loop."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gpt = GPT(cfg)

    def forward(self, idx, targets=None):
        # NOTE nanoGPT semantics: with targets=None only the LAST position's
        # logits are returned (inference optimization). Pass targets to get
        # logits for every position.
        return self.gpt(idx, targets)

    @torch.no_grad()
    def generate_frame(self, ctx, action, tokens_per_frame=64,
                       temperature=1.0, top_k=50):
        """Append an action token, then sample one full frame, token by token.

        This is nanoGPT's generate() with one twist: logits are restricted to
        the frame vocabulary (an action can never appear inside a frame).
        Each step re-runs the full forward — clear, correct, and ~10x slower
        than KVSampler below, which computes the exact same distributions.
        """
        dev = ctx.device
        idx = torch.cat([ctx, torch.tensor([[FRAME_VOCAB + action]], device=dev)], 1)
        for _ in range(tokens_per_frame):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else \
                idx[:, -self.cfg.block_size:]
            logits, _ = self.gpt(idx_cond)             # [B, 1, vocab] (last pos)
            logits = logits[:, -1, :FRAME_VOCAB] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, tok], dim=1)
        return idx[:, -tokens_per_frame:]


# ------------------- OPTIONAL: KV-cached sampling (fast path) ---------------
# Skip this on first read — it contains no new ideas, only speed.
#
# nanoGPT's modules are stateless: attention computes q,k,v from the current
# input and nothing else, so generating N tokens costs N full forward passes.
# A KV cache remembers each layer's keys/values so every new token costs one
# token's worth of compute. That state has no place in nanoGPT's forward()
# signatures — and rather than subclass-and-override every method, KVSampler
# drives the SAME vendored modules (same weights, same math) in a cached loop:
# c_attn/c_proj/mlp/LayerNorms are called directly, only the orchestration
# differs. Two subtleties the cache introduces (tests pin both):
#   * position embeddings index from the cache length, not from 0
#   * is_causal=True is wrong once queries and keys have different lengths —
#     the mask must be a tril offset by the cache length
# test_kv_sampler_matches_full_forward asserts logit equality with plain GPT.

class KVSampler:
    """KV-cached frame sampler over an (unmodified) nanogpt.GPT."""

    def __init__(self, gpt):
        self.gpt = gpt
        self.reset()

    def reset(self):
        self.kv = [None] * len(self.gpt.transformer.h)
        self.pos = 0

    @torch.no_grad()
    def forward(self, idx):
        """Cached forward over new tokens idx [B, s]. Returns [B, s, vocab]."""
        g, cfg = self.gpt.transformer, self.gpt.config
        B, s = idx.shape
        nh, hs = cfg.n_head, cfg.n_embd // cfg.n_head
        pos = torch.arange(self.pos, self.pos + s, device=idx.device)
        x = g.drop(g.wte(idx) + g.wpe(pos))
        for i, blk in enumerate(g.h):
            xa = blk.ln_1(x)
            q, k, v = blk.attn.c_attn(xa).split(cfg.n_embd, dim=2)
            q, k, v = (t.view(B, s, nh, hs).transpose(1, 2) for t in (q, k, v))
            if self.kv[i] is not None:
                k = torch.cat([self.kv[i][0], k], dim=2)
                v = torch.cat([self.kv[i][1], v], dim=2)
            self.kv[i] = (k, v)
            past = k.size(2) - s
            mask = torch.ones(s, k.size(2), dtype=torch.bool,
                              device=idx.device).tril_(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            y = y.transpose(1, 2).contiguous().view(B, s, cfg.n_embd)
            x = x + blk.attn.resid_dropout(blk.attn.c_proj(y))
            x = x + blk.mlp(blk.ln_2(x))
        self.pos += s
        return self.gpt.lm_head(g.ln_f(x))

    @torch.no_grad()
    def generate_frame(self, ctx, action, tokens_per_frame=64,
                       temperature=1.0, top_k=50):
        """Same contract (and same distributions) as WorldModel.generate_frame."""
        dev = ctx.device
        self.reset()
        seq = torch.cat([ctx, torch.tensor([[FRAME_VOCAB + action]], device=dev)], 1)
        logits = self.forward(seq)                     # prefill the cache
        out = []
        for _ in range(tokens_per_frame):
            lg = logits[:, -1, :FRAME_VOCAB] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("Inf")
            tok = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
            out.append(tok)
            logits = self.forward(tok)                 # one cached step
        return torch.cat(out, dim=1)

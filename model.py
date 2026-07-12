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

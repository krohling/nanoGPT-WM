---
license: cc-by-nc-4.0
pretty_name: nano-world-model procgen Chaser frames + actions
---
# nano-world-model: procgen Chaser frames + actions

Curated subset of the offline procgen dataset released with
*"The Generalization Gap in Offline Reinforcement Learning"* (Mediratta et al.,
ICLR 2024, https://github.com/facebookresearch/gen_dgrl), license CC-BY-NC 4.0.
Frames were collected by PPO agents playing procgen **Chaser** at 64×64; this
subset interleaves the release's expert and suboptimal variants 50/50 for
state–action coverage.

Made for **[nanoGPT-WM](https://github.com/krohling/nanoGPT-WM)** — an
educational, minimal action-conditioned world model (nanoGPT for world
models). Non-commercial use only, per the upstream license.

## Format (memmap-friendly flat binaries)

- `train/`, `val/` — `frames.bin` (uint8, N×64×64×3), `actions.bin` (uint8, N),
  `dones.bin` (uint8, N), `meta.json` (`num_frames` + conventions).
  `actions[t]` is taken at `frames[t]` and produces `frames[t+1]`;
  `dones[t]=1` marks an episode's final frame. Episodes are contiguous.
- `test_episodes/ep_###.npz` — 60 complete held-out episodes (`frames`,
  `actions`) for open-loop evaluation and dream priming.

## Splits

| split | frames | episodes |
|---|---|---|
| train | 420,170 | 1,726 (863 expert / 863 suboptimal) |
| val | 20,174 | 83 |
| test_episodes | — | 60 (each ≥ 100 frames) |

Loader reference: `data.py` in the nano-world-model repo.

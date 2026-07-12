# nano-world-model — Design Spec

**Date:** 2026-07-12
**Status:** Approved design, pre-implementation
**Working name:** `nano-world-model` (final name TBD before publishing)

## One-line pitch

nanoGPT for world models: train a VQ tokenizer + GPT dynamics model on procgen Chaser, then drive the hallucinated game with your keyboard.

## Goals

- Teach action-conditioned world models to **ML-literate engineers** (the nanoGPT audience: fluent in PyTorch and transformers, new to world models).
- Minimal, heavily-commented code (~1,000 core lines) that reads top-to-bottom; the README carries the conceptual narrative.
- End-to-end trainable on a **single Colab GPU (T4)** in a few hours; pretrained checkpoints let readers reach the demo in ~5 minutes.
- **Hero moment:** the reader plays *inside the learned world model* — the model hallucinates each next frame from their keyboard actions.

## Non-goals (v1)

- No latent action model / IDM — v1 uses labeled actions (revisit post-v1).
- No RL: no agent training inside the dream.
- No data collection by readers — dataset is pre-hosted; collection is a documented follow-up.
- Not a research artifact: no benchmark claims, no hyperparameter sweeps.

## Architecture

Two trained components. The dynamics model deliberately mirrors nanoGPT's code structure so the audience recognizes every block.

### Tokenizer: `VQVAE`

- Conv encoder → vector-quantized bottleneck → conv decoder.
- Each 64×64×3 frame → **8×8 grid of discrete tokens**, codebook size ~512.
- Canonical VQ-VAE with **EMA codebook updates** (small amount of extra code that buys reliability against codebook collapse; explained in comments).
- **Known risk:** Chaser's small sprites (orbs, enemies, agent) may blur out at 8× spatial downsampling. Validate reconstruction quality *first*, before any dynamics training. Fallback: 16×16 token grid (4× downsample), which lengthens the GPT context but stays tractable.

### Dynamics model: `WorldModel`

- Decoder-only transformer, block-for-block nanoGPT (CausalSelfAttention, MLP, LayerNorm, learned positional embeddings).
- Operates on interleaved sequences: `[a₀, z₀¹…z₀⁶⁴, a₁, z₁¹…z₁⁶⁴, …]`.
- **Single shared vocab:** 512 frame codes + 15 procgen actions (+ padding/BOS if needed). The GPT's token embeddings are its own learned table — codebook vectors live only in the tokenizer; the GPT deals purely in indices.
- Scale: ~10–20M params. Context: horizon 8–16 frames × 65 tokens/frame ≈ 520–1,040 tokens.
- Loss: cross-entropy on next-token prediction, exactly as in language modeling — computed **only on frame-token positions**. Action positions are inputs, not targets (the world model predicts the world, not the player's mind); the mask is one line and gets a comment explaining exactly this.

### Sampling

- Generating a frame = **64 autoregressive categorical samples** (raster order, KV-cached), one 512-way softmax over codebook indices per step. `generate()` is line-for-line nanoGPT's.
- Temperature / top-k exposed as knobs; their effect on rollout stability is a documented teaching point.
- Budget: ~1–3 ms/token on a T4 → 60–200 ms/frame → 5–15 fps. Playable.

### The conceptual spine (README FAQ)

The README includes a FAQ section walking the exact objection sequence a sharp reader raises:

1. **Why discrete tokens at all?** Direct vector/pixel regression under L2 predicts the *mean over futures* — ghost blurs (the 2016 video-prediction failure mode). A categorical softmax holds multiple sharp modes and the sampler *commits* to one. The VQ tokenizer exists to make frames language-shaped so the entire LLM toolkit (cross-entropy, sampling, temperature, top-k) applies. Anchors: Ha & Schmidhuber's MDN head (continuous-space fix), DIAMOND's diffusion (the other modern fix).
2. **Why sample instead of argmax?** The environment has stochastic elements; sampling commits to one plausible future instead of the most-likely-token-per-position collage.
3. **Why autoregressive within a frame — why not one shot?** One-shot multi-head output samples the *product of marginals*, not the joint — attention (masked or not) shares *beliefs* between positions, never sampled *outcomes*. Coordinated stochastic events (an enemy picking a corridor) require realized samples to feed back into subsequent conditioning. AR is the exact chain-rule factorization; MaskGIT-style iterative decoding (Genie) is the amortized approximation; a shared latent (CVAE-style) is the third option.
4. **Why 64 tokens per frame instead of 1?** Exponential tradeoff between sequence length and vocab size: 64 tokens × 512 codes expresses 512⁶⁴ frames with 64×512 learned quantities. One token per frame requires a codebook enumerating all distinguishable game states — combinatorially impossible for procgen. Images are compositional; spatial factorization exploits that, the same way LLMs use ~50k word pieces instead of a vocabulary of sentences.

Reader exercise: swap `generate()` for one-shot parallel prediction and watch where the world breaks.

## Environment: procgen Chaser

Why Chaser:

- **Static camera, fully observable** — the whole maze fits in the 64×64 frame. Transitions are "copy previous frame, move a few sprites," which makes dynamics *inspectable* (per-token-position loss heatmaps show exactly where the model works hard). Scrolling-camera games (CoinRun, Crafter) shift every pixel on every action.
- Dynamic entities (chasing enemies, disappearing orbs, vulnerability power-ups) force the model to learn rules, not just sprite translation.
- Small effective action space (9 movement combos of procgen's 15 actions).
- Procgen lineage connects to Genie et al.

**Constraint:** the `procgen` pip package is unmaintained (wheels ≤ Python 3.10, x86 only) and does not install on modern Colab. Mitigation: readers never need procgen — data and held-out eval episodes are pre-hosted, and the play demo needs no environment at all. Collection/eval-against-env scripts run on a Py3.10/x86 box (Kevin's servers).

## Data

- **Source:** Meta's gen_dgrl offline procgen dataset (Mediratta et al., ICLR 2024; `facebookresearch/gen_dgrl`) — ~1M Chaser transitions from PPO agents, 64×64×3 uint8 observations. Expert + suboptimal variants.
- **Curation:** subsample to ~300–500k training frames + a held-out set of *complete episodes* (frames + actions + done flags) for open-loop eval and dream-priming.
- **Re-host on HuggingFace** in a dead-simple format: uint8 frame shards + action arrays + episode boundaries as `.npz` shards, loadable by ~50 lines in `data.py`. Attribution and license (CC-BY-NC per gen_dgrl) documented in the HF dataset card and README.
- `data.py download` is all a reader ever runs. The gen_dgrl curation script ships in `scripts/` for transparency.
- **Verification gate (first implementation task):** confirm gen_dgrl downloads are live, inspect Chaser action coverage and episode quality. Fallback if inadequate: collect our own via PPO checkpoints (0/25/50/100% training) + ε-noise + random episodes — design already sketched, becomes primary only if needed.

## Repo layout

```
README.md            ← the conceptual narrative + FAQ
model.py             ← VQVAE + WorldModel (~500 lines, heavily commented)
data.py              ← HF download + dataset/dataloader
train_tokenizer.py   ← stage 1 (~30–60 min on T4)
train_wm.py          ← stage 2 (~1–3 hr on T4)
eval.py              ← recon grids, rollout GIFs, drift curves, loss heatmaps
play.py              ← pygame: real-time keyboard play (CPU/MPS-friendly)
walkthrough.ipynb    ← thin Colab: install → download → train (or load ckpts) → play
scripts/             ← gen_dgrl curation; (later) own data collection
```

- Configs: argparse + dataclass defaults inline, nanoGPT style. No Hydra, no Lightning.
- Dependencies: torch, numpy, huggingface_hub, pillow + matplotlib (eval/visualization only), pygame (play only), ipywidgets (notebook only).
- Total core code target: ~1,000 lines.

## Hero moment, de-risked

- **Pretrained tokenizer + WM checkpoints hosted on HF.** The Colab notebook reaches "playing inside the dream" in ~5 minutes, before any training. (Same move as nanoGPT shipping GPT-2 weights.)
- Two play interfaces:
  - `play.py` — pygame window, real-time arrow keys, local machine (model is small enough for CPU/MPS inference).
  - Notebook — button-stepped ipywidgets UI (Colab can't capture keystrokes reliably).
- Dream is primed with a few real frames from a held-out episode, then runs on pure hallucination conditioned on user actions.

## Evaluation

**Pedagogical artifacts (in `eval.py`, screenshotted into README):**

- Tokenizer: original-vs-reconstruction grids; codebook usage histogram.
- Dynamics: teacher-forced next-token accuracy and loss curves; open-loop real-vs-dream side-by-sides from identical prefix + action sequence (works offline via shipped held-out episodes); drift-over-horizon curves; **per-token-position loss heatmap** (shows "most tokens copy; the action-relevant patches are the hard ones").

**Milestone review gate (tokenizer → WM):** after tokenizer training, work **pauses** for Kevin's qualitative review before any world-model training begins. Deliverable: render panels including (a) original-vs-reconstruction grids over diverse held-out frames, deliberately including hard cases (small orbs, enemies mid-chase, vulnerability power-up states, dense mazes); (b) codebook usage histogram; (c) per-pixel reconstruction error heatmaps; (d) if capacity is borderline, an 8×8-vs-16×16 token grid side-by-side comparison. WM training starts only on explicit sign-off.

**Engineering sanity:**

- Overfit-a-single-batch mode in both train scripts.
- Tiny-config smoke run (minutes on CPU) as a de-facto test.
- Full pipeline validated on Kevin's GPU servers before data/checkpoints are published.

**Documented failure modes as teaching moments:** compounding error in open-loop rollouts, temperature effects, tokenizer capacity limits.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Tokenizer loses small sprites at 8×8 tokens | Validate recon first; fall back to 16×16 grid |
| gen_dgrl links dead or coverage poor | Verification gate up front; fallback to own PPO-checkpoint collection |
| Open-loop drift makes play demo mushy | Enough data + temperature/top-k tuning; prime with real frames; document as teaching point |
| Colab interactivity jank | Button-stepped widget as baseline; pygame is the real-time path |
| gen_dgrl license (CC-BY-NC) | NC is fine for an educational repo; attribute clearly on HF card + README |

## Compute targets

- Tokenizer: ≤ 1 hr on Colab T4.
- World model: 1–3 hr on Colab T4 (A100 ~4× faster).
- Development/validation runs on Kevin's GPU servers; published checkpoints come from there.

## Future extensions (explicitly deferred)

1. **Latent action model (Genie-style LAM/IDM)** — learn actions unsupervised from video; swap labeled actions for latent ones, same GPT. Revisit after v1.
2. **MaskGIT-style parallel decoding** — the bridge from IRIS-style AR to Genie; natural "chapter 2" alongside the LAM.
3. **Diffusion head** — replace the categorical head with a diffusion decoder over frame latents (DIAMOND/GameNGen lineage); the continuous-space answer to mode averaging.
4. **Own data collection + comparison** — PPO-checkpoint + ε-noise pipeline vs. gen_dgrl; measure effect of state–action coverage on world-model quality.

## Resolved design decisions (from brainstorming)

- Audience: ML-literate engineers (nanoGPT audience) — not researchers, not beginners.
- Hero moment: interactive play-in-the-dream (over passive rollout videos or RL-in-dream).
- Architecture: VQ tokenizer + AR GPT (over diffusion WM and RSSM/Dreamer) — maximal nanoGPT correspondence.
- Environment: procgen Chaser (over Atari, Crafter, CoinRun) — static camera, full observability, procgen lineage.
- Data: gen_dgrl reuse (over collecting fresh) — readers focus on the model, not the data.
- Actions: ground-truth labels in v1 (LAM deferred).
- Form factor: nanoGPT-style repo + thin Colab notebook (over notebook-first or dual-track).
- Within-frame generation: full AR over 64 tokens (over one-shot parallel and MaskGIT) — exact joint, exact nanoGPT reuse, speed is a non-issue at this scale.

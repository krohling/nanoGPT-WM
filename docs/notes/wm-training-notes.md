# World-model training run 1 — notes (2026-07-12)

## Setup

- Host: larg-aaronson, 1× A100 (CUDA_VISIBLE_DEVICES=0), tokenizer `tok8`
  (8×8 grid, val MSE 0.00034, approved by Kevin at the review gate).
- Data: kevin510/nano-world-model-chaser (420k train / 20k val frames),
  tokens pre-computed to uint16 memmaps (~2 min).
- Model: GPTConfig defaults — 8 layers, 6 heads, 384 dim, block 1039
  (16 frames × 64 tokens + 15 action tokens), ~14M params.
- Run: 40k steps, bs 24, lr 3e-4 cosine, ~27.5 it/s → ~25 min wall clock.

## Results

- Val loss (CE, frame-token positions only): best **0.2230** @ step 35k,
  final 0.2404 (mild late-schedule overfit).
  - Known flaw in run 1: the save logic overwrote the best checkpoint with the
    final one (fixed in train_wm.py right after — `world_model.pt` is now
    always best-val, `world_model_last.pt` the latest). Run-1 artifacts use
    the step-40k weights; delta is small. Retrain with the fixed script
    before publishing checkpoints.
- Open-loop rollouts (60 steps, temp 1.0, eps 0/1/2 + ep 0 @ 0.5): maze stays
  pixel-solid for the full horizon; sprite positions diverge smoothly from the
  real episode (expected under stochastic sampling). No melting, no wall drift.
- Drift curve (20 episodes, 40-step horizon): smooth ~linear growth,
  0 → ~180 uint8-MSE (≈5% RMSE), no explosion. Canonical compounding error.
- Loss heatmap: most token positions ≈0.05 CE ("copy the maze"); hot spots up
  to ~0.75 where agent/enemies move. The "most tokens copy, action-relevant
  patches are hard" teaching artifact works as designed.
- **Directional action test** (dream 10 steps of held RIGHT vs held LEFT from
  the same 4-frame prime, agent tracked by color centroid):
  ep0 +29.0 / −17.7 px, ep1 +12.0 / −20.2 px, ep2 +8.9 / −23.0 px.
  The dreamed agent obeys the pad in all 3 mazes.

## Quality bar (plan Task 10 step 4)

- Maze-coherent ≥ 20 open-loop steps: PASS (60).
- Agent responds to actions: PASS (directional test).
- Drift grows smoothly: PASS.

## Follow-ups

- Retrain with fixed best-ckpt logic before HF checkpoint publish (~25 min).
- Consider step-35k-equivalent early stop or slight dropout for the publish run.

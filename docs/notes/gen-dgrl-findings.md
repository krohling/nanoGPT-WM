# gen_dgrl Chaser data — verification gate findings (2026-07-12)

**Decision: GO.** All criteria pass. Details below.

## Download

- URLs (live, verified by HEAD + full download):
  - `https://dl.fbaipublicfiles.com/DGRL/1M/expert/chaser.tar.xz` (178,687,792 B)
  - `https://dl.fbaipublicfiles.com/DGRL/1M/suboptimal/chaser.tar.xz` (177,259,832 B)
- Archives are per-game — no need to download the full 16-game release.
- **Quota lesson (the hard way):** fully extracted, the two archives balloon to
  ~48 GB (frames stored as float-ish arrays in pickled dicts; xz crushes them
  ~135:1). Extraction nearly blew the 200 GB shared LARG quota (hit 97%).
  Extractions were killed, raw dirs deleted; all downstream processing now
  **streams directly from the .tar.xz archives** (`tarfile` mode `r|xz`) and
  never materializes raw data on disk.

## File format (decoded from gen_dgrl `offline/dataloader.py` + verified empirically)

- Each archive: `chaser/<timestamp>_<counter>_<length>_<levelseed>_<return>.npy`,
  ~1M transitions per variant across ~4k episodes each.
- Each `.npy` is a **pickled dict** (`np.load(..., allow_pickle=True).item()`),
  NOT an npz: keys `observations` [T,3,64,64] (CHW, 0–255 scale), `actions`
  [T,1], `rewards` [T,1], `dones` [T,1].
- Convention confirmed from their `OfflineDataset.__getitem__`: standard SARS —
  `actions[t]` is taken at `observations[t]` and produces `observations[t+1]`.
  Matches our `meta.json` convention exactly.

## Inspection stats (600 episodes, 300 per variant, streamed)

- 147,438 frames; episode length min/med/max = 2/251/647 (curation filters len < 20).
- Extrapolated total: ~2M transitions available; we need ~440k. ✓
- Returns: expert mean 6.83 (0.00–13.32), suboptimal mean 6.91 (0.00–13.32) —
  the two variants are more similar in return than the paper's 75% framing
  suggests, but both span the full range, so skill coverage is fine.
- Action histogram: all 15 actions present, 2.60%–9.65% each (procgen's
  special-key actions 9–14 appear too; they are no-ops in Chaser — useful
  no-op coverage). No action > 50%; ≥ 8 actions ≥ 1%. ✓
- Contact sheet (`contact_sheet.png`): full-maze static-camera views, legible
  agent/enemy/orb/star sprites, clearly varied procgen layouts. ✓
  Sprites are ~5 px — reinforces the planned tokenizer review gate (8×8 token
  grid = 8 px patches; watch small-sprite reconstruction quality).

## Caveats

- Curation takes episodes in archive (timestamp) order, alternating variants,
  rather than a global random shuffle — streaming constraint. Returns show no
  obvious ordering trend in the sampled prefix; acceptable for v1.
- License: CC-BY-NC 4.0 (attribution in HF dataset card + README).

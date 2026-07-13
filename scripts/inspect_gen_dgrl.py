"""Inspect raw gen_dgrl chaser data and emit verification-gate stats.

Usage: python scripts/inspect_gen_dgrl.py RAW_DIR OUT_DIR [LIMIT]
  RAW_DIR contains chaser_expert.tar.xz / chaser_suboptimal.tar.xz

Episodes are STREAMED straight out of the .tar.xz archives — never extracted
to disk. (Fully extracted, the two archives are ~48 GB of float32 frames;
converted to uint8 bins they are ~12x smaller. The shared filer quota cannot
afford the intermediate copy.)

gen_dgrl episode file format (decoded from their offline/dataloader.py):
  np.load -> npz-like with keys observations/actions/rewards/dones;
  observations [T, 3, 64, 64] (CHW), values 0..255; actions [T, 1].
  Filename stem: <timestamp>_<counter>_<length>_<levelseed>_<return>.
"""
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import save_grid  # noqa: E402


def load_episodes(raw_dir, variants=("expert", "suboptimal"), limit=None):
    """Yield dicts: frames [T,64,64,3] u8, actions [T] u8, variant, ret.

    Streams the archives member-by-member, alternating between variants so
    downstream consumers see an interleaved skill mix.
    """
    streams = []
    for v in variants:
        p = Path(raw_dir) / f"chaser_{v}.tar.xz"
        if p.exists():
            tf = tarfile.open(p, mode="r|xz")   # streaming: sequential access only
            streams.append([v, tf, iter(tf)])
    n = 0
    active = streams
    while active:
        keep = []
        for s in active:
            v, tf, it = s
            member = next(it, None)
            while member is not None and not (member.isfile()
                                              and member.name.endswith(".npy")):
                member = next(it, None)
            if member is None:
                tf.close()
                continue
            # files are np.save'd python dicts (pickle), not npz archives;
            # trusted source (Meta's official release), so allow_pickle is OK
            raw_obj = np.load(io.BytesIO(tf.extractfile(member).read()),
                              allow_pickle=True)
            ep = raw_obj.item() if raw_obj.ndim == 0 else dict(raw_obj)
            obs = ep["observations"]
            act = ep["actions"]
            if obs.ndim != 4 or len(obs) < 2:
                keep.append(s)
                continue
            if obs.shape[1] == 3:                       # CHW -> HWC
                obs = obs.transpose(0, 2, 3, 1)
            assert obs.max() > 1.5, "expected 0..255 pixel scale, got <=1"
            frames = np.ascontiguousarray(obs).astype(np.uint8)
            actions = act.reshape(len(act)).astype(np.uint8)
            t = min(len(frames), len(actions))
            yield {"frames": frames[:t], "actions": actions[:t], "variant": v,
                   "ret": float(Path(member.name).stem.split("_")[-1])}
            n += 1
            if limit is not None and n >= limit:
                for _, tf2, _ in active:
                    tf2.close()
                return
            keep.append(s)
        active = keep


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    out.mkdir(parents=True, exist_ok=True)
    lens, act, rets = [], Counter(), {"expert": [], "suboptimal": []}
    sheet = []
    dtype_seen = None
    for i, ep in enumerate(load_episodes(raw, limit=limit)):
        f, a = ep["frames"], ep["actions"]
        assert f.dtype == np.uint8 and f.shape[1:] == (64, 64, 3), f.shape
        assert a.max() < 15, a.max()
        lens.append(len(f))
        act.update(a.tolist())
        rets[ep["variant"]].append(ep["ret"])
        if i % 9 == 0 and len(sheet) < 64:
            sheet.append(f[len(f) // 2])
    print(f"episodes inspected: {len(lens)}")
    print(f"frames: {sum(lens)}  len min/med/max: "
          f"{min(lens)}/{int(np.median(lens))}/{max(lens)}")
    for v, r in rets.items():
        if r:
            print(f"{v}: {len(r)} eps, return mean {np.mean(r):.2f} "
                  f"min {min(r):.2f} max {max(r):.2f}")
    tot = sum(act.values())
    print("action histogram:")
    for k in sorted(act):
        print(f"  action {k:2d}: {act[k] / tot:7.2%}")
    save_grid(np.stack(sheet), out / "contact_sheet.png", ncol=8)
    (out / "stats.json").write_text(json.dumps({
        "episodes": len(lens), "frames": sum(lens),
        "len_min": int(min(lens)), "len_med": int(np.median(lens)),
        "len_max": int(max(lens)),
        "action_hist": {str(k): act[k] / tot for k in sorted(act)},
        "returns": {v: [float(np.mean(r)), float(min(r)), float(max(r))]
                    for v, r in rets.items() if r},
    }, indent=1))
    print(f"wrote {out}/contact_sheet.png and stats.json")

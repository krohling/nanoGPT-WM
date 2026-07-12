"""Play inside the dream: the world model hallucinates each next frame from
your keystrokes. Arrow keys move (procgen action combos), R resets to a fresh
real-frame priming, -/+ adjust sampling temperature, ESC quits.

There is no game engine here. Every frame you see is sampled, token by token,
from the GPT — the maze, the enemies, and the consequences of your actions
exist only in the model's weights.
"""
import argparse
import time

import numpy as np
import pygame
import torch

import data as data_mod
from eval import _crop_to_frames, load_models, to_uint8
from model import FRAME_VOCAB
from train_wm import interleave

# procgen's 15 discrete actions are (LEFT/RIGHT) x (DOWN/UP) combos + special
# keys. index: 0 (L,D) 1 (L) 2 (L,U) 3 (D) 4 noop 5 (U) 6 (R,D) 7 (R) 8 (R,U)
# 9-14 specials (D,A,W,S,Q,E) — unused by Chaser.
KEY_TO_ACTION = {(-1, -1): 0, (-1, 0): 1, (-1, 1): 2, (0, -1): 3, (0, 0): 4,
                 (0, 1): 5, (1, -1): 6, (1, 0): 7, (1, 1): 8}


def action_from_keys(keys):
    lr = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])   # -1, 0, 1
    ud = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
    return KEY_TO_ACTION[(lr, ud)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--prime", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    tok, wm = load_models(a.tokenizer, a.wm, a.device)
    K = tok.tokens_per_frame
    eps = data_mod.list_episodes(a.data)
    rng = np.random.default_rng()

    def reset():
        """Prime the dream with a few real frames from a held-out episode."""
        ep = data_mod.load_episode(eps[rng.integers(len(eps))])
        x = torch.from_numpy(ep["frames"][:a.prime].astype(np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2).to(a.device)
        toks = tok.encode(x).cpu().numpy().astype(np.int64)
        seq = interleave(toks, np.asarray(ep["actions"][:a.prime])).tolist()
        return seq, ep["frames"][a.prime - 1]

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    seq, frame = reset()
    temp, running = a.temperature, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    seq, frame = reset()
                elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    temp = max(0.1, round(temp - 0.1, 1))
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    temp = min(2.0, round(temp + 0.1, 1))
        act = action_from_keys(pygame.key.get_pressed())
        t0 = time.time()
        seq = _crop_to_frames(seq, K, a.seq_len - 1)
        ctx = torch.tensor(seq, device=a.device)[None]
        with torch.no_grad():
            nxt = wm.generate_frame(ctx, act, tokens_per_frame=K, temperature=temp)
        seq += [FRAME_VOCAB + act] + nxt[0].tolist()
        frame = to_uint8(tok.decode(nxt))[0]
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(pygame.transform.scale(surf, (512, 512)), (0, 0))
        pygame.display.flip()
        pygame.display.set_caption(
            f"nano-world-model — {1000 * (time.time() - t0):.0f} ms/frame  "
            f"temp {temp:.1f}  (arrows move, R reset, -/+ temp, ESC quit)")
    pygame.quit()


if __name__ == "__main__":
    main()

import numpy as np
import pytest
import torch

from model import FRAME_VOCAB, VOCAB_SIZE, GPTConfig, KVSampler, WorldModel


def tiny_cfg(**kw):
    d = dict(vocab_size=VOCAB_SIZE, block_size=5 * 5, n_layer=2, n_head=2,
             n_embd=32, dropout=0.0, bias=False)
    d.update(kw)
    return GPTConfig(**d)


def full_logits(m, idx):
    """nanoGPT returns last-position logits when targets=None; pass dummy
    targets to get every position (returned loss is ignored)."""
    logits, _ = m(idx, torch.zeros_like(idx))
    return logits


def test_forward_shapes_and_loss():
    m = WorldModel(tiny_cfg())
    idx = torch.randint(0, VOCAB_SIZE, (2, 20))
    tgt = idx.roll(-1, dims=1).clone()
    tgt[tgt >= FRAME_VOCAB] = -1
    logits, loss = m(idx, tgt)
    assert logits.shape == (2, 20, VOCAB_SIZE)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_forward_without_targets_returns_last_position_only():
    # vendored nanoGPT semantics — documented, so pinned by a test
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, VOCAB_SIZE, (2, 20))
    logits, loss = m(idx)
    assert logits.shape == (2, 1, VOCAB_SIZE) and loss is None


def test_causality():
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, VOCAB_SIZE, (1, 20))
    with torch.no_grad():
        l1 = full_logits(m, idx)
        idx2 = idx.clone()
        idx2[0, 10] = (idx2[0, 10] + 1) % VOCAB_SIZE
        l2 = full_logits(m, idx2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5)
    assert not torch.allclose(l1[0, 10:], l2[0, 10:], atol=1e-5)


def test_loss_ignores_action_positions():
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, FRAME_VOCAB, (1, 10))
    tgt = torch.full((1, 10), -1)
    tgt[0, 3] = 7
    logits, loss = m(idx, tgt)
    per_tok = torch.nn.functional.cross_entropy(
        logits[0], tgt[0].clamp(min=0), reduction="none")
    assert torch.isclose(loss, per_tok[3], atol=1e-5)


def test_kv_sampler_matches_full_forward():
    """The cached fast path must produce the same logits as verbatim nanoGPT,
    including a multi-token prefill on top of an existing cache (the case
    where is_causal=True would silently be wrong)."""
    m = WorldModel(tiny_cfg()).eval()
    idx = torch.randint(0, VOCAB_SIZE, (1, 12))
    with torch.no_grad():
        full = full_logits(m, idx)
        s = KVSampler(m.gpt)
        l1 = s.forward(idx[:, :8])
        l2 = s.forward(idx[:, 8:])
    assert torch.allclose(full[:, :8], l1, atol=1e-4)
    assert torch.allclose(full[:, 8:], l2, atol=1e-4)


def test_generate_frame_range():
    m = WorldModel(tiny_cfg(block_size=200)).eval()
    ctx = torch.randint(0, FRAME_VOCAB, (1, 30))
    out = m.generate_frame(ctx, action=4, tokens_per_frame=16)
    assert out.shape == (1, 16)
    assert out.max() < FRAME_VOCAB  # never samples an action token


def test_fast_and_slow_sampling_identical():
    """Same seed -> KVSampler and the nanoGPT-style slow loop must sample the
    exact same frame (they compute the same distributions in the same order)."""
    m = WorldModel(tiny_cfg(block_size=200)).eval()
    ctx = torch.randint(0, FRAME_VOCAB, (1, 30))
    torch.manual_seed(7)
    slow = m.generate_frame(ctx, action=4, tokens_per_frame=16)
    torch.manual_seed(7)
    fast = KVSampler(m.gpt).generate_frame(ctx, action=4, tokens_per_frame=16)
    assert torch.equal(slow, fast)


def test_interleave_and_targets():
    from train_wm import interleave, make_targets
    tokens = np.arange(8, dtype=np.int64).reshape(2, 4)     # 2 frames, 4 tokens
    actions = np.array([3, 7])
    seq = interleave(tokens, actions)
    # z0 a0 z1 — the trailing action (7) is dropped: its consequence (frame 2)
    # lies outside the window
    assert seq.tolist() == [0, 1, 2, 3, 512 + 3, 4, 5, 6, 7]
    tgt = make_targets(seq)
    # next-token targets, with positions whose TARGET is an action masked out
    assert tgt.tolist() == [1, 2, 3, -1, 4, 5, 6, 7]
    assert len(tgt) == len(seq) - 1


@pytest.mark.slow
def test_action_conditioning_on_toy_world():
    """A 4x4 gridworld: frame = 16 tokens (agent cell=1, rest=0). Actions
    1/7/5/3 move the agent left/right/up/down (procgen movement subset).
    A tiny WM must learn action-conditioned dynamics almost perfectly."""
    from train_wm import interleave, make_targets
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    MOVES = {1: (0, -1), 7: (0, 1), 5: (-1, 0), 3: (1, 0)}

    def episode(T=9):
        r, c = rng.integers(0, 4, 2)
        toks, acts = [], []
        for _ in range(T):
            f = np.zeros(16, np.int64)
            f[r * 4 + c] = 1
            a = int(rng.choice(list(MOVES)))
            toks.append(f)
            acts.append(a)
            dr, dc = MOVES[a]
            r, c = np.clip(r + dr, 0, 3), np.clip(c + dc, 0, 3)
        return np.stack(toks), np.array(acts)

    cfg = tiny_cfg(block_size=9 * 17, n_embd=64)
    m = WorldModel(cfg)
    # nanoGPT's weight tying makes the copy circuit form later on tiny tasks —
    # this needs a higher lr and more steps than an untied toy transformer would
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for step in range(3000):
        seqs = [interleave(*episode()) for _ in range(16)]
        x = torch.tensor(np.stack([s[:-1] for s in seqs]))
        y = torch.tensor(np.stack([make_targets(s) for s in seqs]))
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # rollout: prime one frame, drive 5 chosen actions, check the agent obeys
    m.eval()
    toks, _ = episode(1)
    ctx = torch.tensor(toks[0])[None]   # a bare frame — no priming action needed
    r, c, ok = *(divmod(int(toks[0].argmax()), 4)), 0
    for a in [7, 7, 3, 1, 5]:
        out = m.generate_frame(ctx, action=a, tokens_per_frame=16, temperature=0.01)
        dr, dc = MOVES[a]
        r, c = int(np.clip(r + dr, 0, 3)), int(np.clip(c + dc, 0, 3))
        ok += int(out[0].argmax().item() == r * 4 + c)
        ctx = torch.cat([ctx, torch.tensor([[512 + a]]), out], dim=1)
    assert ok >= 4, f"only {ok}/5 moves obeyed the action"

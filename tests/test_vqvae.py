import pytest
import torch

from model import VQVAE, FRAME_VOCAB, NUM_ACTIONS, VOCAB_SIZE


def test_constants():
    assert FRAME_VOCAB == 512 and NUM_ACTIONS == 15 and VOCAB_SIZE == 527


@pytest.mark.parametrize("grid", [8, 16])
def test_shapes(grid):
    m = VQVAE(codebook_size=512, emb_dim=64, grid=grid)
    x = torch.rand(2, 3, 64, 64)
    recon, commit, idx = m(x)
    assert recon.shape == x.shape
    assert idx.shape == (2, grid * grid) and idx.dtype == torch.long
    assert idx.max() < 512
    assert commit.dim() == 0
    assert m.tokens_per_frame == grid * grid


def test_encode_decode():
    m = VQVAE()
    x = torch.rand(2, 3, 64, 64)
    idx = m.encode(x)
    out = m.decode(idx)
    assert out.shape == (2, 3, 64, 64)
    assert 0.0 <= out.min() and out.max() <= 1.0  # decode clamps for display


def test_gradients_flow_through_encoder():
    m = VQVAE()
    x = torch.rand(2, 3, 64, 64)
    recon, commit, _ = m(x)
    loss = torch.nn.functional.mse_loss(recon, x) + 0.25 * commit
    loss.backward()
    g = next(m.encoder.parameters()).grad
    assert g is not None and g.abs().sum() > 0


@pytest.mark.slow
def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = VQVAE()
    # structured frames (colored squares), not noise — must become reconstructable
    x = torch.zeros(4, 3, 64, 64)
    for i in range(4):
        x[i, i % 3, 8 * i : 8 * i + 24, 8 * i : 8 * i + 24] = 1.0
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(300):
        recon, commit, _ = m(x)
        loss = torch.nn.functional.mse_loss(recon, x) + 0.25 * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.nn.functional.mse_loss(m(x)[0], x).item() < 0.01

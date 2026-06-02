import os
import sys

import pytest
import torch

# Allow running from the repo without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_r1.models.vib_adapter import VIBAdapter, VIBConfig, update_dual_beta


D = 32
B, N, L = 2, 5, 7


def make_adapter(**kw):
    torch.manual_seed(0)
    return VIBAdapter(d_model=D, hidden_dim=16, cond_dim=D, n_cross_heads=4, **kw)


def make_batch(dtype=torch.float32):
    torch.manual_seed(1)
    H = torch.randn(B, N, D, dtype=dtype)
    text = torch.randn(B, L, D, dtype=dtype)
    mask = torch.ones(B, L, dtype=torch.bool)
    return H, text, mask


def test_kl_non_negative():
    adapter = make_adapter(free_bits=0.0)
    H, text, mask = make_batch()
    # Drive the heads away from the prior so raw KL is non-trivial.
    with torch.no_grad():
        adapter.mu_head[-1].bias.normal_(0, 1.0)
        adapter.logvar_head[-1].bias.normal_(0, 1.0)
    _, rate, stats = adapter(H, text, mask, sample=True)
    assert torch.isfinite(rate)
    assert (stats["kl_per_token"] >= -1e-5).all()
    assert rate.item() >= -1e-5


def test_kl_zero_at_prior():
    # Near-prior init (mu=0, logvar=0) => raw per-token KL ~ 0 (within free_bits).
    adapter = make_adapter(free_bits=0.0)
    H, text, mask = make_batch()
    _, rate, stats = adapter(H, text, mask, sample=False)
    assert stats["kl_token_mean"].item() == pytest.approx(0.0, abs=1e-5)
    assert rate.item() == pytest.approx(0.0, abs=1e-5)


def test_initial_rate_equals_free_bits():
    fb = 0.5
    adapter = make_adapter(free_bits=fb)
    H, text, mask = make_batch()
    _, rate, _ = adapter(H, text, mask, sample=True)
    assert rate.item() == pytest.approx(fb, abs=1e-5)


def test_gradients_flow_backbone_frozen():
    adapter = make_adapter(free_bits=0.0)
    # Frozen "backbone" producing H.
    backbone = torch.nn.Linear(D, D)
    for p in backbone.parameters():
        p.requires_grad = False
    x = torch.randn(B, N, D)
    H = backbone(x)
    text, mask = torch.randn(B, L, D), torch.ones(B, L, dtype=torch.bool)
    with torch.no_grad():
        adapter.logvar_head[-1].bias.fill_(0.5)
    Z, rate, _ = adapter(H, text, mask, sample=True)
    (Z.sum() + rate).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in backbone.parameters())


def test_higher_beta_lowers_rate():
    H, text, mask = make_batch()
    target = torch.randn(B, N, D)

    def train_rate(beta, steps=200):
        adapter = make_adapter(free_bits=0.0)
        readout = torch.nn.Linear(D, D)
        opt = torch.optim.Adam(list(adapter.parameters()) + list(readout.parameters()), lr=5e-3)
        rate = None
        for _ in range(steps):
            opt.zero_grad()
            Z, rate, _ = adapter(H, text, mask, sample=True)
            task = ((readout(Z) - target) ** 2).mean()
            (task + beta * rate).backward()
            opt.step()
        return rate.item()

    rates = [train_rate(b) for b in (0.0, 0.1, 1.0)]
    assert rates[0] >= rates[1] - 1e-3
    assert rates[1] >= rates[2] - 1e-3


def test_shapes_and_bf16_no_nan():
    adapter = make_adapter(free_bits=0.5)
    H, text, mask = make_batch(dtype=torch.bfloat16)
    adapter = adapter.to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        Z, rate, stats = adapter(H, text, mask, sample=True)
    assert Z.shape == H.shape
    assert Z.dtype == torch.bfloat16
    # KL must be computed in fp32 regardless of compute dtype.
    assert rate.dtype == torch.float32
    assert torch.isfinite(rate)
    assert torch.isfinite(stats["kl_per_token"].float()).all()


def test_instruction_agnostic_independent_of_text():
    adapter = make_adapter(free_bits=0.0, instruction_conditioned=False)
    with torch.no_grad():
        adapter.logvar_head[-1].bias.fill_(0.3)
        adapter.mu_head[-1].weight.normal_(0, 0.1)
    H = torch.randn(B, N, D)
    text_a = torch.randn(B, L, D)
    text_b = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    Za, _, _ = adapter(H, text_a, mask, sample=False)
    Zb, _, _ = adapter(H, text_b, mask, sample=False)
    assert torch.allclose(Za, Zb)
    # Also runs with no text at all.
    Zc, _, _ = adapter(H, None, None, sample=False)
    assert torch.allclose(Za, Zc)


def test_residual_mode_preserves_signal_at_init():
    adapter = make_adapter(free_bits=0.0, residual=True)
    H, text, mask = make_batch()
    Z, _, _ = adapter(H, text, mask, sample=False)
    # mu ~ 0 at init, residual => Z ~ H.
    assert torch.allclose(Z, H, atol=1e-5)


def test_dual_beta_update():
    beta = 0.1
    # rate above target -> beta increases; below -> decreases; clamped to [0, max].
    up = update_dual_beta(beta, torch.tensor(2.0), r_star=1.0, lr_beta=0.5, beta_max=10.0)
    down = update_dual_beta(beta, torch.tensor(0.0), r_star=1.0, lr_beta=0.5, beta_max=10.0)
    assert up > beta
    assert down < beta
    assert down >= 0.0
    capped = update_dual_beta(5.0, torch.tensor(100.0), r_star=1.0, lr_beta=1.0, beta_max=6.0)
    assert capped == 6.0


def test_apply_vib_per_sample_and_pop_rate():
    import types

    from open_r1.models.qwen_vib_patch import _apply_vib, pop_vib_rate

    IMG = 99
    adapter = make_adapter(free_bits=0.5)
    model = types.SimpleNamespace()
    model.vib = adapter
    model.config = types.SimpleNamespace(image_token_id=IMG)
    model.training = True
    model.vib_eval_mode = "mean"
    model.shared_noise_per_prompt = False
    model._vib_group_seed = None

    # Two samples with 2 and 3 image tokens respectively; rest is text.
    input_ids = torch.tensor([[IMG, IMG, 1, 2, 0], [IMG, IMG, IMG, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    n_img = int((input_ids == IMG).sum())
    image_embeds = torch.randn(n_img, D, requires_grad=True)
    inputs_embeds = torch.randn(2, 5, D)

    new_embeds = _apply_vib(model, image_embeds, input_ids, inputs_embeds, attention_mask)
    assert new_embeds.shape == image_embeds.shape

    rate = pop_vib_rate(model)
    assert rate is not None and rate.requires_grad
    # popping again clears it
    assert pop_vib_rate(model) is None

    rate2 = _apply_vib(model, image_embeds, input_ids, inputs_embeds, attention_mask)
    model._last_vib_rate.sum().backward()
    assert any(p.grad is not None for p in adapter.parameters())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

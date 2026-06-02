"""Visual Information Bottleneck (VIB) adapter for aerial VLN policies.

Learns an instruction-conditioned, per-token Gaussian bottleneck over the
post-merger visual tokens fed to the LLM. The realized information rate
(mean per-token KL to a unit-Gaussian prior) becomes a controllable axis,
decoupling "information rate" from "input resolution".
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VIBConfig:
    """Configuration for the visual information bottleneck.

    Mirrors the ``vib:`` block documented in the README. Kept as a plain
    dataclass so it can be built from flat CLI flags or a yaml config.
    """

    enabled: bool = False
    mode: str = "fixed"  # "fixed" | "dual"
    beta: float = 1e-3
    R_star: Optional[float] = None  # target nats/token for dual mode
    lr_beta: float = 0.01
    beta_max: float = 10.0
    free_bits: float = 0.5
    instruction_conditioned: bool = True
    eval_mode: str = "mean"  # "mean" | "sample" | "sample_avg_k"
    eval_samples: int = 4
    n_cross_heads: int = 8
    hidden_dim: Optional[int] = None  # defaults to d_model
    cond_dim: Optional[int] = None  # defaults to d_model
    logvar_min: float = -8.0
    logvar_max: float = 2.0
    residual: bool = False
    suppress_eps: float = 1e-2
    shared_noise_per_prompt: bool = False
    train_backbone: bool = False  # keep frozen; only VIB trainable

    def __post_init__(self):
        if self.mode not in ("fixed", "dual"):
            raise ValueError(f"vib.mode must be 'fixed' or 'dual', got {self.mode!r}")
        if self.eval_mode not in ("mean", "sample", "sample_avg_k"):
            raise ValueError(
                f"vib.eval_mode must be one of mean|sample|sample_avg_k, got {self.eval_mode!r}"
            )
        if self.mode == "dual" and self.R_star is None:
            raise ValueError("vib.mode='dual' requires vib.R_star (target nats/token)")


def update_dual_beta(
    beta: float, rate: torch.Tensor, r_star: float, lr_beta: float, beta_max: float
) -> float:
    """Dual ascent update of the Lagrange multiplier ``beta`` toward a target rate.

    beta <- clamp(beta + lr_beta * (R_hat - R_star), 0, beta_max). The gap is
    detached so this update never contributes gradients to the policy.
    """
    gap = float(rate.detach().item()) - float(r_star)
    new_beta = beta + lr_beta * gap
    return float(min(max(new_beta, 0.0), beta_max))


class VIBAdapter(nn.Module):
    """Per-token diagonal-Gaussian information bottleneck over visual tokens.

    forward(H, text_emb, text_mask, sample) -> (Z, rate, stats) where Z is the
    (noisy, reparameterized) representation that replaces H downstream and
    ``rate`` is the free-bits mean per-token KL in nats/token.
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_cross_heads: int = 8,
        free_bits: float = 0.5,
        logvar_min: float = -8.0,
        logvar_max: float = 2.0,
        instruction_conditioned: bool = True,
        residual: bool = False,
        suppress_eps: float = 1e-2,
        shared_noise_per_prompt: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim if cond_dim is not None else d_model
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model
        self.free_bits = float(free_bits)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.instruction_conditioned = bool(instruction_conditioned)
        self.residual = bool(residual)
        self.suppress_eps = float(suppress_eps)
        self.shared_noise_per_prompt = bool(shared_noise_per_prompt)

        if self.instruction_conditioned:
            self.q_proj = nn.Linear(d_model, self.cond_dim)
            self.kv_proj = nn.Linear(d_model, self.cond_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.cond_dim,
                num_heads=n_cross_heads,
                batch_first=True,
            )
        else:
            # Instruction-agnostic ablation: c_i is a learned constant (init zeros).
            self.const_cond = nn.Parameter(torch.zeros(self.cond_dim))

        in_dim = d_model + self.cond_dim
        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, d_model),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, d_model),
        )
        self._init_near_prior()

    def _init_near_prior(self):
        # Zero the final layers so initial mu ~ 0 and logvar ~ 0 (std ~ 1):
        # the posterior starts at the unit prior, so the realized rate starts
        # at ~free_bits and training only spends bits the task actually needs.
        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.zeros_(self.logvar_head[-1].weight)
        nn.init.zeros_(self.logvar_head[-1].bias)

    def _condition(self, H: torch.Tensor, text_emb, text_mask) -> torch.Tensor:
        if not self.instruction_conditioned:
            return self.const_cond.to(H.dtype).view(1, 1, -1).expand(H.size(0), H.size(1), -1)
        if text_emb is None:
            # No text available: fall back to zero context rather than failing.
            return H.new_zeros(H.size(0), H.size(1), self.cond_dim)
        q = self.q_proj(H)
        kv = self.kv_proj(text_emb)
        key_padding_mask = None
        if text_mask is not None:
            # nn.MultiheadAttention: True entries are *ignored*.
            key_padding_mask = ~text_mask.bool()
        c, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return c

    def forward(
        self,
        H: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
        eps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # H: [B, N, d]   text_emb: [B, L, d]   text_mask: [B, L]
        c = self._condition(H, text_emb, text_mask)
        feat = torch.cat([H, c], dim=-1)

        mu = self.mu_head(feat)
        logvar = self.logvar_head(feat).clamp(self.logvar_min, self.logvar_max)

        if sample:
            std = torch.exp(0.5 * logvar)
            if eps is None:
                if generator is not None:
                    eps = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
                else:
                    eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        Z = H + z if self.residual else z

        # KL is computed in fp32 for numerical stability under bf16/fp16 autocast.
        mu32 = mu.float()
        logvar32 = logvar.float()
        kl_per_dim = 0.5 * (torch.exp(logvar32) + mu32.pow(2) - 1.0 - logvar32)
        kl_per_token = kl_per_dim.sum(dim=-1)  # [B, N], raw (pre free-bits)

        kl_fb = kl_per_token.clamp_min(self.free_bits)
        rate = kl_fb.mean()

        # suppressed_frac / kl_per_token are reported on the RAW (pre free-bits)
        # KL, since the diagnostic question is which tokens carry ~no information.
        stats = {
            "rate": rate.detach(),
            "kl_token_mean": kl_per_token.mean().detach(),
            "suppressed_frac": (kl_per_token < self.suppress_eps).float().mean().detach(),
            "mu_abs_mean": mu32.abs().mean().detach(),
            "std_mean": torch.exp(0.5 * logvar32).mean().detach(),
            "kl_per_token": kl_per_token.detach(),
        }
        return Z, rate, stats

    @torch.no_grad()
    def forward_eval(self, H, text_emb=None, text_mask=None, eval_mode="mean", eval_samples=4):
        """Eval-time representation under the configured eval mode.

        - "mean": deterministic Z = mu (default for reported eval).
        - "sample": single noisy reparameterized sample.
        - "sample_avg_k": average the representation over k samples. NOTE this
          averages Z (a single-forward proxy); true prediction-averaging over k
          decodes is left to the eval harness.
        """
        if eval_mode == "mean":
            Z, _, stats = self.forward(H, text_emb, text_mask, sample=False)
            return Z, stats
        if eval_mode == "sample":
            Z, _, stats = self.forward(H, text_emb, text_mask, sample=True)
            return Z, stats
        zs = []
        last_stats = None
        for _ in range(max(1, int(eval_samples))):
            Z, _, last_stats = self.forward(H, text_emb, text_mask, sample=True)
            zs.append(Z)
        return torch.stack(zs, 0).mean(0), last_stats

"""Magma: Momentum-Aligned Gradient Masking wrapper for PyTorch optimizers."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


class Magma:
    """
    A Pytorch optimizer wraper with block-wise stochastic masking
    modulated by momentum-gradient alignment.
    As explained here https://arxiv.org/abs/2602.15322
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mask_prob: float = 0.5,
        tau: float = 2.0,
        momentum_beta: float = 0.9,
        alignment_ema: float = 0.9,
        exclude: set[Tensor] | None = None,
    ) -> None:
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.optimizer = optimizer
        self.mask_prob = mask_prob
        self.tau = tau
        self.momentum_beta = momentum_beta
        self.alignment_ema = alignment_ema
        self._exclude_ids: set[int] = {id(t) for t in (exclude or ())}

        # Magma-specific per-parameter state keyed by param id for momentum and alignment
        self._state: dict[int, dict[str, Tensor]] = {}

    def step(self, closure=None):
        """Perform a single Magma-wrapped optimisation step."""

        # --- Phase 1: pre-step (momentum, alignment, snapshot) --------
        snapshots: dict[int, Tensor] = {}  # id(p) -> p.data clone
        alignment_scores: dict[int, float] = {}

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None or id(p) in self._exclude_ids:
                    continue

                pid = id(p)
                grad = p.grad.detach()

                # Lazily initialise per-param Magma state
                if pid not in self._state:
                    self._state[pid] = {
                        "momentum": torch.zeros_like(p.data),
                        "alignment": torch.tensor(1.0, device=p.device),
                    }

                st = self._state[pid]

                #  μ_t = β μ_{t-1} + (1-β) g_t
                st["momentum"].mul_(self.momentum_beta).add_(
                    grad, alpha=1.0 - self.momentum_beta
                )

                #  cossim(μ_t, g_t)
                cos = torch.nn.functional.cosine_similarity(
                    st["momentum"].flatten().unsqueeze(0),
                    grad.flatten().unsqueeze(0),
                ).item()
                
                # sigmoid(cos/τ) 
                s_tilde = torch.sigmoid(
                    torch.tensor(cos / self.tau, device=p.device)
                ).item()

                # EMA smoothing
                s_prev = st["alignment"].item()
                s = self.alignment_ema * s_prev + (1.0 - self.alignment_ema) * s_tilde
                st["alignment"].fill_(s)

                alignment_scores[pid] = s

                # Save current params
                snapshots[pid] = p.data.clone()

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in snapshots:
                    continue

                # m_t ~ Bernoulli(mask_prob)
                mask = float(torch.bernoulli(torch.tensor(self.mask_prob)).item())
                s = alignment_scores[pid]
                # blend = s_t * m_t
                blend = s * mask

                if blend == 0.0:
                    # Fully revert parameter to pre-step value
                    p.data.copy_(snapshots[pid])
                elif blend != 1.0:
                    # θ = blend*θ_new + (1-blend)*θ_old
                    p.data.mul_(blend).add_(snapshots[pid], alpha=1.0 - blend)
                # blend == 1.0, keep as it is

        return loss

    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group: dict):
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        id_to_key: dict[int, tuple[int, int]] = {}
        for gi, group in enumerate(self.optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                id_to_key[id(p)] = (gi, pi)

        magma_state = {}
        for pid, st in self._state.items():
            key = id_to_key.get(pid)
            if key is not None:
                magma_state[key] = {k: v.clone() for k, v in st.items()}

        return {
            "base": self.optimizer.state_dict(),
            "magma_state": magma_state,
            "mask_prob": self.mask_prob,
            "tau": self.tau,
            "momentum_beta": self.momentum_beta,
            "alignment_ema": self.alignment_ema,
        }

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict["base"])
        self.mask_prob = state_dict["mask_prob"]
        self.tau = state_dict["tau"]
        self.momentum_beta = state_dict["momentum_beta"]
        self.alignment_ema = state_dict["alignment_ema"]

        key_to_id: dict[tuple[int, int], int] = {}
        for gi, group in enumerate(self.optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                key_to_id[(gi, pi)] = id(p)

        self._state = {}
        for key, st in state_dict["magma_state"].items():
            key = tuple(key) if isinstance(key, list) else key
            pid = key_to_id.get(key)
            if pid is not None:
                self._state[pid] = {k: v.clone() for k, v in st.items()}

    def __getattr__(self, name: str):
        # Fallback to base optimizer for anything not on Magma itself
        try:
            return getattr(self.optimizer, name)
        except AttributeError:
            raise AttributeError(
                f"Neither 'Magma' nor the base optimizer have attribute '{name}'"
            )

    def __repr__(self) -> str:
        return (
            f"Magma(\n"
            f"  mask_prob={self.mask_prob},\n"
            f"  tau={self.tau},\n"
            f"  momentum_beta={self.momentum_beta},\n"
            f"  alignment_ema={self.alignment_ema},\n"
            f"  base={self.optimizer}\n"
            f")"
        )

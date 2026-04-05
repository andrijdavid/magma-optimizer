import torch
import torch.nn as nn
from magma import Magma


def _make_model():
    """Simple linear model for testing."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    return model


def _toy_data(n=32):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)
    return x, y


class TestMagmaWrapping:
    def test_wrap_adam(self):
        model = _make_model()
        base = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt = Magma(base)
        assert opt.param_groups is base.param_groups

    def test_wrap_sgd(self):
        model = _make_model()
        base = torch.optim.SGD(model.parameters(), lr=1e-2)
        opt = Magma(base)
        assert opt.param_groups is base.param_groups


class TestMagmaStep:
    def test_step_reduces_loss(self):
        model = _make_model()
        base = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt = Magma(base, mask_prob=1.0)  # no masking to guarantee progress
        x, y = _toy_data()

        loss_fn = nn.MSELoss()
        initial_loss = loss_fn(model(x), y).item()

        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        final_loss = loss_fn(model(x), y).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_mask_prob_one_scales_by_alignment(self):
        """With mask_prob=1.0, every param is updated but scaled by alignment."""
        model = _make_model()
        base = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = Magma(base, mask_prob=1.0, tau=2.0)
        x, y = _toy_data()

        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        # After one step, alignment states should exist for all params with grad
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    pid = id(p)
                    assert pid in opt._state
                    assert "alignment" in opt._state[pid]

    def test_mask_prob_zero_no_param_change(self):
        """With mask_prob=0, parameters should not change (all updates reverted)."""
        model = _make_model()
        base = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt = Magma(base, mask_prob=0.0)
        x, y = _toy_data()

        # Save original params
        orig_params = {name: p.data.clone() for name, p in model.named_parameters()}

        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        opt.step()

        for name, p in model.named_parameters():
            assert torch.equal(p.data, orig_params[name]), (
                f"Param {name} changed with mask_prob=0"
            )


class TestExclude:
    def test_exclude_skips_params(self):
        model = _make_model()
        # Exclude the first layer's weight
        excluded_param = list(model.parameters())[0]
        base = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt = Magma(base, mask_prob=1.0, exclude={excluded_param})
        x, y = _toy_data()

        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # Save the excluded param before step
        orig = excluded_param.data.clone()
        opt.step()

        # Excluded param should have been updated by base optimizer directly
        # (no masking/alignment applied), so it should differ from original
        # unless gradient is zero (unlikely here)
        pid = id(excluded_param)
        assert pid not in opt._state, "Excluded param should not have Magma state"


class TestStateDictRoundtrip:
    def test_save_load(self):
        model = _make_model()
        base = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt = Magma(base, mask_prob=0.7, tau=1.5)
        x, y = _toy_data()

        # Run a few steps to populate state
        for _ in range(3):
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            opt.step()

        sd = opt.state_dict()

        # Create a fresh wrapper and load
        model2 = _make_model()
        base2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        opt2 = Magma(base2)
        opt2.load_state_dict(sd)

        assert opt2.mask_prob == 0.7
        assert opt2.tau == 1.5
        assert opt2.momentum_beta == opt.momentum_beta
        assert opt2.alignment_ema == opt.alignment_ema

        # Check magma state was restored
        for (gi, group) in enumerate(opt2.param_groups):
            for pi, p in enumerate(group["params"]):
                pid = id(p)
                if pid in opt2._state:
                    assert "momentum" in opt2._state[pid]
                    assert "alignment" in opt2._state[pid]

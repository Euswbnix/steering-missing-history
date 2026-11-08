"""Independent analytic checks for the CfC cell; no ncps dependency.

These tests cover gate limits, the bias-only update, time-step sensitivity, shape
validation, initialization, bounded states and gradients. They complement the
numerical reference comparisons in tests/test_cfc_oracle.py.

An inverted interpolation can match the reference after swapping its candidate
heads. The negative control below shows why numerical agreement alone is
insufficient: the gate-limit check detects that inversion independently.
"""

import math

import pytest
import torch
import torch.nn as nn

from models.cfc import CfCCell, CfC

I, H, B = 32, 64, 256
TIGHT = dict(rtol=0, atol=0)


def cell(seed=0, i=I, h=H, dtype=torch.float64, **kw):
    torch.manual_seed(seed)
    return CfCCell(i, h, **kw).to(dtype)


def inputs(seed=1, b=B, i=I, h=H, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, i, generator=g, dtype=dtype),
            torch.randn(b, h, generator=g, dtype=dtype) * 0.3)


# ------------------------------------------------------------------ 1. gate limits ----
# THE anchor that closes the remap hole. It pins WHICH candidate the gate drives the
# state toward, intrinsically, so no naming permutation can satisfy both it and the
# oracle. Zero f_head's weight and o_head entirely, then drive the bias to +-10.

def _pin_gate(c, f_bias):
    with torch.no_grad():
        c.f_head.weight.zero_(); c.f_head.bias.fill_(f_bias)
        c.o_head.weight.zero_(); c.o_head.bias.zero_()


def test_gate_limit_large_dt_goes_to_fast_candidate_g():
    c = cell(); z, h = inputs()
    _pin_gate(c, +10.0)
    out = c(z, h, torch.full((B, 1), 50.0, dtype=torch.float64))
    x = torch.cat([z, h], dim=-1)
    torch.testing.assert_close(out, torch.tanh(c.g_head(x)), **TIGHT)


def test_gate_limit_negative_slope_goes_to_equilibrium_candidate_h():
    c = cell(); z, h = inputs()
    _pin_gate(c, -10.0)
    out = c(z, h, torch.full((B, 1), 50.0, dtype=torch.float64))
    x = torch.cat([z, h], dim=-1)
    torch.testing.assert_close(out, torch.tanh(c.h_head(x)), **TIGHT)


class InvertedCfCCell(CfCCell):
    """The wrong cell the oracle cannot see: lerp the other way round."""

    def forward(self, z, h, dt):
        x = torch.cat([z, h], dim=-1)
        cand_h = torch.tanh(self.h_head(x))
        cand_g = torch.tanh(self.g_head(x))
        gamma = torch.sigmoid(self.f_head(x) * dt + self.o_head(x))
        return gamma * cand_h + (1.0 - gamma) * cand_g          # <-- inverted


def test_inverted_cell_is_caught_by_gate_limit():
    """The whole point of Layer 1. This same cell passes the oracle at 0.0 -- see
    tests/test_cfc_oracle.py::test_oracle_alone_cannot_catch_the_inverted_cell."""
    torch.manual_seed(0)
    bad = InvertedCfCCell(I, H).double()
    z, h = inputs()
    _pin_gate(bad, +10.0)
    out = bad(z, h, torch.full((B, 1), 50.0, dtype=torch.float64))
    x = torch.cat([z, h], dim=-1)
    residual = (out - torch.tanh(bad.g_head(x))).abs().max().item()
    assert residual > 0.1, "the gate-limit anchor failed to separate an inverted lerp"


# --------------------------------------------------------- 2. bias-only closed form ----

def test_bias_only_closed_form():
    """All weights zero, z = 0, h = 0  =>  h1 = (1-sig(b_o))*tanh(b_h) + sig(b_o)*tanh(b_g).
    Pins every bias to the right term of the update."""
    c = cell()
    with torch.no_grad():
        for head in (c.h_head, c.g_head, c.f_head, c.o_head):
            head.weight.zero_()
        c.h_head.bias.fill_(0.7); c.g_head.bias.fill_(-1.3)
        c.f_head.bias.fill_(0.0); c.o_head.bias.fill_(0.4)
    out = c(torch.zeros(1, I, dtype=torch.float64), torch.zeros(1, H, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64))
    s = 1.0 / (1.0 + math.exp(-0.4))
    expected = (1 - s) * math.tanh(0.7) + s * math.tanh(-1.3)
    torch.testing.assert_close(out, torch.full((1, H), expected, dtype=torch.float64),
                               rtol=0, atol=1e-15)


# ------------------------------------------------------ 3. dt enters only via f_head ----

def test_dt_enters_only_through_f_head():
    """Zeroing the time-dependent head makes the update independent of dt."""
    c = cell()
    with torch.no_grad():
        c.f_head.weight.zero_(); c.f_head.bias.zero_()
    z, h = inputs()
    ref = c(z, h, torch.zeros(B, 1, dtype=torch.float64))
    for v in (0.1, 1.0, 10.0, 1e4):
        torch.testing.assert_close(c(z, h, torch.full((B, 1), v, dtype=torch.float64)),
                                   ref, **TIGHT)


# ------------------------------------------------------- 4. constant-dt degeneracy ----

@pytest.mark.parametrize("c_scale", [0.25, 2.0, 7.0])
def test_constant_dt_degeneracy(c_scale):
    """A cell with f_head scaled by c, evaluated at dt=1, EQUALS the original at dt=c.

    At a fixed step, dt can be absorbed into the gate's affine parameters. This
    checks the algebra; it does not establish performance under irregular sampling.
    """
    a = cell(); z, h = inputs()
    b = cell()                                    # same seed => same weights
    with torch.no_grad():
        b.f_head.weight.mul_(c_scale); b.f_head.bias.mul_(c_scale)
    # NOT atol=0, and the reason is worth knowing: (w*c).x and (w.x)*c are algebraically
    # equal but the multiply lands at a different point in the expression, so they differ
    # by float64 rounding (measured max 1.4e-15). The bit-exact anchors above are the ones
    # where the arithmetic is literally identical. An inverted lerp errs by O(1), so the
    # margin here is still eleven orders of magnitude -- a tolerance that has to be TUNED
    # is a tolerance that hides failures; this one is derived, not tuned.
    torch.testing.assert_close(
        b(z, h, torch.ones(B, 1, dtype=torch.float64)),
        a(z, h, torch.full((B, 1), c_scale, dtype=torch.float64)), rtol=0, atol=1e-14)


# ------------------------------------------- 5. dt sensitivity and shape contract ----

def test_per_sample_dt_actually_varies_the_output():
    c = cell(); z, h = inputs(b=8)
    dt = torch.tensor([[0.1], [5.0]] * 4, dtype=torch.float64)
    out = c(z, h, dt)
    assert (out[0] - out[1]).abs().max() > 1e-3


@pytest.mark.parametrize("bad_shape", [(B,), (B, 4), (1, 1), ()])
def test_dt_shape_contract_rejects_wrong_shapes(bad_shape):
    """Closes the entire ncps bug class in one assert -- including the silent wrong-axis
    broadcast that raises no error when B happens to equal the hidden width, and which
    batch-independence testing cannot catch because the single-row rerun broadcasts the
    same wrong way."""
    c = cell(); z, h = inputs()
    with pytest.raises(AssertionError):
        c(z, h, torch.ones(bad_shape, dtype=torch.float64))


# ---------------------------------------------------------- 6. initialisation stats ----

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gate_is_initialised_in_its_high_gradient_regime(seed):
    """The oracle COPIES weights before comparing, so the object it tests is never the
    object that gets trained. A cell initialised N(0,8) with bias -12 -- 93.6% gate-
    saturated at step 0 -- passes every oracle assertion at 0.0, and so does torch's
    default kaiming_uniform_(a=sqrt(5)), which gives a ~40% narrower gate than the
    paper's Xavier gain 1."""
    c = cell(seed=seed, dtype=torch.float32)
    g = torch.Generator().manual_seed(100 + seed)
    z = torch.randn(1024, I, generator=g)
    h = torch.zeros(1024, H)
    x = torch.cat([z, h], dim=-1)
    gamma = torch.sigmoid(c.f_head(x) * 1.0 + c.o_head(x))
    assert 0.45 <= gamma.mean().item() <= 0.55
    assert 0.15 <= gamma.std().item() <= 0.23
    assert ((gamma < 0.01) | (gamma > 0.99)).float().mean().item() == 0.0


# ------------------------------------------------------------- 7. boundedness both ----

def test_state_is_strictly_bounded_random_inputs():
    c = cell(); z, h = inputs()
    for v in (0.0, 1.0, 1e3):
        out = c(z, h, torch.full((B, 1), v, dtype=torch.float64))
        assert out.abs().max().item() < 1.0


def test_state_is_strictly_bounded_over_a_long_rollout():
    torch.manual_seed(0)
    m = CfC(I, H).double()
    g = torch.Generator().manual_seed(7)
    z = torch.randn(4, 200, I, generator=g, dtype=torch.float64) * 3
    out, _ = m(z, dt=torch.rand(4, 200, generator=g, dtype=torch.float64) * 10)
    assert out.abs().max().item() < 1.0


def test_readout_removes_the_range_ceiling():
    """A linear readout can represent steering angles beyond the bounded state range."""
    readout = nn.Linear(H, 1)
    with torch.no_grad():
        readout.weight.fill_(2.0); readout.bias.zero_()
    y = readout(torch.full((1, H), 0.9))
    assert y.abs().item() > 100.0


def test_gradcheck_through_z_h_and_dt():
    c = cell(i=4, h=3)
    z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    h = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    dt = (torch.rand(2, 1, dtype=torch.float64) + 0.5).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b, d: c(a, b, d), (z, h, dt), eps=1e-6,
                                    atol=1e-8)

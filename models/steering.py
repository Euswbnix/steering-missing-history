"""Steering models and sparse-history forwarding on precomputed visual features."""

import torch
import torch.nn as nn

from .cfc import CfC

MIN_DT = 1e-3


class LTCBlock(nn.Module):
    """ncps LTC driven one step at a time, with dt passed per sample as (B, 1).

    Stepping manually is what avoids the known defect where a batch equal to the hidden size makes
    `timespans[:, t].squeeze()` broadcast one sample's elapsed time onto another (ncps 1.0.1,
    PR #85). The reversal potentials are seeded per run instead of the library's fixed 1111."""

    def __init__(self, hidden, seed, ode_unfolds=6, compile_cell=False):
        super().__init__()
        from ncps.torch.ltc_cell import LTCCell
        from ncps.wirings import FullyConnected
        self.cell = LTCCell(FullyConnected(hidden, erev_init_seed=seed), in_features=hidden,
                            ode_unfolds=ode_unfolds)
        self.hidden = hidden
        # The compiled cell wraps the same module and shares its parameters, but is kept out of
        # the registered submodules so the state dict (and every checkpoint) stays identical.
        # dynamic=True: the last batch of every session has its own size, and with static shapes
        # each new size can force recompilation.
        if compile_cell:
            import torch._dynamo
            torch._dynamo.config.recompile_limit = 64
        step = torch.compile(self.cell, dynamic=True) if compile_cell else self.cell
        object.__setattr__(self, "_step", step)

    def forward(self, x, dt):
        h = x.new_zeros(x.shape[0], self.hidden)
        out = []
        for i in range(x.shape[1]):
            # contiguous slices: their strides would otherwise change with the window length and
            # force a recompilation per length (and a silent fall-back to eager after eight)
            h, _ = self._step(x[:, i].contiguous(), h, dt[:, i:i + 1].clamp(min=MIN_DT).contiguous())
            out.append(h)
        return torch.stack(out, 1)


class TransformerBlock(nn.Module):
    """Causal Transformer with real elapsed-time encoding.

    One pre-norm encoder layer (64 wide, 4 heads, feed-forward 128, GELU, no dropout) and no
    final LayerNorm. Time enters as a parameter-free sinusoidal code of each kept frame's real time
    relative to the target frame, p = (t_i - t_target) / 0.1 s, added to the projected features;
    dropped frames leave gaps in p rather than being renumbered. Attention is causal and ignores
    padding. The kept frames are in time order with the target frame last, so the caller reads the
    last valid token. MHA's inference fast path is switched off so eval runs the same maths as
    training (and no nested-tensor conversion can happen)."""

    def __init__(self, hidden=64, heads=4, ff=128, time_code="real"):
        super().__init__()
        # Alternative time encodings are available for controlled comparisons.
        assert time_code in ("real", "index", "none"), time_code
        self.mode = time_code
        torch.backends.mha.set_fastpath_enabled(False)
        self.layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=ff, dropout=0.0,
                                                activation="gelu", batch_first=True,
                                                norm_first=True)
        j = torch.arange(hidden // 2, dtype=torch.float32)
        self.register_buffer("inv_freq", 10000.0 ** (-2 * j / hidden), persistent=False)

    def code(self, p):
        """The sinusoid for positions p in 0.1 s steps (real) or in kept-frame indices (index)."""
        p = p[..., None] * self.inv_freq                                   # (B, T, hidden/2)
        return torch.stack([torch.sin(p), torch.cos(p)], -1).flatten(-2)   # dims 2j, 2j+1

    def time_code(self, rel):
        return self.code(rel / 0.1)

    def forward(self, x, rel, valid):
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        if self.mode == "real":
            x = x + self.time_code(rel)
        elif self.mode == "index":
            last = valid.long().sum(1, keepdim=True) - 1                   # the target frame
            x = x + self.code((torch.arange(T, device=x.device)[None] - last).to(x.dtype))
        return self.layer(x, src_mask=causal, src_key_padding_mask=~valid)


class SteeringModel(nn.Module):
    """Shared projection and readout; the temporal block is the only difference."""

    def __init__(self, kind, dim, hidden=64, seed=0, ode_unfolds=6, compile_ltc=False, dt_unit=1.0,
                 time_code="real", n_out=1):
        """Use n_out=1 for the current angle or n_out=4 for anticipation.

        The single_frame baseline applies an MLP to the current frame only.
        """
        super().__init__()
        self.kind = kind
        # dt arrives in seconds; dt_unit permits an explicit, fixed unit conversion.
        self.dt_unit = float(dt_unit)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, hidden)
        self.rnn = (nn.LSTM(hidden + 1, hidden, batch_first=True) if kind == "lstm" else
                    CfC(hidden, hidden) if kind == "cfc" else
                    TransformerBlock(hidden, time_code=time_code) if kind == "transformer" else
                    nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                  nn.Linear(4 * hidden, hidden)) if kind == "single_frame" else
                    LTCBlock(hidden, seed, ode_unfolds, compile_ltc))
        self.head = nn.Linear(hidden, n_out)

    def forward(self, f, dt, valid, rel=None):
        """rel: each kept frame's time minus the target frame's, in seconds (Transformer only)."""
        x = self.proj(self.norm(f)) * valid[:, :, None]
        last = valid.float().cumsum(1).argmax(1)
        rows = torch.arange(len(x), device=x.device)
        if self.kind == "single_frame":                       # the current frame only, whatever was kept
            return self.head(self.rnn(x[rows, last])).squeeze(-1)
        if self.dt_unit != 1.0:
            dt = dt / self.dt_unit
        if self.kind == "lstm":
            out, _ = self.rnn(torch.cat([x, dt[:, :, None]], -1))
        elif self.kind == "cfc":
            out, _ = self.rnn(x, dt)
        elif self.kind == "transformer":
            out = self.rnn(x, rel, valid)
        else:
            out = self.rnn(x, dt)
        return self.head(out[rows, last]).squeeze(-1)

    def blocks(self):
        n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.rnn.parameters() if not p.requires_grad)
        return dict(projection=n(self.proj), temporal=n(self.rnn), readout=n(self.head),
                    temporal_frozen=frozen)


def masked(t, keep, gen):
    """Keep the current frame and a random `keep` share of the history; dt over what survives."""
    mask = torch.rand(t.shape, device=t.device, generator=gen) < keep if keep < 1 else \
        torch.ones_like(t, dtype=torch.bool)
    mask[:, -1] = True
    T = t.shape[1]
    order = torch.argsort(torch.where(mask, torch.arange(T, device=t.device).expand_as(t),
                                      torch.full_like(t, T + 1, dtype=torch.long)), dim=1)
    return mask, order


def forward_masked(model, f, t, keep, gen):
    mask, order = masked(t, keep, gen)
    n = int(mask.sum(1).max())
    idx = order[:, :n]
    kf = torch.gather(f, 1, idx[:, :, None].expand(-1, -1, f.shape[2]))
    kt = torch.gather(t, 1, idx)
    valid = torch.gather(mask, 1, idx)
    dt = torch.zeros_like(kt)
    dt[:, 1:] = (kt[:, 1:] - kt[:, :-1]).clamp(min=0)
    rel = kt - t[:, -1:]                      # the target frame is always kept, and kept last
    return model(kf, dt.to(kf.dtype), valid, rel.to(kf.dtype))

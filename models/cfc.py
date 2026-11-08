"""Closed-form continuous-time cell and sequence wrapper.

The caller supplies elapsed times in a consistent unit. The default steering
configuration uses seconds, and training and inference must use the same unit.
CfCCell matches the default ncps==1.0.1 cell after the weight mapping exercised
in tests/test_cfc_oracle.py; tests/test_cfc_anchors.py check the update independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeCunTanh(nn.Module):
    """LeCun (1998) scaled tanh. No parameters. Used only by the optional backbone."""

    def forward(self, x):
        return 1.7159 * torch.tanh(0.666 * x)


class CfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, backbone_units=0, backbone_layers=0,
                 mask=None, mask_gate=True):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.mask_gate = mask_gate
        if backbone_layers > 0:
            layers = [nn.Linear(input_size + hidden_size, backbone_units), LeCunTanh()]
            for _ in range(backbone_layers - 1):
                layers += [nn.Linear(backbone_units, backbone_units), LeCunTanh()]
            self.backbone = nn.Sequential(*layers)
            D = backbone_units
        else:
            self.backbone = None
            D = input_size + hidden_size
        self.h_head = nn.Linear(D, hidden_size)
        self.g_head = nn.Linear(D, hidden_size)
        self.f_head = nn.Linear(D, hidden_size)
        self.o_head = nn.Linear(D, hidden_size)
        # buffer, NEVER nn.Parameter -- ncps registers its mask as a frozen Parameter,
        # which is why `sum(p.numel() for p in m.parameters())` overcounts by 24.4%.
        self.register_buffer("mask", None if mask is None else mask.clone().float())
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier gain 1.0 is what the paper documents; zero biases match the reference.
        # Measured at I=32, H=64, h=0, dt=1: gamma mean 0.501, std 0.190, 0.00% of units
        # saturated outside (0.01, 0.99). Asserted in tests/test_cfc_anchors.py, because
        # the oracle copies weights and therefore never tests initialisation.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _aff(self, head, x, maskable):
        if self.mask is None or not maskable:
            return head(x)
        return F.linear(x, head.weight * self.mask, head.bias)

    def forward(self, z, h, dt):
        # (B,1) is enforced: ncps' wrapper does `timespans[:, t].squeeze()`, which strips
        # singleton dims, so B>1 misbroadcasts -- and when B happens to equal the hidden
        # width it is silently wrong rather than raising. This assert closes that class.
        assert dt.dim() == 2 and dt.shape == (z.shape[0], 1), \
            f"dt must be (B,1); got {tuple(dt.shape)}"
        x = torch.cat([z, h], dim=-1)          # order is load-bearing: matches ncps
        if self.backbone is not None:
            x = self.backbone(x)
        cand_h = torch.tanh(self._aff(self.h_head, x, True))
        cand_g = torch.tanh(self._aff(self.g_head, x, True))
        gamma = torch.sigmoid(self._aff(self.f_head, x, self.mask_gate) * dt
                              + self._aff(self.o_head, x, self.mask_gate))
        return (1.0 - gamma) * cand_h + gamma * cand_g


class CfC(nn.Module):
    """Return the state sequence and final state, with optional elapsed times and
    initial state for stateful chunked rollout."""

    def __init__(self, input_size, hidden_size, **kw):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, **kw)

    def forward(self, z, dt=None, hx=None):
        B, T, _ = z.shape
        h = z.new_zeros(B, self.cell.hidden_size) if hx is None else hx
        if dt is None:
            dt = z.new_ones(B, T)             # honest regularly-sampled mode
        out = []
        for t in range(T):
            h = self.cell(z[:, t], h, dt[:, t:t + 1])
            out.append(h)
        return torch.stack(out, dim=1), h

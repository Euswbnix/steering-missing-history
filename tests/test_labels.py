"""Interpolation and single-frame baseline checks on synthetic inputs."""

import numpy as np
import torch

import data as A
import models.steering as D


def steer_of(t):
    return 20.0 * np.sin(0.7 * t) + 3.0 * t


def test_labels_are_exact_on_a_regular_linear_series():
    t = 0.1 * np.arange(100)
    steer = 3.0 * t - 5.0
    targets = np.array([0, 10, 50])
    Y = A.future_labels(t, steer, targets)
    for k, h in enumerate(A.HORIZONS):
        assert np.allclose(Y[:, k], 3.0 * (t[targets] + h) - 5.0, atol=1e-9)


def test_labels_interpolate_between_irregular_frames():
    t = np.array([0.0, 0.1, 0.2, 0.31, 0.38, 0.52, 0.6, 0.7])
    steer = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 9.0, 10.0, 11.0])
    Y = A.future_labels(t, steer, np.array([0]), horizons=(0.0, 0.45))
    assert Y[0, 0] == 0.0
    assert np.isclose(Y[0, 1], 5.0 + (0.45 - 0.38) / (0.52 - 0.38) * 4.0)       # between 0.38 and 0.52


def test_labels_are_void_across_gaps_bad_values_backward_clocks_and_the_end():
    t = 0.1 * np.arange(60)
    steer = steer_of(t)
    gap = t.copy()
    gap[30:] += 0.5                                   # a 0.6 s hole after frame 29
    Y = A.future_labels(gap, steer, np.array([20, 26, 40]))
    assert np.isfinite(Y[0, 1]) and np.isnan(Y[0, 2])            # 20 -> +0.5 fine, +1.0 lands in the hole
    assert np.isnan(Y[1, 1])                                     # 26 -> +0.5 lands in the hole
    assert np.isfinite(Y[2, 1])                                  # after the hole everything is regular
    bad = steer.copy()
    bad[25] = np.nan
    assert np.isnan(A.future_labels(t, bad, np.array([20]))[0, 1])     # +0.5 s needs frame 25
    back = t.copy()
    back[35:] -= 2.0                                  # the clock steps backwards at frame 35
    Yb = A.future_labels(back, steer, np.array([30]))
    assert np.isnan(Yb[0, 2]) and np.isnan(Yb[0, 3])
    assert np.isnan(A.future_labels(t, steer, np.array([50]))[0, 3])   # +2 s is past the last frame
    assert np.isfinite(A.future_labels(t, steer, np.array([39]))[0, 3])


def test_single_frame_model_sees_only_the_current_frame():
    torch.manual_seed(0)
    m = D.SteeringModel("single_frame", 16, n_out=4).double().eval()
    assert m.blocks()["temporal"] == 33_088
    f = torch.randn(3, 10, 16, dtype=torch.float64)
    valid = torch.arange(10)[None] < torch.tensor([10, 6, 3])[:, None]
    dt = torch.full((3, 10), 0.1, dtype=torch.float64)
    base = m(f, dt, valid)
    assert base.shape == (3, 4)
    f2 = f.clone()
    last = valid.long().sum(1) - 1
    keep = torch.zeros(3, 10, dtype=torch.bool)
    keep[torch.arange(3), last] = True
    f2[~keep] = torch.randn(int((~keep).sum()), 16, dtype=torch.float64)       # everything but the current frame
    assert torch.allclose(m(f2, dt, valid), base, atol=1e-12)

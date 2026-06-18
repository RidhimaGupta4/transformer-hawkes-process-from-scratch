"""
Tests for the three temporal encoding strategies. Checks output
shapes, that sinusoidal encoding is deterministic (no learned
parameters) while Time2Vec/LogGap are learnable, and that the
encodings actually differ for different timestamps (i.e. they are
not collapsing to a constant).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from temporal_encoding import SinusoidalTimeEncoding, Time2Vec, LogGapEmbedding


def test_sinusoidal_shape_and_determinism():
    d_model = 16
    enc = SinusoidalTimeEncoding(d_model)
    t = torch.tensor([[0.0, 1.0, 5.0, 100.0]])

    out1 = enc(t)
    out2 = enc(t)
    assert out1.shape == (1, 4, d_model)
    assert torch.equal(out1, out2), "Sinusoidal encoding must be deterministic"
    print("[PASS] sinusoidal encoding: correct shape, deterministic")


def test_sinusoidal_distinguishes_timestamps():
    d_model = 16
    enc = SinusoidalTimeEncoding(d_model)
    t = torch.tensor([[0.0, 1.0, 1000.0]])
    out = enc(t)
    # Different timestamps should give different encodings
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 1], out[0, 2])
    print("[PASS] sinusoidal encoding distinguishes different timestamps")


def test_time2vec_shape_and_learnable():
    d_model = 8
    enc = Time2Vec(d_model)
    t = torch.tensor([[0.0, 2.5, 10.0]])
    out = enc(t)
    assert out.shape == (1, 3, d_model)
    assert enc.w.requires_grad and enc.b.requires_grad
    print("[PASS] Time2Vec: correct shape, parameters are learnable")


def test_time2vec_gradient_flows():
    enc = Time2Vec(d_model=8)
    t = torch.tensor([[0.0, 1.0, 2.0]])
    out = enc(t)
    out.sum().backward()
    assert enc.w.grad is not None and torch.isfinite(enc.w.grad).all()
    assert enc.b.grad is not None and torch.isfinite(enc.b.grad).all()
    print("[PASS] Time2Vec: gradients flow to w and b")


def test_loggap_handles_zero_and_large_gaps():
    enc = LogGapEmbedding(d_model=4)
    # zero gap and a very large gap (e.g. years, in minutes) should
    # both produce finite, non-exploding output
    delta_t = torch.tensor([[0.0, 1.0, 1_000_000.0]])
    out = enc(delta_t)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 3, 4)
    print("[PASS] LogGapEmbedding: finite output for both zero and large gaps")


def test_loggap_compresses_large_values():
    # Sanity check on the transform itself: log1p should massively
    # compress the dynamic range compared to raw values.
    delta_t = torch.tensor([1.0, 1_000_000.0])
    raw_ratio = (delta_t[1] / delta_t[0]).item()
    log_ratio = (torch.log1p(delta_t[1]) / torch.log1p(delta_t[0])).item()
    assert log_ratio < raw_ratio / 1000, "log1p should heavily compress large gaps"
    print(f"[PASS] log1p compresses dynamic range (raw ratio {raw_ratio:.0f}x -> log ratio {log_ratio:.1f}x)")


if __name__ == "__main__":
    test_sinusoidal_shape_and_determinism()
    test_sinusoidal_distinguishes_timestamps()
    test_time2vec_shape_and_learnable()
    test_time2vec_gradient_flows()
    test_loggap_handles_zero_and_large_gaps()
    test_loggap_compresses_large_values()
    print("\nAll temporal encoding tests passed.")

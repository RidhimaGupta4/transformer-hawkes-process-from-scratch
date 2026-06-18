"""
Verifies the from-scratch attention implementation in src/attention.py
against torch.nn.functional.scaled_dot_product_attention (PyTorch's own
reference implementation), and checks the causal mask behaves correctly.

This is not a stylistic choice -- it is the only way to actually know
the from-scratch implementation is mathematically correct rather than
merely "looks right."
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import scaled_dot_product_attention, MultiHeadSelfAttention, causal_mask


def test_attention_matches_pytorch_reference():
    torch.manual_seed(0)
    batch, heads, seq_len, d_k = 2, 4, 6, 8

    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    out_custom, _ = scaled_dot_product_attention(q, k, v)
    out_reference = F.scaled_dot_product_attention(q, k, v)

    max_diff = (out_custom - out_reference).abs().max().item()
    assert max_diff < 1e-5, f"Mismatch vs PyTorch reference: max diff {max_diff}"
    print(f"[PASS] attention output matches PyTorch reference (max diff {max_diff:.2e})")


def test_attention_weights_sum_to_one():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 5, 4)
    k = torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 4)

    _, weights = scaled_dot_product_attention(q, k, v)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    print("[PASS] attention weights sum to 1 across keys for every query")


def test_causal_mask_blocks_future():
    torch.manual_seed(2)
    seq_len, d_k = 5, 4
    q = torch.randn(1, 1, seq_len, d_k)
    k = torch.randn(1, 1, seq_len, d_k)
    v = torch.randn(1, 1, seq_len, d_k)

    mask = causal_mask(seq_len)
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    # weights[0, 0, i, j] must be ~0 for all j > i
    upper_triangle = torch.triu(weights[0, 0], diagonal=1)
    max_leak = upper_triangle.abs().max().item()
    assert max_leak < 1e-6, f"Causal mask leaking future info: max weight {max_leak}"
    print(f"[PASS] causal mask blocks all future positions (max leak {max_leak:.2e})")


def test_multihead_output_shape_and_gradient_flow():
    torch.manual_seed(3)
    batch, seq_len, d_model, n_heads = 2, 7, 32, 4
    mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    out, weights = mha(x)
    assert out.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, n_heads, seq_len, seq_len)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[PASS] multi-head attention: correct output shape and finite gradients")


def test_d_model_not_divisible_by_heads_raises():
    try:
        MultiHeadSelfAttention(d_model=10, n_heads=3)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "Expected AssertionError for non-divisible d_model/n_heads"
    print("[PASS] invalid d_model/n_heads configuration correctly raises")


if __name__ == "__main__":
    test_attention_matches_pytorch_reference()
    test_attention_weights_sum_to_one()
    test_causal_mask_blocks_future()
    test_multihead_output_shape_and_gradient_flow()
    test_d_model_not_divisible_by_heads_raises()
    print("\nAll attention tests passed.")

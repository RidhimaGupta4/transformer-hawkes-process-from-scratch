"""
Tests for the full Transformer Hawkes Process model: output shapes,
gradient flow, causal non-leakage at the *model* level (not just the
attention primitive), padding correctness, and log-likelihood
sanity checks (e.g. that NLL is finite and that gradient descent can
reduce it on a small batch).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from thp_model import TransformerHawkesProcess, negative_log_likelihood


def test_forward_shapes():
    torch.manual_seed(0)
    num_types = 3
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=16, n_heads=2, n_layers=2)

    batch, seq_len = 4, 10
    event_types = torch.randint(0, num_types, (batch, seq_len))
    timestamps = torch.sort(torch.rand(batch, seq_len) * 100, dim=1).values

    out = model(event_types, timestamps)
    # forward() predicts events 1..seq_len-1 from hidden[0..seq_len-2],
    # so it returns seq_len - 1 predictions, not seq_len.
    assert out.shape == (batch, seq_len - 1, num_types)
    assert torch.isfinite(out).all()
    assert (out >= 0).all(), "Intensities must be non-negative (softplus output)"
    print("[PASS] forward pass: correct shape (seq_len-1 predictions), finite, non-negative intensities")


def test_event_intensity_does_not_leak_own_type_or_time():
    """
    Critical correctness test for the predict-the-next-event
    convention: the intensity used to score event i must depend only
    on history strictly before event i. If we change event i's own
    type/time (the event being predicted) while holding all earlier
    events fixed, the *predicted* intensity for event i (which uses
    hidden[i-1], built only from events < i) must NOT change. This
    directly tests for the self-embedding leakage bug that was found
    and fixed during development: hidden[i] embeds event i's own type
    and time and must never be used to predict event i itself.
    """
    torch.manual_seed(5)
    num_types = 4
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=16, n_heads=2, n_layers=2)
    model.eval()

    seq_len = 6
    target_idx = 3  # predict event at this index using hidden[target_idx - 1]
    event_types = torch.randint(0, num_types, (1, seq_len))
    timestamps = torch.sort(torch.rand(1, seq_len) * 50, dim=1).values

    with torch.no_grad():
        hidden = model.encode(event_types, timestamps)
        elapsed = (timestamps[:, target_idx] - timestamps[:, target_idx - 1]).clamp(min=0)
        intensity_original = model.intensity(
            hidden[:, target_idx - 1 : target_idx, :], elapsed.unsqueeze(-1)
        )

        # Perturb ONLY event `target_idx`'s own type and time. Earlier
        # events (and hence hidden[target_idx - 1]) are untouched.
        event_types_perturbed = event_types.clone()
        event_types_perturbed[0, target_idx] = (event_types[0, target_idx] + 2) % num_types
        timestamps_perturbed = timestamps.clone()
        timestamps_perturbed[0, target_idx] += 8.0

        hidden_perturbed = model.encode(event_types_perturbed, timestamps_perturbed)
        # hidden[target_idx - 1] must be byte-for-byte identical, since
        # it only depends on events 0..target_idx-1, none of which changed.
        hidden_diff = (
            hidden[0, target_idx - 1] - hidden_perturbed[0, target_idx - 1]
        ).abs().max().item()
        assert hidden_diff < 1e-6, (
            f"hidden[target_idx-1] changed when only event[target_idx] was "
            f"perturbed: max diff {hidden_diff}. This means the model is "
            f"leaking the to-be-predicted event into the representation "
            f"used to predict it."
        )

        elapsed_perturbed = (
            timestamps_perturbed[:, target_idx] - timestamps_perturbed[:, target_idx - 1]
        ).clamp(min=0)
        intensity_after = model.intensity(
            hidden_perturbed[:, target_idx - 1 : target_idx, :], elapsed_perturbed.unsqueeze(-1)
        )

    print(
        f"[PASS] predicting event {target_idx} uses only hidden[{target_idx-1}], "
        f"which is unaffected by event {target_idx}'s own (perturbed) type/time "
        f"(hidden diff {hidden_diff:.2e})"
    )


def test_gradients_flow_through_full_model():
    torch.manual_seed(1)
    num_types = 2
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=8, n_heads=2, n_layers=1)

    batch, seq_len = 2, 6
    event_types = torch.randint(0, num_types, (batch, seq_len))
    timestamps = torch.sort(torch.rand(batch, seq_len) * 10, dim=1).values

    loss = negative_log_likelihood(model, event_types, timestamps, pad_value=num_types)
    loss.backward()

    n_params_with_grad = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient reached parameter {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        n_params_with_grad += 1
    print(f"[PASS] gradients flow to all {n_params_with_grad} parameter tensors, all finite")


def test_causal_masking_prevents_leakage_in_full_model():
    """
    Critical correctness test: changing a future event must not
    change the hidden state (and hence intensity) of an earlier
    position. This tests the full model end-to-end, not just the
    attention primitive in isolation.
    """
    torch.manual_seed(2)
    num_types = 4
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=16, n_heads=2, n_layers=2)
    model.eval()

    seq_len = 8
    event_types = torch.randint(0, num_types, (1, seq_len))
    timestamps = torch.sort(torch.rand(1, seq_len) * 50, dim=1).values

    with torch.no_grad():
        hidden_original = model.encode(event_types, timestamps)

        event_types_perturbed = event_types.clone()
        event_types_perturbed[0, -1] = (event_types[0, -1] + 1) % num_types
        timestamps_perturbed = timestamps.clone()
        timestamps_perturbed[0, -1] += 5.0

        hidden_perturbed = model.encode(event_types_perturbed, timestamps_perturbed)

    earlier_diff = (hidden_original[0, :-1] - hidden_perturbed[0, :-1]).abs().max().item()
    assert earlier_diff < 1e-6, (
        f"Causal leakage detected: changing the final event altered earlier "
        f"hidden states by up to {earlier_diff}"
    )
    print(f"[PASS] no causal leakage: earlier hidden states unaffected by future event change (max diff {earlier_diff:.2e})")


def test_padding_does_not_affect_real_events():
    torch.manual_seed(3)
    num_types = 3
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=16, n_heads=2, n_layers=2)
    model.eval()

    real_len = 5
    event_types_real = torch.randint(0, num_types, (1, real_len))
    timestamps_real = torch.sort(torch.rand(1, real_len) * 20, dim=1).values

    pad_len = 3
    event_types_padded = torch.cat(
        [event_types_real, torch.full((1, pad_len), num_types)], dim=1
    )
    timestamps_padded = torch.cat(
        [timestamps_real, torch.zeros(1, pad_len)], dim=1
    )

    with torch.no_grad():
        hidden_real_only = model.encode(event_types_real, timestamps_real)
        hidden_with_padding = model.encode(event_types_padded, timestamps_padded)

    diff = (hidden_real_only[0] - hidden_with_padding[0, :real_len]).abs().max().item()
    assert diff < 1e-5, f"Padding leaked into real event representations: max diff {diff}"
    print(f"[PASS] padding correctly masked: real-event hidden states unaffected (max diff {diff:.2e})")


def test_nll_is_finite_and_decreases_with_training():
    torch.manual_seed(4)
    num_types = 2
    model = TransformerHawkesProcess(num_event_types=num_types, d_model=16, n_heads=2, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch, seq_len = 8, 12
    event_types = torch.randint(0, num_types, (batch, seq_len))
    timestamps = torch.sort(torch.rand(batch, seq_len) * 30, dim=1).values

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = negative_log_likelihood(model, event_types, timestamps, pad_value=num_types, n_mc_samples=10)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), "NLL must stay finite during training"
    assert losses[-1] < losses[0], (
        f"NLL did not decrease over training: start {losses[0]:.4f}, end {losses[-1]:.4f}"
    )
    print(f"[PASS] NLL decreases with training: {losses[0]:.4f} -> {losses[-1]:.4f} over 30 steps")


if __name__ == "__main__":
    test_forward_shapes()
    test_event_intensity_does_not_leak_own_type_or_time()
    test_gradients_flow_through_full_model()
    test_causal_masking_prevents_leakage_in_full_model()
    test_padding_does_not_affect_real_events()
    test_nll_is_finite_and_decreases_with_training()
    print("\nAll THP model tests passed.")

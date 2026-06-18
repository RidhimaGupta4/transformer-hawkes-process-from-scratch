# `tests/`

23 tests across 4 files, each checking against a closed-form or known reference rather than just "the code runs without an exception."

- **`test_attention.py`** -- output matches `torch.nn.functional.scaled_dot_product_attention`, attention weights sum to 1, causal mask blocks all future positions.
- **`test_temporal_encoding.py`** -- shape and determinism checks for all three encodings, plus a check that `log1p` actually compresses large inter-event gaps the way it's meant to.
- **`test_hawkes_simulation.py`** -- empirical event rates from simulation matched against the closed-form theoretical stationary rate, for both the univariate and multivariate case. This is the test that originally caught a sign/orientation bug in the multivariate rate formula.
- **`test_thp_model.py`** -- shapes, gradient flow, and two dedicated leakage probes: one for causal masking in attention, one (`test_event_intensity_does_not_leak_own_type_or_time`) for the data-leakage bug described in the main README, which perturbs a single event and checks the hidden state used to predict it is provably unaffected.

Run all of them with `../run_tests.sh` from the repo root, or individually with `python3 test_<name>.py`.

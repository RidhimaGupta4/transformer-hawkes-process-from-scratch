# `src/`

Core implementation, no training or plotting logic here -- that lives in the notebook.

- **`attention.py`** -- multi-head self-attention from first principles (no `nn.MultiheadAttention`), plus the causal mask used to stop a position from attending to future events. Verified in `tests/test_attention.py` against PyTorch's own reference kernel.
- **`temporal_encoding.py`** -- three ways to encode irregular event timestamps: fixed sinusoidal encoding, learned Time2Vec, and a learned log-gap embedding.
- **`hawkes_simulation.py`** -- simulates a multivariate Hawkes process (Ogata's thinning algorithm) so the notebook has data with a *known* ground-truth intensity function to validate against.
- **`thp_model.py`** -- the full Transformer Hawkes Process: encoder built from the two modules above, the intensity function, and the Monte Carlo log-likelihood loss used for training.

Import these directly (`from attention import ...`) rather than as a package; the notebook adds this folder to `sys.path` at the top.

# `notebooks/`

- **`thp_implementation.ipynb`** -- the full walkthrough: attention sanity checks, temporal encoding comparisons, Hawkes simulation and ground-truth recovery, training the THP model, and an honest limitations section. Runs end-to-end on CPU in a few minutes; no GPU needed at this scale.

The notebook documents the data-leakage bug found during development (see the main README) at the point in the training section where it was actually caught, rather than only summarising it afterwards. Saved outputs reflect the most recent full execution -- if you re-run it yourself, exact figures should match closely but not necessarily bit-for-bit, since Monte Carlo sampling is involved in the loss.

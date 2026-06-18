"""
Temporal encoding strategies for irregularly-timed event sequences.

Standard transformer positional encoding assumes evenly-spaced
discrete positions (token 1, token 2, ...). Event sequences violate
this: the gap between event i and event i+1 can be a second or a
year, and that gap is informative. This module implements three
encodings that take the actual continuous timestamp into account,
not just the integer position in the sequence.

1. Sinusoidal positional encoding (Vaswani et al., 2017), applied to
   continuous time rather than integer position -- the encoding used
   in the original Transformer Hawkes Process (Zuo et al., 2020).
2. Time2Vec (Kazemi et al., 2019) -- a learned periodic + linear
   embedding of continuous time.
3. Log-normalised inter-event gap embedding -- a simple learned
   embedding of log(1 + delta_t), included because raw inter-event
   gaps in real event data are heavy-tailed (most gaps are short,
   a few are very long), and log-scaling is the standard way to
   stabilise this before feeding it to a neural network.
"""

import torch
import torch.nn as nn


class SinusoidalTimeEncoding(nn.Module):
    """
    Encodes a continuous timestamp t using fixed (non-learned)
    sinusoids of varying frequency, following the THP formulation:

        PE(t, 2i)   = sin(t / 10000^(2i/d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

    Unlike standard Transformer positional encoding (which encodes
    integer token position), this is evaluated at the actual event
    timestamp, so it naturally reflects irregular spacing.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Precompute the frequency denominators 10000^(2i/d_model)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        freqs = torch.exp(-i / d_model * torch.log(torch.tensor(10000.0)))
        self.register_buffer("freqs", freqs)  # shape (d_model // 2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestamps, shape (batch, seq_len)
        Returns:
            encoding, shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, 1) * (d_model // 2,) -> (batch, seq_len, d_model // 2)
        angles = t.unsqueeze(-1) * self.freqs

        encoding = torch.zeros(*t.shape, self.d_model, device=t.device)
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)
        return encoding


class Time2Vec(nn.Module):
    """
    Time2Vec (Kazemi et al., 2019): a learned time embedding with
    one linear component and (d_model - 1) learned periodic
    components:

        t2v(t)[0] = w_0 * t + b_0
        t2v(t)[i] = sin(w_i * t + b_i)   for i = 1, ..., d_model - 1

    Unlike sinusoidal encoding, the frequencies w_i and phases b_i
    are learned from data rather than fixed, allowing the model to
    discover whichever periodicities (daily, weekly, seasonal) are
    actually present in the event stream.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Parameter(torch.randn(d_model))
        self.b = nn.Parameter(torch.randn(d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: timestamps, shape (batch, seq_len)
        Returns:
            encoding, shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, 1) * (d_model,) + (d_model,) -> (batch, seq_len, d_model)
        linear_and_periodic = t.unsqueeze(-1) * self.w + self.b

        encoding = torch.empty_like(linear_and_periodic)
        encoding[..., 0] = linear_and_periodic[..., 0]  # linear (trend) term
        encoding[..., 1:] = torch.sin(linear_and_periodic[..., 1:])  # periodic terms
        return encoding


class LogGapEmbedding(nn.Module):
    """
    Embeds the inter-event gap delta_t = t_i - t_{i-1} using a
    log(1 + delta_t) transform followed by a learned linear
    projection to d_model.

    Real inter-event gaps (e.g. time between insurance claims, or
    between StackOverflow badges) are heavy-tailed: most gaps are
    short, but a small number are very long. Feeding raw gaps
    directly into a linear layer lets a handful of extreme values
    dominate the gradient. log(1 + delta_t) compresses the tail
    while leaving short gaps close to their original scale, which is
    why it is the standard transform used for this kind of duration
    data in survival analysis and point process models.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(1, d_model)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: non-negative inter-event gaps, shape (batch, seq_len)
        Returns:
            encoding, shape (batch, seq_len, d_model)
        """
        log_gap = torch.log1p(delta_t).unsqueeze(-1)  # (batch, seq_len, 1)
        return self.projection(log_gap)

"""
Transformer Hawkes Process (THP) encoder and intensity function,
following Zuo et al. (2020), built on the from-scratch attention
(src/attention.py) and temporal encoding (src/temporal_encoding.py)
modules in this repository.

Architecture:
    1. Each event (t_i, k_i) is embedded as:
           h_i_in = Embedding(k_i) + SinusoidalTimeEncoding(t_i)
    2. A stack of causally-masked self-attention + feed-forward
       layers produces contextual hidden states h_i.
    3. The conditional intensity for event type u just after event i
       is parameterised as a softplus of a linear readout of h_i:
           lambda_u(t) = softplus(w_u^T h_i + w_u,t * (t - t_i))
       which follows the THP paper's formulation: an affine function
       of elapsed time since the last event, passed through softplus
       to guarantee positivity (a valid intensity must be >= 0).
    4. The model is trained by maximising the point process
       log-likelihood, which rewards high intensity at observed
       event times and penalises high intensity at times when
       nothing happened (the integral term, approximated by Monte
       Carlo integration since it has no closed form for a general
       neural intensity).

Reference:
    Zuo, S., Jiang, H., Li, Z., Zhao, T., Zha, H. (2020).
    "Transformer Hawkes Process." ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadSelfAttention, causal_mask
from temporal_encoding import SinusoidalTimeEncoding


class FeedForward(nn.Module):
    """Position-wise feed-forward block, standard Transformer component."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class THPEncoderLayer(nn.Module):
    """One Transformer encoder layer: causal self-attention + FFN,
    each wrapped in a residual connection and layer normalisation."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerHawkesProcess(nn.Module):
    """
    Full THP model: event-type embedding + sinusoidal time encoding,
    a stack of causal THPEncoderLayers, and a per-type intensity head.
    """

    def __init__(
        self,
        num_event_types: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_event_types = num_event_types
        self.d_model = d_model

        # Reserve index `num_event_types` as the PAD token.
        self.event_embedding = nn.Embedding(
            num_event_types + 1, d_model, padding_idx=num_event_types
        )
        self.time_encoding = SinusoidalTimeEncoding(d_model)

        self.layers = nn.ModuleList(
            [THPEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Intensity head: w_u^T h_i term, plus a learned per-type
        # time-decay coefficient w_u,t for the elapsed-time term.
        self.intensity_weight = nn.Linear(d_model, num_event_types)
        self.intensity_time_coef = nn.Parameter(torch.ones(num_event_types) * 0.1)

    def encode(self, event_types: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_types: (batch, seq_len) integer event types,
                         padded with `num_event_types`.
            timestamps:  (batch, seq_len) event times.
        Returns:
            hidden states, shape (batch, seq_len, d_model)
        """
        seq_len = event_types.size(1)
        x = self.event_embedding(event_types) + self.time_encoding(timestamps)

        causal = causal_mask(seq_len, device=event_types.device)
        # Also block attention to PAD positions as keys.
        pad_mask = (event_types == self.num_event_types).unsqueeze(1).unsqueeze(2)
        full_mask = causal | pad_mask

        for layer in self.layers:
            x = layer(x, full_mask)
        return x

    def intensity(self, hidden: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        """
        Compute lambda_u(t) for every event type u, at `elapsed` time
        after the event whose hidden state is `hidden`.

        Args:
            hidden:  (batch, seq_len, d_model)
            elapsed: (batch, seq_len) non-negative elapsed time since
                     the corresponding event.
        Returns:
            intensities, shape (batch, seq_len, num_event_types)
        """
        base = self.intensity_weight(hidden)  # (batch, seq_len, K)
        time_term = elapsed.unsqueeze(-1) * self.intensity_time_coef  # (batch, seq_len, K)
        return F.softplus(base + time_term)

    def forward(self, event_types: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode the sequence, then return predicted intensities for
        each event i (i = 1, ..., seq_len-1) using only hidden[i-1]
        (history strictly before event i), evaluated at the actual
        elapsed time since event i-1. Position 0 has no predecessor
        and is not included in the output.

        Returns:
            intensities, shape (batch, seq_len - 1, num_event_types)
        """
        hidden = self.encode(event_types, timestamps)
        hidden_prev = hidden[:, :-1, :]
        elapsed = (timestamps[:, 1:] - timestamps[:, :-1]).clamp(min=0)
        return self.intensity(hidden_prev, elapsed)


def negative_log_likelihood(
    model: TransformerHawkesProcess,
    event_types: torch.Tensor,
    timestamps: torch.Tensor,
    pad_value: int,
    n_mc_samples: int = 20,
) -> torch.Tensor:
    """
    Negative log-likelihood of a batch of event sequences under the
    model, following the standard point-process decomposition:

        log L = sum_i log(lambda_{k_i}(t_i)) - integral_0^T lambda(s) ds

    The event term rewards high intensity for the type that actually
    occurred. The integral term penalises high total intensity at
    times when nothing happened -- without it the model could
    trivially maximise the event term alone by predicting unbounded
    intensity everywhere. The integral has no closed form for a
    neural intensity, so it is estimated via Monte Carlo: sample
    `n_mc_samples` random times within each inter-event interval and
    average the predicted total intensity there, scaled by the
    interval length.

    IMPORTANT (predict-the-next-event convention): the intensity used
    to evaluate event i must be built ONLY from history strictly
    before event i, i.e. from hidden[i-1], not hidden[i]. This is
    because `model.encode` embeds each position's own event type and
    time into that same position's input (x[i] = Embedding(type_i) +
    TimeEncoding(t_i)) before attention runs; even with a causal mask
    restricting *attention*, hidden[i] still directly encodes event
    i's own type and time through this input embedding and the
    residual connections around attention. Using hidden[i] to predict
    event i's own type is therefore a data leakage bug: the model can
    partly decode the answer from its own input rather than predicting
    it from preceding history. Shifting by one position -- using
    hidden[i-1] to predict event i, and to integrate over the
    interval (t_{i-1}, t_i] -- removes this leakage. The first event
    in each sequence has no preceding hidden state and is therefore
    excluded from the loss (it carries no information about the
    model's predictive ability, only about the base rate mu).

    Args:
        model: TransformerHawkesProcess.
        event_types: (batch, seq_len), padded with `pad_value`.
        timestamps:  (batch, seq_len).
        pad_value: padding index used in event_types.
        n_mc_samples: Monte Carlo samples per interval.

    Returns:
        scalar negative log-likelihood, averaged over the batch and
        normalised per valid, predicted (i.e. non-first, non-padded)
        event.
    """
    hidden = model.encode(event_types, timestamps)  # (batch, seq_len, d_model)
    batch, seq_len = timestamps.shape

    # Shift by one: to predict event i (i = 1, ..., seq_len-1), use
    # the hidden state of event i-1 (history strictly before event i).
    # hidden_prev[i] = hidden[i-1] for i >= 1; position 0 has no
    # predecessor and is excluded via valid_mask below.
    hidden_prev = hidden[:, :-1, :]  # (batch, seq_len-1, d_model) -- hidden[0..seq_len-2]
    target_types = event_types[:, 1:]  # event i for i = 1..seq_len-1
    target_times = timestamps[:, 1:]
    prev_times = timestamps[:, :-1]

    valid_mask = (target_types != pad_value).float()  # (batch, seq_len-1)

    # --- Event term: log lambda_{k_i}(t_i), using ONLY hidden_prev
    # (i.e. history strictly before event i) ---
    elapsed_at_event = (target_times - prev_times).clamp(min=0)
    intensities = model.intensity(hidden_prev, elapsed_at_event)  # (batch, seq_len-1, K)

    safe_types = target_types.clamp(max=model.num_event_types - 1)
    event_intensity = torch.gather(
        intensities, dim=2, index=safe_types.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)

    log_event_term = torch.log(event_intensity + 1e-9) * valid_mask
    event_term = log_event_term.sum(dim=1)  # (batch,)

    # --- Non-event term: Monte Carlo estimate of integral of total
    # intensity over each inter-event interval (t_{i-1}, t_i], using
    # hidden_prev throughout (the intensity over this whole interval
    # is governed by history up to and including event i-1 only) ---
    gaps = elapsed_at_event * valid_mask  # (batch, seq_len-1)

    integral_estimate = torch.zeros(batch, seq_len - 1, device=timestamps.device)
    for _ in range(n_mc_samples):
        u = torch.rand_like(gaps)
        sample_elapsed = u * gaps
        sample_intensities = model.intensity(hidden_prev, sample_elapsed)  # (batch, seq_len-1, K)
        total_intensity_sample = sample_intensities.sum(dim=-1)  # (batch, seq_len-1)
        integral_estimate += total_intensity_sample * gaps * valid_mask

    integral_estimate = integral_estimate / n_mc_samples
    non_event_term = integral_estimate.sum(dim=1)  # (batch,)

    log_likelihood = event_term - non_event_term  # (batch,)
    n_valid = valid_mask.sum(dim=1).clamp(min=1)
    per_event_nll = -log_likelihood / n_valid

    return per_event_nll.mean()

"""
Multi-head self-attention implemented from first principles.

This module implements the scaled dot-product attention mechanism
(Vaswani et al., 2017) without using torch.nn.MultiheadAttention,
to make every step of the computation explicit and verifiable.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Reference:
    Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
"""

import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        q: Queries,  shape (batch, heads, seq_len_q, d_k)
        k: Keys,     shape (batch, heads, seq_len_k, d_k)
        v: Values,   shape (batch, heads, seq_len_k, d_v)
        mask: Optional boolean mask, shape broadcastable to
              (batch, heads, seq_len_q, seq_len_k).
              Positions where mask == True are disallowed
              (set to -inf before softmax). Used here for
              causal masking, since event-sequence models must
              not attend to future events.

    Returns:
        output:  shape (batch, heads, seq_len_q, d_v)
        weights: attention weights, shape (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)

    # QK^T / sqrt(d_k)
    # (batch, heads, seq_len_q, d_k) @ (batch, heads, d_k, seq_len_k)
    #   -> (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Large negative value rather than -inf to avoid NaNs from
        # softmax(-inf - (-inf)) when an entire row is masked.
        scores = scores.masked_fill(mask, float("-1e9"))

    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention block built directly from the
    scaled_dot_product_attention primitive above.

    Splits the model dimension d_model into n_heads heads of
    dimension d_k = d_model / n_heads, applies attention
    independently per head, then concatenates and projects back
    to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Single linear layers projecting to all heads at once;
        # equivalent to a separate W_q, W_k, W_v per head but more
        # efficient as one matrix multiply.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, n_heads * d_k)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (batch, seq_len, d_model)
            mask: optional, shape broadcastable to
                  (batch, 1, seq_len, seq_len)

        Returns:
            output:  shape (batch, seq_len, d_model)
            weights: shape (batch, n_heads, seq_len, seq_len)
        """
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        attn_output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = self.dropout(attn_output)

        merged = self._merge_heads(attn_output)
        output = self.w_o(merged)
        return output, weights


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Build a causal (look-ahead) mask so that position i cannot attend
    to position j > i. Required for event-sequence modelling: the
    representation of an event must only depend on past events, not
    future ones, otherwise the model leaks future information into
    the present and produces an invalid (non-causal) point process.

    Returns:
        Boolean tensor of shape (1, 1, seq_len, seq_len), True where
        attention is disallowed.
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)

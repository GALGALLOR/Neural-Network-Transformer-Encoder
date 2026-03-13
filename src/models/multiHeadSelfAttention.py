import torch
import torch.nn as nn

from .scaledDotProductAttention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.split_heads(self.W_q(x))
        key = self.split_heads(self.W_k(x))
        value = self.split_heads(self.W_v(x))

        attention_output, attention_weights = self.attention(query, key, value, mask)
        attention_output = self.combine_heads(attention_output)
        output = self.dropout(self.out_proj(attention_output))
        return output, attention_weights

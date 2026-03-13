import torch
import torch.nn as nn

from .encoderBlock import EncoderBlock
from .positionalEncoding import PositionalEncoding


def create_padding_mask(input_ids: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (input_ids != pad_idx).unsqueeze(1).unsqueeze(2)


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)

        all_attention_weights: list[torch.Tensor] = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask) 
            all_attention_weights.append(attention_weights) 

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            sequence_mask = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
            masked_x = x * sequence_mask
            pooled = masked_x.sum(dim=1) / sequence_mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        return logits, all_attention_weights

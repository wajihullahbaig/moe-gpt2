import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from config import CONTEXT_LENGTH, DROPOUT, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS
from models.base import Expert


class CausalTransformerBlock(nn.Module):
    """Post-LN causal self-attention block."""

    def __init__(self, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """Masked self-attention then feed-forward, each with a residual and LayerNorm."""
        mask = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).triu(1)
        attended, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False, is_causal=True)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feed_forward(x))


class TransformerExpert(Expert):
    """Learned positions followed by NUM_LAYERS causal transformer blocks."""

    def __init__(self, d_model=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.positions = nn.Embedding(CONTEXT_LENGTH, d_model)
        self.blocks = nn.Sequential(*(CausalTransformerBlock(d_model) for _ in range(num_layers)))

    def forward(self, x):
        """Add positions and run the block stack."""
        return self.blocks(x + self.positions(torch.arange(x.size(1), device=x.device)))


class GPT2Expert(Expert):
    """A small Hugging Face GPT-2 stack (own positions, causal attention) fed with shared embeddings."""

    def __init__(self, d_model=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        # vocab_size=1: the expert receives embeddings, so its own token table would be dead weight
        config = GPT2Config(vocab_size=1, n_positions=CONTEXT_LENGTH, n_embd=d_model,
                            n_layer=num_layers, n_head=NUM_HEADS)
        self.gpt = GPT2Model(config)

    def forward(self, x):
        """Final GPT-2 hidden states for the given embeddings."""
        return self.gpt(inputs_embeds=x).last_hidden_state

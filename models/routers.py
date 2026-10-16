import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (BALANCE_LOSS_WEIGHT, DROPOUT, GUIDED_LEARNED_SHARE, GUIDED_ROUTE_TEMPERATURE, HIDDEN_DIM,
                    LEARNED_ROUTE_TEMPERATURE, ROUTING_LOSS_WEIGHT, VOCAB_SIZE)
from models.base import Router, Routing


def vocab_partition(num_experts):
    """Map every token id to an expert by splitting the vocabulary into contiguous ranges."""
    return torch.arange(VOCAB_SIZE) * num_experts // VOCAB_SIZE


def frequency_partition(token_counts, num_experts):
    """Map tokens, sorted by training frequency, to experts in bands carrying equal shares of the token mass."""
    order = token_counts.argsort(descending=True)
    mass = token_counts[order].double().cumsum(0) / token_counts.sum()
    partition = torch.empty(VOCAB_SIZE, dtype=torch.long)
    partition[order] = ((mass - 1e-9) * num_experts).long().clamp(0, num_experts - 1)
    return partition


class LearnedRouter(Router):
    """MLP over the causal prefix mean of embeddings, softened by a temperature; pushed towards balanced use."""

    def __init__(self, num_experts, temperature=LEARNED_ROUTE_TEMPERATURE):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_experts),
        )
        self.register_buffer('temperature', torch.tensor(temperature))

    def log_probs(self, x):
        """Learned routing log-probabilities [B, T, E]; position t only sees tokens <= t."""
        steps = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
        return F.log_softmax(self.mlp(x.cumsum(dim=1) / steps) / self.temperature, dim=-1)

    def forward(self, x, ids):
        """Learned weights plus the balance loss."""
        weights = self.log_probs(x).exp()
        usage = weights.mean(dim=(0, 1))
        balance = F.mse_loss(usage, torch.full_like(usage, 1 / usage.numel()))
        return Routing(weights, {'balance': BALANCE_LOSS_WEIGHT * balance})


class PartitionGuidedRouter(LearnedRouter):
    """Learned router blended with a fixed token→expert partition and trained to agree with it."""

    def __init__(self, num_experts, token_to_expert, persistent=False, temperature=GUIDED_ROUTE_TEMPERATURE,
                 learned_share=GUIDED_LEARNED_SHARE):
        super().__init__(num_experts, temperature)
        self.learned_share = learned_share
        self.register_buffer('token_to_expert', token_to_expert, persistent=persistent)

    def forward(self, x, ids):
        """Blend of learned and partition-assigned weights plus the agreement loss."""
        log_probs = self.log_probs(x)
        assignment = self.token_to_expert[ids]
        assigned = F.one_hot(assignment, log_probs.size(-1)).float()
        weights = self.learned_share * log_probs.exp() + (1 - self.learned_share) * assigned
        agreement = F.nll_loss(log_probs.flatten(0, 1), assignment.flatten())
        return Routing(weights, {'routing': ROUTING_LOSS_WEIGHT * agreement})

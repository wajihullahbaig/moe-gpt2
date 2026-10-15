import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DIVERSITY_LOSS_WEIGHT, HIDDEN_DIM, VOCAB_SIZE
from models.base import Router, Routing


def expert_diversity_loss(expert_outputs):
    """Mean absolute cosine similarity between every pair of experts' outputs ([B, E, T, D])."""
    flat = F.normalize(expert_outputs.flatten(2), dim=-1)
    similarity = flat @ flat.transpose(1, 2)
    off_diagonal = ~torch.eye(similarity.size(1), dtype=torch.bool, device=similarity.device)
    return similarity[:, off_diagonal].abs().mean()


class MixtureOfExpertsLM(nn.Module):
    """Shared token embedding → every expert → per-position router mix → vocabulary logits."""

    def __init__(self, name, expert_cls, router: Router, num_experts):
        super().__init__()
        self.name = name
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.experts = nn.ModuleList(expert_cls() for _ in range(num_experts))
        self.router = router
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def route(self, ids) -> Routing:
        """Routing for a batch of token ids, as used in forward."""
        return self.router(self.embedding(ids), ids)

    def forward(self, ids, return_losses=False):
        """Next-token logits [B, T, V], plus weighted auxiliary losses when return_losses is set."""
        x = self.embedding(ids)
        routing = self.router(x, ids)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, T, D]
        logits = self.output(torch.einsum('bte,betd->btd', routing.weights, expert_outputs))
        if not return_losses:
            return logits
        return logits, {'diversity': DIVERSITY_LOSS_WEIGHT * expert_diversity_loss(expert_outputs), **routing.losses}

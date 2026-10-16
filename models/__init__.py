import torch

from config import NUM_EXPERTS, VOCAB_SIZE
from models.experts import GPT2Expert, TransformerExpert
from models.moe import MixtureOfExpertsLM
from models.routers import LearnedRouter, PartitionGuidedRouter, frequency_partition, vocab_partition

# Register new architectures or routing strategies here; the CLI picks them up automatically.
# Router factories take (num_experts, training token counts [V]).
EXPERTS = {'transformer': TransformerExpert, 'gpt2': GPT2Expert}
ROUTERS = {
    'learned': lambda experts, counts: LearnedRouter(experts),
    'guided': lambda experts, counts: PartitionGuidedRouter(experts, vocab_partition(experts)),
    # persistent: the data-derived partition is saved with the weights, so inference needs no data
    'freq': lambda experts, counts: PartitionGuidedRouter(experts, frequency_partition(counts, experts),
                                                          persistent=True),
}


def build_model(arch, routing, token_counts=None, num_experts=NUM_EXPERTS):
    """Build the MoE LM for a registered expert architecture and routing strategy.

    token_counts are training-token frequencies; without them (e.g. at inference) data-derived partitions start
    as placeholders and are restored from the checkpoint.
    """
    counts = torch.ones(VOCAB_SIZE) if token_counts is None else token_counts
    router = ROUTERS[routing](num_experts, counts)
    return MixtureOfExpertsLM(f'{arch}-{routing}', EXPERTS[arch], router, num_experts)

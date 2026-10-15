from config import NUM_EXPERTS
from models.experts import GPT2Expert, TransformerExpert
from models.moe import MixtureOfExpertsLM
from models.routers import LearnedRouter, VocabGuidedRouter

# Register new architectures or routing strategies here; the CLI picks them up automatically.
EXPERTS = {'transformer': TransformerExpert, 'gpt2': GPT2Expert}
ROUTERS = {'learned': LearnedRouter, 'guided': VocabGuidedRouter}


def build_model(arch, routing, num_experts=NUM_EXPERTS):
    """Build the MoE language model for a registered expert architecture and routing strategy."""
    return MixtureOfExpertsLM(f'{arch}-{routing}', EXPERTS[arch], ROUTERS[routing](num_experts), num_experts)

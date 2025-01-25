import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from datetime import datetime

from constants import CONTEXT_LENGTH, NUM_EXPERTS, STRIDE, VOCAB_SIZE


def set_seed(seed: Optional[int] = 42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def visualize_expert_usage(model, val_loader, device, epoch, save_path='./plots'):
    from main_transformer_arch import GuidedMoETransformer
    from main_gpt2_arch import GuidedGPT2MoE
    """Visualize expert usage patterns"""
    model.eval()
    expert_usage = []
    token_expert_map = np.zeros((VOCAB_SIZE, NUM_EXPERTS))
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            
            # Get expert weights based on model type
            if isinstance(model, GuidedMoETransformer) or isinstance(model, GuidedGPT2MoE):
                # For guided model, get both learned and token-based weights
                x = model.embedding(inputs)
                # Get the correct slice of positional encoding
                seq_length = x.size(1)
                pos_enc = model.pos_encoding[:, :seq_length, :]
                x = x + pos_enc
                
                learned_weights = F.softmax(model.router(x.mean(dim=1)), dim=-1)
                token_weights = model.compute_token_expert_weights(inputs)
                routing_weights = 0.5 * learned_weights + 0.5 * token_weights
            else:
                # For unguided model, get routing weights
                x = model.embedding(inputs)
                # Get the correct slice of positional encoding
                seq_length = x.size(1)
                pos_enc = model.pos_encoding[:, :seq_length, :]
                x = x + pos_enc
                
                avg_hidden = x.mean(dim=1)
                routing_logits = model.router(avg_hidden)
                routing_weights = F.softmax(routing_logits, dim=-1)
            
            expert_usage.append(routing_weights.cpu().numpy())
            
            # Track token-to-expert mapping
            for batch_idx in range(inputs.size(0)):
                for token in inputs[batch_idx]:
                    token_expert_map[token.item()] += routing_weights[batch_idx].cpu().numpy()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Overall Expert Usage
    plt.subplot(2, 2, 1)
    avg_usage = np.mean(np.concatenate(expert_usage, axis=0), axis=0)
    plt.bar(range(NUM_EXPERTS), avg_usage)
    plt.title('Overall Expert Utilization')
    plt.xlabel('Expert ID')
    plt.ylabel('Average Usage')
    
    # 2. Expert Usage Distribution
    plt.subplot(2, 2, 2)
    usage_dist = np.concatenate(expert_usage, axis=0)
    sns.boxplot(data=usage_dist)
    plt.title('Expert Usage Distribution')
    plt.xlabel('Expert ID')
    plt.ylabel('Usage')
    
    # 3. Token-Expert Heat Map (Top 100 most common tokens)
    plt.subplot(2, 2, 3)
    top_tokens = 100
    token_usage_sum = token_expert_map.sum(axis=1)
    top_token_indices = np.argsort(token_usage_sum)[-top_tokens:]
    sns.heatmap(token_expert_map[top_token_indices], 
                cmap='YlOrRd',
                xticklabels=[f'E{i}' for i in range(NUM_EXPERTS)],
                yticklabels=False)
    plt.title(f'Top {top_tokens} Tokens Expert Assignment')
    plt.xlabel('Expert ID')
    plt.ylabel('Token ID')
    
    # 4. Expert Specialization Score
    plt.subplot(2, 2, 4)
    specialization = np.zeros(NUM_EXPERTS)
    for expert_idx in range(NUM_EXPERTS):
        if isinstance(model, GuidedMoETransformer) or isinstance(model,GuidedGPT2MoE):
            # For guided model, compare with assigned vocabulary ranges
            assigned_tokens = model.expert_assignments[expert_idx]
            specialization[expert_idx] = np.mean(token_expert_map[assigned_tokens, expert_idx])
        else:
            # For unguided model, measure concentration of token assignments
            token_probs = token_expert_map[:, expert_idx] / (token_expert_map.sum(axis=1) + 1e-8)
            specialization[expert_idx] = np.mean(token_probs > 0.5)
    
    plt.bar(range(NUM_EXPERTS), specialization)
    plt.title('Expert Specialization Score')
    plt.xlabel('Expert ID')
    plt.ylabel('Specialization')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(model, GuidedMoETransformer) or isinstance(model,GuidedGPT2MoE):
        model_type = "guided"
    else:
        model_type = "unguided"
    save_file = os.path.join(save_path, f'expert_viz_{model_type}_epoch{epoch}_{timestamp}.png')
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        'expert_usage': avg_usage,
        'specialization': specialization
    }


def calculate_diversity_loss(expert_outputs, method='cosine_abs', eps=1e-8):
        """
        Calculate diversity loss between experts.
        
        Args:
        - expert_outputs: Tensor of shape (batch_size, num_experts, ...)
        - method: String specifying the method to use. Options are:
                'cosine', 'squared_diff', 'kl_div', 'cosine_abs', 'cosine_relu', 'cosine_squared'
        - eps: Small value to avoid division by zero
        
        Returns:
        - diversity_loss: Scalar tensor representing the diversity loss
        """
        batch_size = expert_outputs.size(0)
        num_experts = expert_outputs.size(1)
        expert_outputs_flat = expert_outputs.view(batch_size, num_experts, -1)
        
        if method == 'cosine':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        
        elif method == 'squared_diff':
            differences = (expert_outputs_flat.unsqueeze(2) - expert_outputs_flat.unsqueeze(1)).pow(2).sum(-1)
            diversity_loss = -torch.mean(torch.triu(differences.mean(0), diagonal=1))
        
        elif method == 'kl_div':
            expert_probs = F.softmax(expert_outputs_flat, dim=-1)
            kl_div = F.kl_div(expert_probs.log().unsqueeze(1), expert_probs.unsqueeze(2), reduction='none').sum(-1)
            diversity_loss = -torch.mean(torch.triu(kl_div.mean(0), diagonal=1))
        
        elif method == 'cosine_abs':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.abs(torch.mean(torch.triu(similarities.mean(0), diagonal=1)))
        
        elif method == 'cosine_relu':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.relu(torch.mean(torch.triu(similarities.mean(0), diagonal=1)))
        
        elif method == 'cosine_squared':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = (similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)).pow(2)
            diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return diversity_loss



def create_expert_assignments():
    """Create token-type to expert assignments"""
    vocab_per_expert = VOCAB_SIZE // NUM_EXPERTS
    assignments = {}
    for expert_idx in range(NUM_EXPERTS):
        start_idx = expert_idx * vocab_per_expert
        end_idx = start_idx + vocab_per_expert if expert_idx < NUM_EXPERTS-1 else VOCAB_SIZE
        assignments[expert_idx] = list(range(start_idx, end_idx))
    return assignments
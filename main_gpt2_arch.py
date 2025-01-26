import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import GPT2Config, GPT2Model

from common import calculate_diversity_loss, create_expert_assignments, set_seed
from constants import BALANCE_LOSS_WEIGHTAGE, BATCH_SIZE, CONTEXT_LENGTH, DATA_FRACTION, DIVERSITY_LOSS_WEIGHTAGE, LEARNED_WEIGHTS_WEIGHTAGE, ROUTING_LOSS_WEIGHTAGE, TOKEN_ASSIGNMENT_WEIGHTAGE, TOTAL_EPOCHS, VOCAB_SIZE, NUM_EXPERTS, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS
from dataset_loading import load_data
from train_val import count_parameters, test_generation, train_model


class GPT2Expert(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = GPT2Config(
                vocab_size=VOCAB_SIZE,
                n_positions=CONTEXT_LENGTH,
                n_embd=HIDDEN_DIM,
                n_layer=NUM_LAYERS,
                n_head=NUM_HEADS,
                n_inner=None,
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                bos_token_id=50256,
                eos_token_id=50256,
            )
        self.gpt = GPT2Model(config)
        
    def forward(self, x):
        # GPT2 expects input_ids, but we're passing embeddings
        # We need to get the hidden states from the GPT2 model
        outputs = self.gpt(inputs_embeds=x)
        return outputs.last_hidden_state

class UnGuidedGPT2MoE(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.3,route_temp = 2.0):
        super().__init__()
        # Initialize GPT2 embedding layer
        config = GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=CONTEXT_LENGTH,
            n_embd=HIDDEN_DIM
        )
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.route_temp = nn.Parameter(torch.ones(1)*route_temp)
        # Position encoding (from GPT2)
        position = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
        div_term = torch.exp(torch.arange(0, HIDDEN_DIM, 2).float() * (-math.log(10000.0) / HIDDEN_DIM))
        
        pe = torch.zeros(1, CONTEXT_LENGTH, HIDDEN_DIM)
        pe[0, :, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[0, :, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        
        self.register_buffer('pos_encoding', pe)
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, num_experts)
        )
        
        # Initialize experts as GPT2 models
        self.experts = nn.ModuleList([
            GPT2Expert() for _ in range(num_experts)
        ])
        
        # Output layer
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)


        
    def forward(self, x, return_losses=False):
        # Embedding
        x = self.embedding(x)
        
        # Calculate routing weights
        avg_hidden = x.mean(dim=1)
        routing_logits = self.router(avg_hidden)
        routing_weights = F.softmax(routing_logits/self.route_temp, dim=-1)
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Reshape routing weights for multiplication
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Combine expert outputs
        combined = torch.sum(expert_outputs * routing_weights, dim=1)
        
        # Generate output logits
        logits = self.output(combined)
        
        if return_losses and self.training:
            # Load balancing loss
            balance_loss = F.mse_loss(
                routing_weights.mean(0),
                torch.ones_like(routing_weights.mean(0)) / NUM_EXPERTS
            )
            diversity_loss = calculate_diversity_loss(expert_outputs)
            return logits, {
                    'balance_loss': balance_loss * BALANCE_LOSS_WEIGHTAGE,
                    "diversity_loss":diversity_loss * DIVERSITY_LOSS_WEIGHTAGE
                                }
            
        return logits

class GuidedGPT2MoE(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.3,route_temp=1.2):
        super().__init__()
        # Initialize GPT2 embedding layer
        config = GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=CONTEXT_LENGTH,
            n_embd=HIDDEN_DIM
        )
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.route_temp = nn.Parameter(torch.ones(1)*route_temp)
        # Expert assignments based on token types
        self.expert_assignments = create_expert_assignments()
        
        # Position encoding (from GPT2)
        self.register_buffer(
            "pos_encoding",
            torch.arange(0, CONTEXT_LENGTH).unsqueeze(0).unsqueeze(-1).expand(1, -1, HIDDEN_DIM)
        )
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, num_experts)
        )
        
        # Initialize experts as GPT2 models
        self.experts = nn.ModuleList([
            GPT2Expert() for _ in range(num_experts)
        ])
        
        # Output layer
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)


    def compute_token_expert_weights(self, tokens):
        """Compute expert weights based on token assignments"""
        batch_size = tokens.size(0)
        weights = torch.zeros(batch_size, len(self.expert_assignments), device=tokens.device)
        
        for expert_idx, vocab_range in self.expert_assignments.items():
            # Create tensor of assigned vocab indices for this expert
            vocab_indices = torch.tensor(vocab_range, device=tokens.device)
            
            # Check if each token in the sequence belongs to this expert's range
            # mask shape will be [batch_size]
            mask = torch.any(
                torch.isin(
                    tokens.view(batch_size, -1),  # flatten sequence dimension
                    vocab_indices
                ),
                dim=1  # check across sequence length
            )
            
            # Expand mask to match weights shape
            weights[:, expert_idx] = mask.float()
        
        # Normalize weights
        row_sums = weights.sum(dim=1, keepdim=True)
        weights = weights / (row_sums + 1e-8)

        return weights   
            
    def forward(self, x, return_losses=False):
        # Embedding
        x = self.embedding(x)
        
        # Calculate routing weights
        avg_hidden = x.mean(dim=1)
        routing_logits = self.router(avg_hidden)
        learned_weights = F.softmax(routing_logits/self.route_temp, dim=-1)
        token_weights = self.compute_token_expert_weights(x)
        
        # Combine learned and token-based weights
        routing_weights = LEARNED_WEIGHTS_WEIGHTAGE * learned_weights + TOKEN_ASSIGNMENT_WEIGHTAGE * token_weights
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Reshape routing weights for multiplication
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Combine expert outputs
        combined = torch.sum(expert_outputs * routing_weights, dim=1)
        
        # Generate output logits
        logits = self.output(combined)
        
        if return_losses and self.training:
            routing_loss = F.kl_div(
                learned_weights.log(), token_weights, reduction='batchmean'
            )
            diversity_loss = calculate_diversity_loss(expert_outputs)
            
            return logits, {'routing_loss': routing_loss  * ROUTING_LOSS_WEIGHTAGE, 
                            'diversity_loss':diversity_loss * DIVERSITY_LOSS_WEIGHTAGE
                            }
            
        return logits
    
def main():
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    viz_save_path = "./plots_gpt2"        
    # Load data with specified fraction (e.g., 0.1 for 10% of data)        
    train_loader, val_loader, test_loader = load_data(
        batch_size=BATCH_SIZE,
        data_fraction=DATA_FRACTION
    )
    
    # Train unguided model
    print("Training Unguided MoE GPT2...")
    unguided_model = UnGuidedGPT2MoE().to(device)
    print(f"\nTotal trainable parameters in UnGuided GPT2 MoE: {count_parameters(unguided_model):,}")
    train_model(unguided_model, train_loader, val_loader,num_epochs=TOTAL_EPOCHS,viz_path=viz_save_path)
    
    # Train guided model
    print("\nTraining Guided MoE GPT2...")
    guided_model = GuidedGPT2MoE().to(device)
    print(f"\nTotal trainable parameters in Guided GPT 2 MoE: {count_parameters(guided_model):,}")
    train_model(guided_model, train_loader, val_loader,num_epochs=TOTAL_EPOCHS,viz_path=viz_save_path)

    print("\nTesting Unguided MoE GPT2 Generation:")
    test_generation(unguided_model)
    
    print("\nTesting Guided MoE GPT2 Generation:")
    test_generation(guided_model)
    
if __name__ == '__main__':
    main()
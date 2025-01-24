import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from common import calculate_diversity_loss, create_expert_assignments, set_seed
from constants import BATCH_SIZE, CONTEXT_LENGTH, DATA_FRACTION, TOTAL_EPOCHS, VOCAB_SIZE, NUM_EXPERTS, HIDDEN_DIM, NUM_HEADS
from dataset_loading import load_data
from train_val import count_parameters, test_generation, train_model


class TransformerExpert(nn.Module):
    def __init__(self, d_model=HIDDEN_DIM, nhead=NUM_HEADS,dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

class UnGuidedMoETransformer(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.3,route_temp = 1.2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.route_temp = nn.Parameter(torch.ones(1)*route_temp)
        if not hasattr(self, 'pos_encoding'):
            pos_encoding = torch.randn(1, CONTEXT_LENGTH, HIDDEN_DIM)
            self.register_buffer('pos_encoding', pos_encoding)
        # Router network
        self.router = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(HIDDEN_DIM, num_experts)
        )        
        
        # Output layer
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    # Experts
        self.experts = nn.ModuleList([
            TransformerExpert() for _ in range(num_experts)
        ])

    def get_pos_encoding_slice(self, seq_length):
        """Get the appropriate slice of positional encoding"""
        return self.pos_encoding[:, :seq_length, :]                

    
    def forward(self, x, return_losses=False):
        # x shape: [batch_size, seq_length]
        
        # Embedding + positional encoding
        x = self.embedding(x)  # [batch_size, seq_length, hidden_dim]
        pos_enc = self.get_pos_encoding_slice(x.size(1))
        x = x + pos_enc
        
        # Calculate routing weights
        avg_hidden = x.mean(dim=1)  # [batch_size, hidden_dim]
        routing_logits = self.router(avg_hidden)  # [batch_size, num_experts]
        routing_weights = F.softmax(routing_logits/self.route_temp, dim=-1)  # [batch_size, num_experts]
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, seq_length, hidden_dim]
        
        # Fix: Reshape routing weights to match expert_outputs dimensions
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_experts, 1, 1]
        
        # Combine expert outputs
        combined = torch.sum(expert_outputs * routing_weights, dim=1)  # [batch_size, seq_length, hidden_dim]
        
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
                    'balance_loss': balance_loss,
                    "diversity_loss":diversity_loss 
                                }
            
        return logits

class GuidedMoETransformer(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.3,route_temp=1.2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.route_temp = nn.Parameter(torch.ones(1)*route_temp)
        if not hasattr(self, 'pos_encoding'):
            pos_encoding = torch.randn(1, CONTEXT_LENGTH, HIDDEN_DIM)
            self.register_buffer('pos_encoding', pos_encoding)
        # Expert assignments based on token types
        self.expert_assignments = create_expert_assignments()
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(HIDDEN_DIM, num_experts)
        )
        
        # Experts
        self.experts = nn.ModuleList([
            TransformerExpert() for _ in range(num_experts)
        ])
        
        # Output layer
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def get_pos_encoding_slice(self, seq_length):
        """Get the appropriate slice of positional encoding"""
        return self.pos_encoding[:, :seq_length, :]   

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
        # Embedding + positional encoding
        x = self.embedding(x)
        pos_enc = self.get_pos_encoding_slice(x.size(1))
        x = x + pos_enc
        
        # Calculate routing weights
        avg_hidden = x.mean(dim=1)
        routing_logits = self.router(avg_hidden)
        learned_weights = F.softmax(routing_logits/self.route_temp, dim=-1)
        token_weights = self.compute_token_expert_weights(x)
        
        # Combine learned and token-based weights
        routing_weights = 0.8 * learned_weights + 0.2 * token_weights
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Fix: Reshape routing weights to match expert_outputs dimensions
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_experts, 1, 1]
        # Combine expert outputs
        combined = torch.sum(expert_outputs * routing_weights, dim=1)  # [batch_size, seq_length, hidden_dim]
        
        # Generate output logits
        logits = self.output(combined)
        
        if return_losses and self.training:
            # Compute routing loss
            routing_loss = F.kl_div(
                learned_weights.log(), token_weights, reduction='batchmean'
            )
            diversity_loss = calculate_diversity_loss(expert_outputs)
            
            return logits, {'routing_loss': routing_loss, 
                            'diversity_loss':diversity_loss
                            }
            
        return logits

    
def main():
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    viz_save_path = "./plots_transformer"    
    # Load data with specified fraction (e.g., 0.1 for 10% of data)
    train_loader, val_loader, test_loader = load_data(
        batch_size=BATCH_SIZE,
        data_fraction=DATA_FRACTION
    )
    
    # Train unguided model
    print("Training Unguided MoE Transformer...")
    unguided_model = UnGuidedMoETransformer().to(device)
    print(f"\nTotal trainable parameters in UnGuided Transformer MoE: {count_parameters(unguided_model):,}")
    train_model(unguided_model, train_loader, val_loader,num_epochs=TOTAL_EPOCHS,viz_path=viz_save_path)
    
    # Train guided model
    print("\nTraining Guided MoE Transformer...")
    guided_model = GuidedMoETransformer().to(device)
    print(f"\nTotal trainable parameters in Guided Transformer MoE: {count_parameters(guided_model):,}")
    train_model(guided_model, train_loader, val_loader,num_epochs=TOTAL_EPOCHS,viz_path=viz_save_path)

    print("\nTesting Unguided Model Generation:")
    test_generation('best_model_UnGuidedMoETransformer.pth', "unguided")
    
    print("\nTesting Guided Model Generation:")
    test_generation('best_model_GuidedMoETransformer.pth', "guided")
    
if __name__ == '__main__':
    main()
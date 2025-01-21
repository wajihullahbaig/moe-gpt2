import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, Optional
import random
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from datetime import datetime
import os


# Constants
CONTEXT_LENGTH = 128
VOCAB_SIZE = 50257  # GPT-2 vocabulary size
NUM_EXPERTS = 5
HIDDEN_DIM = 256 
NUM_HEADS = 8   

def set_seed(seed: Optional[int] = 42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def visualize_expert_usage(model, val_loader, device, epoch, save_path='./plots'):
    """Visualize expert usage patterns"""
    model.eval()
    expert_usage = []
    token_expert_map = np.zeros((VOCAB_SIZE, NUM_EXPERTS))
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            
            # Get expert weights based on model type
            if isinstance(model, GuidedMoETransformer):
                # For guided model, get both learned and token-based weights
                x = model.embedding(inputs)
                # Get the correct slice of positional encoding
                seq_length = x.size(1)
                pos_enc = model.pos_encoding[:, :seq_length, :]
                x = x + pos_enc
                
                learned_weights = F.softmax(model.router(x.mean(dim=1)), dim=-1)
                token_weights = model.compute_token_expert_weights(inputs)
                routing_weights = 0.7 * learned_weights + 0.3 * token_weights
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
        if isinstance(model, GuidedMoETransformer):
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
    model_type = "guided" if isinstance(model, GuidedMoETransformer) else "unguided"
    save_file = os.path.join(save_path, f'expert_viz_{model_type}_epoch{epoch}_{timestamp}.png')
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        'expert_usage': avg_usage,
        'specialization': specialization
    }

            
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=CONTEXT_LENGTH, data_fraction=1.0, stride=None):
        self.encodings = []
        self.labels = []
        self.total_tokens = 0
        
        # If stride is None, use non-overlapping windows for test/val
        if stride is None:
            stride = max_length  # No overlap
        
        # Calculate how many texts to use based on fraction
        num_texts = int(len(texts) * data_fraction)
        texts = texts[:num_texts]
        
        for text in tqdm(texts, desc="Processing texts"):
            if isinstance(text, str) and text.strip():
                tokens = tokenizer.encode(text)
                self.total_tokens += len(tokens)
                
                # Use stride to control overlap
                for i in range(0, len(tokens) - max_length, stride):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.encodings.append(chunk[:-1])
                        self.labels.append(chunk[1:])
    
    def __len__(self):
        return len(self.encodings)
    
    def get_stats(self):
        """Return statistics about the dataset"""
        return {
            'num_sequences': len(self.encodings),
            'total_tokens': self.total_tokens,
            'avg_tokens_per_seq': self.total_tokens / len(self.encodings) if self.encodings else 0
        }
        
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx]), torch.tensor(self.labels[idx])           

def load_data(batch_size=32, data_fraction=1.0):
    """Load and prepare data with specified fraction and token counting"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"Loading Wikitext-2 dataset with {data_fraction*100}% of data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Get texts
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']
    test_texts = dataset['test']['text']
    
    print(f"Creating datasets...")
    # Training data can have overlapping windows
    train_dataset = TextDataset(
        train_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=CONTEXT_LENGTH  
    )
    
    # Validation and test should not have overlapping windows
    val_dataset = TextDataset(
        val_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=CONTEXT_LENGTH//2
    )
    test_dataset = TextDataset(
        test_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=CONTEXT_LENGTH//2
    )
    
    # Print detailed statistics
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        stats = dataset.get_stats()
        print(f"\n{name} Dataset Statistics:")
        print(f"Number of sequences: {stats['num_sequences']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Average tokens per sequence: {stats['avg_tokens_per_seq']:.2f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffling for validation
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffling for test
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

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
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
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

    def calculate_diversity_loss(self, expert_outputs):
        """Calculate diversity loss between experts"""
        # Flatten expert outputs for similarity calculation
        batch_size = expert_outputs.size(0)
        expert_outputs_flat = expert_outputs.view(batch_size, NUM_EXPERTS, -1)
        
        # Calculate cosine similarity between expert outputs
        similarities = torch.matmul(
            expert_outputs_flat, 
            expert_outputs_flat.transpose(1, 2)
        )
        norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
        similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + 1e-8)
        
        # We want to minimize similarity between different experts
        diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        return diversity_loss 
        
    def forward(self, x, return_losses=False):
        # x shape: [batch_size, seq_length]
        
        # Embedding + positional encoding
        x = self.embedding(x)  # [batch_size, seq_length, hidden_dim]
        pos_enc = self.get_pos_encoding_slice(x.size(1))
        x = x + pos_enc
        
        # Calculate routing weights
        avg_hidden = x.mean(dim=1)  # [batch_size, hidden_dim]
        routing_logits = self.router(avg_hidden)  # [batch_size, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)  # [batch_size, num_experts]
        
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
                diversity_loss = self.calculate_diversity_loss(expert_outputs)
                return logits, {
                    'balance_loss': balance_loss,
                    "diversity_loss":diversity_loss 
                                }
            
        return logits

class GuidedMoETransformer(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS,dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        if not hasattr(self, 'pos_encoding'):
            pos_encoding = torch.randn(1, CONTEXT_LENGTH, HIDDEN_DIM)
            self.register_buffer('pos_encoding', pos_encoding)
        # Expert assignments based on token types
        self.expert_assignments = self.create_expert_assignments()
        
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
    
    def create_expert_assignments(self):
        """Create token-type to expert assignments"""
        # Simple assignment strategy: split vocabulary into ranges
        vocab_per_expert = VOCAB_SIZE // NUM_EXPERTS
        assignments = {}
        for expert_idx in range(NUM_EXPERTS):
            start_idx = expert_idx * vocab_per_expert
            end_idx = start_idx + vocab_per_expert if expert_idx < NUM_EXPERTS-1 else VOCAB_SIZE
            assignments[expert_idx] = list(range(start_idx, end_idx))
        return assignments
        
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
    
    def calculate_diversity_loss(self, expert_outputs):
        """Calculate diversity loss between experts"""
        # Flatten expert outputs for similarity calculation
        batch_size = expert_outputs.size(0)
        expert_outputs_flat = expert_outputs.view(batch_size, NUM_EXPERTS, -1)
        
        # Calculate cosine similarity between expert outputs
        similarities = torch.matmul(
            expert_outputs_flat, 
            expert_outputs_flat.transpose(1, 2)
        )
        norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
        similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + 1e-8)
        
        # We want to minimize similarity between different experts
        diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        return diversity_loss
        
    def forward(self, x, return_losses=False):
        # Embedding + positional encoding
        x = self.embedding(x)
        pos_enc = self.get_pos_encoding_slice(x.size(1))
        x = x + pos_enc
        
        # Calculate routing weights
        learned_weights = F.softmax(self.router(x.mean(dim=1)), dim=-1)
        token_weights = self.compute_token_expert_weights(x)
        
        # Combine learned and token-based weights
        routing_weights = 0.7 * learned_weights + 0.3 * token_weights
        
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
            diversity_loss = self.calculate_diversity_loss(expert_outputs)
            
            return logits, {'routing_loss': routing_loss, 
                            'diversity_loss':diversity_loss
                            }
            
        return logits

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_loss = float('inf')
    total_tokens_processed = 0  # Track total tokens across all epochs
    
    # Create plot directories
    os.makedirs('./plots', exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'expert_usage': [],
        'specialization': [],
        'tokens_per_epoch': [],  # Track tokens for each epoch
        'total_tokens': []       # Running total of tokens
    }
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_aux_losses = defaultdict(float)
        epoch_tokens = 0  # Track tokens for this epoch
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Count tokens in this batch
            batch_tokens = inputs.numel()
            epoch_tokens += batch_tokens
            
            optimizer.zero_grad()
            outputs, aux_losses = model(inputs, return_losses=True)
            
            # Compute main loss
            main_loss = F.cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            
            # Track auxiliary losses
            for loss_name, loss_value in aux_losses.items():
                epoch_aux_losses[loss_name] += loss_value.item()
            
            # Add auxiliary losses
            total_batch_loss = main_loss + sum(aux_losses.values())
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        # Update total tokens
        total_tokens_processed += epoch_tokens
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_aux_losses = {name: value / len(train_loader) 
                         for name, value in epoch_aux_losses.items()}
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        
        # Visualize expert behavior
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            viz_stats = visualize_expert_usage(model, val_loader, device, epoch)
            history['expert_usage'].append(viz_stats['expert_usage'])
            history['specialization'].append(viz_stats['specialization'])
        
        # Print metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}')
        print(f'Tokens this epoch: {epoch_tokens:,}')
        print(f'Total tokens processed: {total_tokens_processed:,}')
        for name, value in avg_aux_losses.items():
            print(f'{name} = {value:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 
                      f'best_model_{type(model).__name__}.pth')
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['tokens_per_epoch'].append(epoch_tokens)
        history['total_tokens'].append(total_tokens_processed)
    
    # Print final token statistics
    print('\nFinal Token Statistics:')
    print(f'Average tokens per epoch: {np.mean(history["tokens_per_epoch"]):,.0f}')
    print(f'Total tokens processed: {total_tokens_processed:,}')
    
    return history

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def generate_text(model, tokenizer, input_text, max_new_tokens=50, temperature=0.7, device='cuda'):
    """Generate text with token counting"""
    model.eval()
    
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_token_count = input_ids.size(1)
    
    # Initialize generated sequence with input
    generated = input_ids
    generated_token_count = 0
    
    # Generate one token at a time
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Take last context_length tokens if input is too long
            if generated.size(1) > CONTEXT_LENGTH - 1:
                context = generated[:, -(CONTEXT_LENGTH - 1):]
            else:
                context = generated
                
            try:
                outputs = model(context)
                next_token_logits = outputs[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                generated_token_count += 1
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
            except RuntimeError as e:
                print(f"Error during generation at token {generated_token_count}")
                print(f"Current sequence length: {generated.size(1)}")
                print(f"Context length: {context.size(1)}")
                raise e
    
    # Decode generated tokens
    try:
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as e:
        print("Error during decoding")
        print(f"Generated tensor shape: {generated.shape}")
        print(f"Generated tokens: {generated.tolist()}")
        raise e
    
    # Return text and token counts
    return {
        'text': generated_text,
        'input_tokens': input_token_count,
        'generated_tokens': generated_token_count,
        'total_tokens': input_token_count + generated_token_count
    }
def evaluate_generation(model, tokenizer, test_prompts, device='cuda'):
    """Evaluate generation with token counting"""
    model.eval()
    
    print("\nGenerating text from test prompts:")
    print("-" * 50)
    
    total_input_tokens = 0
    total_generated_tokens = 0
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("\nGenerated continuation:")
        
        for temp in [0.5,0.7, 1.0,1.5, 2.0]:
            result = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_new_tokens=50, 
                temperature=temp,
                device=device
            )
            
            total_input_tokens += result['input_tokens']
            total_generated_tokens += result['generated_tokens']
            
            # Print continuation and token counts
            continuation = result['text'][len(prompt):]
            print(f"\nTemperature {temp}:")
            print(continuation)
            print(f"Tokens - Input: {result['input_tokens']}, "
                  f"Generated: {result['generated_tokens']}, "
                  f"Total: {result['total_tokens']}")
            print("-" * 30)
    
    print("\nOverall Token Statistics:")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total tokens processed: {total_input_tokens + total_generated_tokens}")


# Example usage:
def test_generation(model_path, model_type="guided", device='cuda'):
    """
    Test text generation with a saved model
    
    Args:
        model_path (str): Path to saved model weights
        model_type (str): "guided" or "unguided"
        device: torch device
    """
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Initialize model
    if model_type == "guided":
        model = GuidedMoETransformer().to(device)
    else:
        model = UnGuidedMoETransformer().to(device)
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path))
    
    # Test prompts
    test_prompts = [
        "The history of artificial intelligence",
        "In recent scientific discoveries,",
        "The most important aspect of learning is",
        "The future of technology lies in",
        "The great peot of Persia, Jallaludin Rumi once said, if the world brings you to your knees, you are in the"
    ]
    
    # Generate text
    evaluate_generation(model, tokenizer, test_prompts, device)

def count_parameters(model):
    """Count total trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def main():
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
        
    # Load data with specified fraction (e.g., 0.1 for 10% of data)
    data_fraction = 0.01    
    train_loader, val_loader, test_loader = load_data(
        batch_size=16,
        data_fraction=data_fraction
    )
    
    # Train unguided model
    print("Training Unguided MoE Transformer...")
    unguided_model = UnGuidedMoETransformer().to(device)
    print(f"\nTotal trainable parameters in UnGuided MoE: {count_parameters(unguided_model):,}")
    train_model(unguided_model, train_loader, val_loader,num_epochs=50)
    
    # Train guided model
    print("\nTraining Guided MoE Transformer...")
    guided_model = GuidedMoETransformer().to(device)
    print(f"\nTotal trainable parameters in Guided MoE: {count_parameters(guided_model):,}")
    train_model(guided_model, train_loader, val_loader,num_epochs=50)

    print("\nTesting Unguided Model Generation:")
    test_generation('best_model_UnGuidedMoETransformer.pth', "unguided")
    
    print("\nTesting Guided Model Generation:")
    test_generation('best_model_GuidedMoETransformer.pth', "guided")
    
if __name__ == '__main__':
    main()
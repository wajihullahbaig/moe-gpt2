
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch


from common import visualize_expert_usage
from constants import CONTEXT_LENGTH, VOCAB_SIZE


def train_model(model, train_loader, val_loader, num_epochs=10,viz_path=None, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_loss = float('inf')
    total_tokens_processed = 0  # Track total tokens across all epochs
    
   
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
            viz_stats = visualize_expert_usage(model, val_loader, device, epoch,save_path=viz_path)
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
    from main_transformer_arch import GuidedMoETransformer, UnGuidedMoETransformer
    from main_gpt2_arch import GuidedMoETransformer, UnGuidedMoETransformer
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
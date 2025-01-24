from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch
from datasets import load_dataset

from constants import CONTEXT_LENGTH, STRIDE

            
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
        stride=STRIDE
    )
    
    # Validation and test should not have overlapping windows
    val_dataset = TextDataset(
        val_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=STRIDE
    )
    test_dataset = TextDataset(
        test_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=STRIDE
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
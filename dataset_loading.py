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
        
        if stride is None:
            stride = max_length
        
        num_texts = int(len(texts) * data_fraction)
        texts = texts[:num_texts]
        
        for text in tqdm(texts, desc="Processing texts"):
            if isinstance(text, str) and text.strip():
                tokens = tokenizer.encode(text)
                self.total_tokens += len(tokens)
                
                for i in range(0, len(tokens), stride):
                    chunk = tokens[i:i + max_length]
                    
                    # Ensure chunk is long enough
                    if len(chunk) == max_length:
                        self.encodings.append(chunk[:-1])
                        self.labels.append(chunk[1:])
        
        # Add debug print
        print(f"Processed {len(self.encodings)} sequences")
        print(f"First encoding sample: {self.encodings[0] if self.encodings else 'No encodings'}")
        print(f"First label sample: {self.labels[0] if self.labels else 'No labels'}")
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # Add error checking
        if idx < 0 or idx >= len(self.encodings):
            raise IndexError(f"Index {idx} out of range")
        
        # Ensure both encodings and labels are not None
        if self.encodings[idx] is None or self.labels[idx] is None:
            raise ValueError(f"None value at index {idx}")
        
        return torch.tensor(self.encodings[idx]), torch.tensor(self.labels[idx])
    
    def get_stats(self):
        return {
            'num_sequences': len(self.encodings),
            'total_tokens': self.total_tokens,
            'avg_tokens_per_seq': self.total_tokens / len(self.encodings) if self.encodings else 0
        }

def load_data(batch_size=32, data_fraction=1.0):
    """Load and prepare data with specified fraction and token counting"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"Loading Wikitext-2 dataset with {data_fraction*100}% of data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    #dataset = load_dataset('wikimedia/wikipedia', '20231101.en')
    
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



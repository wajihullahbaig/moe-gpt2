from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch
from datasets import load_dataset
import re
from typing import List, Optional
from bs4 import BeautifulSoup
import html

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
        
        # Preprocess all texts first
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            processed = self.preprocess_text(text)
            if processed:
                processed_texts.append(processed)
        
        for text in tqdm(processed_texts, desc="Tokenizing texts"):
            if isinstance(text, str) and text.strip():
                tokens = tokenizer.encode(text)
                self.total_tokens += len(tokens)
                
                for i in range(0, len(tokens), stride):
                    chunk = tokens[i:i + max_length]
                    
                    # Ensure chunk is long enough
                    if len(chunk) == max_length:
                        self.encodings.append(chunk[:-1])
                        self.labels.append(chunk[1:])
        
        print(f"Processed {len(self.encodings)} sequences")
        print(f"First encoding sample: {self.encodings[0] if self.encodings else 'No encodings'}")
    
    @staticmethod
    def clean_wiki_links(text: str) -> str:
        """Clean Wikipedia-style links while preserving readable text."""
        # Replace [[link|text]] with text, or [[text]] with text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        return text

    @staticmethod
    def preprocess_text(text: str) -> Optional[str]:
        """
        Preprocess text using BeautifulSoup for better HTML handling.
        Returns None if text should be filtered out.
        """
        if not isinstance(text, str):
            return None
            
        if not text.strip():
            return None

        try:
            # First unescape any HTML entities
            text = html.unescape(text)
            
            # Create a BeautifulSoup object for better HTML parsing
            # Use 'html.parser' as it's built into Python
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Remove HTML comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text):
                comment.extract()
            
            # Get text content
            text = soup.get_text()
            
            # Clean wiki-specific patterns
            text = TextDataset.clean_wiki_links(text)
            
            # Remove section headers
            text = re.sub(r'={2,}.*?={2,}', '', text)
            
            # Remove templates
            text = re.sub(r'\{\{.*?\}\}', '', text)
            
            # Remove reference numbers
            text = re.sub(r'\[\d+\]', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # Split into lines and clean
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip administrative lines
                if re.match(r'^(Category:|File:|Image:|Reference|See also|External links)', line):
                    continue
                # Skip numeric-only lines
                if re.match(r'^\s*\d+\s*$', line):
                    continue
                # Skip short lines
                if len(line) < 20:
                    continue
                cleaned_lines.append(line)
            
            # Join lines back together
            text = ' '.join(cleaned_lines)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Final length check
            if len(text) < 100:
                return None
            
            return text
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return None

    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.encodings):
            raise IndexError(f"Index {idx} out of range")
        
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
    """Load and prepare Wikipedia data with preprocessing"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"Loading Wikitext-2 dataset with {data_fraction*100}% of data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    # Alternatively for full Wikipedia:
    # dataset = load_dataset('wikipedia', '20231101.en')
    
    # Get texts
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']
    test_texts = dataset['test']['text']
    
    print(f"Creating datasets with preprocessing...")
    # Training data with overlapping windows
    train_dataset = TextDataset(
        train_texts, 
        tokenizer, 
        data_fraction=data_fraction,
        stride=STRIDE
    )
    
    # Validation and test sets
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
    
    # Print statistics for each dataset
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        stats = dataset.get_stats()
        print(f"\n{name} Dataset Statistics:")
        print(f"Number of sequences: {stats['num_sequences']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Average tokens per sequence: {stats['avg_tokens_per_seq']:.2f}")
    
    # Create dataloaders with multiprocessing
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
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
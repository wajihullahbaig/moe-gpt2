import logging
import os
import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from config import CONTEXT_LENGTH, DATA_CACHE_DIR, NUM_WORKERS, STRIDE

log = logging.getLogger('moe')

CACHE_VERSION = 1  # bump whenever cleaning or tokenization changes
TITLE = re.compile(r'^ = [^=].* = $')  # top-level " = Article = " line
HEADER = re.compile(r'^ =( =)+ .* =( =)+ $')  # " = = Section = = " lines
DETOKENIZE = [
    (re.compile(r' @([-,.])@ '), r'\1'),  # 1 @,@ 000 -> 1,000 · role @-@ playing -> role-playing
    (re.compile(r" (['’](?:s|re|ve|d|ll|m)\b|n't)"), r'\1'),  # game 's -> game's
    (re.compile(r' ([.,;:!?%)\]])'), r'\1'),  # space before closing punctuation
    (re.compile(r'([(\[]) '), r'\1'),  # space after opening brackets
]


def clean_line(line):
    """Undo WikiText's tokenized spacing and @x@ escapes in one paragraph."""
    for pattern, replacement in DETOKENIZE:
        line = pattern.sub(replacement, line)
    return line.strip()


def wikitext_articles(lines):
    """Group raw WikiText lines into cleaned article texts, dropping titles, section headers and blanks."""
    articles, paragraphs = [], []
    for line in (raw.rstrip('\n') for raw in lines):
        if TITLE.match(line):
            if paragraphs:
                articles.append('\n'.join(paragraphs))
            paragraphs = []
        elif line.strip() and not HEADER.match(line):
            paragraphs.append(clean_line(line))
    if paragraphs:
        articles.append('\n'.join(paragraphs))
    return articles


def tokenize_split(lines, tokenizer):
    """One flat token stream for a split: every article followed by an EOS token."""
    ids = tokenizer(wikitext_articles(lines), verbose=False)['input_ids']
    return torch.tensor([token for article in ids for token in article + [tokenizer.eos_token_id]])


def cached_tokens(split, lines, tokenizer):
    """Tokenize a split once and reuse it from DATA_CACHE_DIR afterwards."""
    path = os.path.join(DATA_CACHE_DIR, f'wikitext2_{split}_v{CACHE_VERSION}.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    tokens = tokenize_split(lines, tokenizer)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    torch.save(tokens, path)
    return tokens


class TokenWindows(Dataset):
    """(input, next-token target) pairs from windows of `window` tokens taken every `stride` tokens."""

    def __init__(self, tokens, window=CONTEXT_LENGTH, stride=CONTEXT_LENGTH):
        self.tokens, self.window, self.stride = tokens, window, stride

    def __len__(self):
        return max(0, (len(self.tokens) - self.window) // self.stride + 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx * self.stride: idx * self.stride + self.window]
        return chunk[:-1], chunk[1:]


def load_data(tokenizer, batch_size, data_fraction=1.0):
    """Train/validation/test DataLoaders over WikiText-2.

    Training uses overlapping windows (STRIDE) on the first data_fraction of the stream; validation and test
    always use the full split with stride CONTEXT_LENGTH - 1, so every token is predicted exactly once.
    """
    log.info(f'Loading WikiText-2 (train fraction {data_fraction:.0%}, full validation/test)')
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    loaders = []
    for split in ('train', 'validation', 'test'):
        tokens = cached_tokens(split, dataset[split]['text'], tokenizer)
        is_train = split == 'train'
        if is_train:
            tokens = tokens[:int(len(tokens) * data_fraction)]
        windows = TokenWindows(tokens, stride=STRIDE if is_train else CONTEXT_LENGTH - 1)
        log.info(f'  {split:<10} {len(windows):>9,} windows │ {len(tokens):>11,} tokens')
        loaders.append(DataLoader(windows, batch_size=batch_size, shuffle=is_train, num_workers=NUM_WORKERS,
                                  pin_memory=True, persistent_workers=NUM_WORKERS > 0))
    return loaders

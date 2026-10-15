import torch
import torch.nn.functional as F

from config import CONTEXT_LENGTH
from log_utils import banner, log

PROMPTS = [
    "The history of artificial intelligence",
    "In recent scientific discoveries,",
    "The most important aspect of learning is",
    "The future of technology lies in",
    "The great poet of Persia, Jallaludin Rumi once said, if the world brings you to your knees, you are in the",
]
TEMPERATURES = (0.7, 1.0, 1.5)
PREVIEW_PROMPTS, PREVIEW_TEMPERATURE, PREVIEW_TOKENS = PROMPTS[:2], 0.8, 30


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=50, temperature=1.0):
    """Sample a continuation of prompt token by token and return only the new text."""
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = ids.size(1)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -(CONTEXT_LENGTH - 1):])[:, -1] / temperature
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True)


def _one_line(text):
    """Collapse whitespace so a sample fits on one log line."""
    return ' '.join(text.split())


def log_preview(model, tokenizer, device):
    """Log a short sample for a couple of prompts (used whenever a new best checkpoint is saved)."""
    for prompt in PREVIEW_PROMPTS:
        sample = generate(model, tokenizer, prompt, device, PREVIEW_TOKENS, PREVIEW_TEMPERATURE)
        log.info(f'    ✎ {prompt} ⟶ {_one_line(sample)}')


def log_samples(model, tokenizer, device):
    """Log sampled continuations of every prompt at several temperatures; return them as {prompt: {T: text}}."""
    banner(f'Samples · {model.name}')
    samples = {}
    for prompt in PROMPTS:
        log.info(f'▸ {prompt}')
        samples[prompt] = {}
        for temperature in TEMPERATURES:
            text = _one_line(generate(model, tokenizer, prompt, device, temperature=temperature))
            samples[prompt][str(temperature)] = text
            log.info(f'    T={temperature:<3} │ {text}')
    return samples

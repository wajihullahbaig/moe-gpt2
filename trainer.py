import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import EPOCHS, GRAD_CLIP, LEARNING_RATE, PATIENCE, PLOT_EVERY, TASK_LOSS_WEIGHT, WEIGHT_DECAY
from generation import log_preview
from log_utils import Table, log
from plotting import plot_expert_usage


def set_seed(seed):
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state():
    """Capture every RNG so a resumed run continues the same random stream."""
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    """Restore RNGs captured by _rng_state."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    """Token-averaged cross-entropy over a loader."""
    model.eval()
    total, tokens = torch.zeros((), device=device), 0
    for ids, targets in loader:
        ids, targets = ids.to(device), targets.to(device)
        total += F.cross_entropy(model(ids).flatten(0, 1), targets.flatten(), reduction='sum')
        tokens += targets.numel()
    return total.item() / tokens


def _per_expert(values):
    """Format per-expert values compactly, e.g. 0.52/0.48."""
    return '/'.join(f'{v:.2f}' for v in values) if values is not None else ''


def _train_epoch(model, loader, optimizer, device, desc):
    """Run one optimisation pass; return (mean train CE, mean aux losses, tokens per second)."""
    model.train()
    start = time.perf_counter()
    ce_sum, aux_sums, tokens = torch.zeros((), device=device), {}, 0
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (ids, targets) in enumerate(bar):
        ids, targets = ids.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits, aux = model(ids, return_losses=True)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss = TASK_LOSS_WEIGHT * ce + sum(aux.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        ce_sum += ce.detach()
        for name, value in aux.items():
            aux_sums[name] = aux_sums.get(name, 0) + value.detach()
        tokens += targets.numel()
        if step % 50 == 0:
            bar.set_postfix(ce=f'{ce.item():.3f}')

    steps = len(loader)
    train_ce = ce_sum.item() / steps  # single GPU sync per epoch
    return train_ce, {name: value.item() / steps for name, value in aux_sums.items()}, tokens / (time.perf_counter() - start)


def _cosine_lr(epoch, epochs):
    """Closed-form cosine schedule: stateless, so a resumed run (even with a new EPOCHS) stays on schedule."""
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * epoch / epochs))


def _stale_epochs(history):
    """Epochs since the best validation CE (0 when the latest epoch is the best)."""
    best = min(range(len(history)), key=lambda i: history[i]['val_ce'], default=0)
    return max(0, len(history) - 1 - best)


def _load_checkpoint(path, model, optimizer, epochs):
    """Restore a resumable checkpoint; return (start epoch, best val CE, history, seconds)."""
    if not os.path.exists(path):
        return 0, float('inf'), [], 0.0
    state = torch.load(path, map_location='cpu', weights_only=False)  # RNG states must stay on CPU
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    _set_rng_state(state['rng'])
    log.info(f"Resumed {model.name} at epoch {state['epoch']}/{epochs} (best val ce {state['best_val']:.4f})")
    return state['epoch'], state['best_val'], state['history'], state['seconds']


def _save_checkpoint(path, model, optimizer, **progress):
    """Atomically save everything needed to resume, so an interrupt never leaves a broken file."""
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng_state(), **progress}
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def _epoch_table(aux_names, num_experts):
    """Column layout of the per-epoch log table."""
    expert_width = max(8, 5 * num_experts - 1)
    return Table([('epoch', 7), ('lr', 8), ('train ce', 8), *((name, 9) for name in aux_names),
                  ('val ce', 7), ('val ppl', 9), ('usage', expert_width), ('affinity', expert_width),
                  ('tok/s', 7), ('time', 5), ('', 1)])


def train_model(model, train_loader, val_loader, device, run_dir, tokenizer, epochs=EPOCHS):
    """Train (resuming, early-stopping), reload the best weights; return (history, best val CE, seconds, best stats)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_path = os.path.join(run_dir, f'{model.name}.pth')
    last_path = os.path.join(run_dir, f'{model.name}_last.pth')
    start_epoch, best_val, history, seconds = _load_checkpoint(last_path, model, optimizer, epochs)

    table = None
    for epoch in range(start_epoch, epochs):
        if _stale_epochs(history) >= PATIENCE:
            log.info(f'Early stop: no val improvement for {PATIENCE} epochs (best val ce {best_val:.4f})')
            break
        start = time.perf_counter()
        lr = _cosine_lr(epoch, epochs)
        for group in optimizer.param_groups:
            group['lr'] = lr
        train_ce, aux, tokens_per_second = _train_epoch(model, train_loader, optimizer, device,
                                                        f'{model.name} {epoch + 1}/{epochs}')
        val_ce = evaluate(model, val_loader, device)
        stats = None
        if epoch % PLOT_EVERY == 0 or epoch == epochs - 1:
            stats = plot_expert_usage(model, val_loader, device, f'{model.name} · epoch {epoch + 1}',
                                      os.path.join(run_dir, 'plots', f'{model.name}_epoch{epoch + 1:03d}.png'), tokenizer)

        is_best = val_ce < best_val
        if is_best:
            best_val = val_ce
            torch.save(model.state_dict(), best_path)
        epoch_seconds = time.perf_counter() - start
        seconds += epoch_seconds
        history.append({'epoch': epoch + 1, 'lr': lr, 'train_ce': train_ce, 'val_ce': val_ce, **aux,
                        **({k: v.tolist() for k, v in stats.items()} if stats else {})})
        _save_checkpoint(last_path, model, optimizer,
                         epoch=epoch + 1, best_val=best_val, history=history, seconds=seconds)

        table = table or _epoch_table(aux, len(model.experts))
        table.row(f'{epoch + 1}/{epochs}', f'{lr:.2e}', f'{train_ce:.4f}', *(f'{v:.4f}' for v in aux.values()),
                  f'{val_ce:.4f}', f'{math.exp(val_ce):.1f}',
                  _per_expert(stats and stats['usage']), _per_expert(stats and stats['affinity']),
                  f'{tokens_per_second / 1e3:.1f}k', f'{epoch_seconds:.0f}s', '★' if is_best else '')
        if is_best:
            log_preview(model, tokenizer, device)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    log.info(f'Loaded best weights (val ce {best_val:.4f}) from {best_path}')
    best_stats = plot_expert_usage(model, val_loader, device, f'{model.name} · best checkpoint',
                                   os.path.join(run_dir, 'plots', f'{model.name}_best.png'), tokenizer)
    return history, best_val, seconds, {k: v.tolist() for k, v in best_stats.items()}

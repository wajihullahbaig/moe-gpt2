"""Train, test and sample MoE language models.

    python train.py --arch transformer                    # new run, every routing strategy
    python train.py --arch gpt2 --routing guided          # new run, one strategy
    python train.py --arch transformer --resume latest    # continue the newest transformer run
"""
import argparse
import glob
import json
import math
import os
from datetime import datetime

import torch
from transformers import GPT2TokenizerFast

import config
from config import BATCH_SIZE, DATA_FRACTION, RUNS_DIR, SEED, VOCAB_SIZE
from data import load_data
from generation import log_samples
from log_utils import Table, banner, log, setup_logger
from models import EXPERTS, ROUTERS, build_model
from plotting import plot_training_curves
from trainer import count_parameters, evaluate, set_seed, train_model


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Train MoE language models on WikiText-2.')
    parser.add_argument('--arch', choices=sorted(EXPERTS), required=True, help='expert architecture')
    parser.add_argument('--routing', nargs='+', choices=list(ROUTERS), default=list(ROUTERS),
                        help='routing strategies to train, in order (default: all)')
    parser.add_argument('--resume', metavar='RUN_DIR', help="run folder to continue, or 'latest' for this --arch")
    return parser.parse_args()


def resolve_run_dir(arch, resume):
    """Return a fresh run folder, or the existing one to resume."""
    if not resume:
        return os.path.join(RUNS_DIR, f'{arch}_{datetime.now():%Y%m%d_%H%M%S}')
    if resume == 'latest':
        runs = sorted(glob.glob(os.path.join(RUNS_DIR, f'{arch}_*')))
        if not runs:
            raise SystemExit(f'No {arch} runs found in {RUNS_DIR}/ to resume')
        return runs[-1]
    if not os.path.isdir(resume):
        raise SystemExit(f'Run folder not found: {resume}')
    return resume


def save_settings(run_dir, settings):
    """Write settings.json, warning about any value that differs from the run being resumed."""
    path = os.path.join(run_dir, 'settings.json')
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        for name in (n for n in settings if saved.get(n) != settings[n]):
            log.warning(f'⚠ {name} changed since this run started: {saved.get(name)} → {settings[name]}')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)


def train_and_test(arch, routing, loaders, device, run_dir, tokenizer, token_counts):
    """Train one model (resuming if possible), test it, sample from it and return its results row."""
    train_loader, val_loader, test_loader = loaders
    set_seed(SEED)
    model = build_model(arch, routing, token_counts).to(device)
    history_path = os.path.join(run_dir, f'{model.name}_history.json')
    finished_before = os.path.exists(history_path)
    banner(f'Training · {model.name} · {count_parameters(model):,} parameters')
    history, best_val, seconds, best_stats = train_model(model, train_loader, val_loader, device, run_dir, tokenizer)
    test_ce = evaluate(model, test_loader, device)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    if not finished_before:
        with open(os.path.join(run_dir, f'{model.name}_samples.json'), 'w', encoding='utf-8') as f:
            json.dump(log_samples(model, tokenizer, device), f, indent=2, ensure_ascii=False)
    return {'model': model.name, 'params': count_parameters(model), 'best_val_ce': best_val,
            'test_ce': test_ce, 'test_ppl': math.exp(test_ce), 'minutes': seconds / 60,
            'epochs': len(history), **best_stats}


def log_results(results, run_dir):
    """Log the final comparison table, save it as results.json and plot the training curves."""
    histories = {}
    for r in results:
        with open(os.path.join(run_dir, f"{r['model']}_history.json")) as f:
            histories[r['model']] = json.load(f)
    plot_training_curves(histories, os.path.join(run_dir, 'plots', 'training_curves.png'))
    banner('Results')
    table = Table([('model', -22), ('params', 9), ('best val ce', 11), ('test ce', 7), ('test ppl', 9), ('minutes', 7)])
    for r in results:
        table.row(r['model'], f"{r['params']:,}", f"{r['best_val_ce']:.4f}", f"{r['test_ce']:.4f}",
                  f"{r['test_ppl']:.1f}", f"{r['minutes']:.1f}")
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'Artifacts written to {run_dir}')


def main():
    """Set up the run folder, data and logging, then train and compare every requested routing strategy."""
    args = parse_args()
    run_dir = resolve_run_dir(args.arch, args.resume)
    setup_logger(run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = {'ARCH': args.arch, 'ROUTING': args.routing,
                **{name: value for name, value in vars(config).items() if name.isupper()}}

    banner(f'MoE · {args.arch} · {run_dir}')
    save_settings(run_dir, settings)
    gpu = f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''
    log.info(f'device {device}{gpu} │ torch {torch.__version__}')
    for name, value in settings.items():
        log.info(f'  {name:<28} {value}')

    set_seed(SEED)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    loaders = load_data(tokenizer, BATCH_SIZE, DATA_FRACTION)

    token_counts = torch.bincount(loaders[0].dataset.tokens, minlength=VOCAB_SIZE)
    results = [train_and_test(args.arch, routing, loaders, device, run_dir, tokenizer, token_counts)
               for routing in args.routing]
    log_results(results, run_dir)


if __name__ == '__main__':
    main()

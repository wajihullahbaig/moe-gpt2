"""Generate text with a trained model from a run folder.

    python infer.py --prompt "The history of artificial intelligence"        # newest run, first model
    python infer.py --run runs/gpt2_20261015_120000 --model gpt2-guided      # interactive prompt loop
"""
import argparse
import glob
import json
import os

import torch

import config


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Sample text from a trained MoE language model.')
    parser.add_argument('--run', default='latest', help="run folder, or 'latest' (default)")
    parser.add_argument('--model', help='model name inside the run, e.g. transformer-guided (default: first trained)')
    parser.add_argument('--prompt', help='prompt to continue; omit for an interactive loop')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--seed', type=int, default=config.SEED)
    return parser.parse_args()


def resolve_run(run):
    """Return the run folder, picking the most recently modified one for 'latest'."""
    if run != 'latest':
        return run
    runs = glob.glob(os.path.join(config.RUNS_DIR, '*_*'))
    if not runs:
        raise SystemExit(f'No runs found in {config.RUNS_DIR}/')
    return max(runs, key=os.path.getmtime)


def apply_settings(run_dir):
    """Restore the run's config values so the model is rebuilt exactly as it was trained."""
    with open(os.path.join(run_dir, 'settings.json')) as f:
        settings = json.load(f)
    for name, value in settings.items():
        if hasattr(config, name):
            setattr(config, name, value)
    return settings


def main():
    """Load a trained model from a run and complete one prompt or loop interactively."""
    args = parse_args()
    run_dir = resolve_run(args.run)
    settings = apply_settings(run_dir)

    # Imported after apply_settings: these modules read config values at import time.
    from transformers import GPT2TokenizerFast
    from generation import generate
    from models import build_model
    from trainer import set_seed

    name = args.model or f"{settings['ARCH']}-{settings['ROUTING'][0]}"
    arch, routing = name.split('-', 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(arch, routing).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, f'{name}.pth'), map_location=device, weights_only=True))
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    set_seed(args.seed)
    print(f'{name} from {run_dir} on {device} · temperature {args.temperature}')

    def complete(prompt):
        """Print the prompt followed by its sampled continuation in bold."""
        print(f'\n{prompt}\033[1m{generate(model, tokenizer, prompt, device, args.max_new_tokens, args.temperature)}\033[0m\n')

    if args.prompt:
        complete(args.prompt)
        return
    while prompt := input('prompt (empty to quit) › ').strip():
        complete(prompt)


if __name__ == '__main__':
    main()

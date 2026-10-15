"""Turn finished runs into README material: curves, expert-usage plots, a metrics table and side-by-side samples.

    python report.py runs/transformer_20261015_211524 runs/gpt2_20261016_013000
"""
import argparse
import glob
import json
import os
import shutil

from plotting import plot_training_curves

DOCS_DIR = 'docs'


def load_json(path):
    """Read a UTF-8 JSON file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def collect(run_dirs):
    """Merge results, histories and samples of every model across the given runs."""
    results, histories, samples = [], {}, {}
    for run_dir in run_dirs:
        for row in load_json(os.path.join(run_dir, 'results.json')):
            name = row['model']
            results.append(row)
            histories[name] = load_json(os.path.join(run_dir, f'{name}_history.json'))
            samples[name] = load_json(os.path.join(run_dir, f'{name}_samples.json'))
            latest_plot = sorted(glob.glob(os.path.join(run_dir, 'plots', f'{name}_epoch*.png')))[-1]
            shutil.copy(latest_plot, os.path.join(DOCS_DIR, f'{name}_experts.png'))
    return results, histories, samples


def results_table(results, histories):
    """Markdown metrics table, best test perplexity in bold."""
    best = min(r['test_ppl'] for r in results)
    lines = ['| Model | Params | Epochs | Best val CE | Test CE | Test PPL | Final usage | Final affinity | Minutes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        last = next(h for h in reversed(histories[r['model']]) if 'usage' in h)
        ppl = f"**{r['test_ppl']:.1f}**" if r['test_ppl'] == best else f"{r['test_ppl']:.1f}"
        lines.append(f"| `{r['model']}` | {r['params']:,} | {len(histories[r['model']])} | {r['best_val_ce']:.4f} "
                     f"| {r['test_ce']:.4f} | {ppl} | {'/'.join(f'{u:.2f}' for u in last['usage'])} "
                     f"| {'/'.join(f'{a:.2f}' for a in last['affinity'])} | {r['minutes']:.0f} |")
    return '\n'.join(lines)


def samples_tables(samples, temperature, max_words=40):
    """One markdown table per prompt comparing every model's continuation at one temperature."""
    blocks = []
    for prompt in next(iter(samples.values())):
        rows = [f'**{prompt}** …', '', '| Model | Continuation (T = {}) |'.format(temperature), '|---|---|']
        for name, by_prompt in samples.items():
            words = by_prompt[prompt][temperature].replace('|', '\\|').split()
            rows.append(f"| `{name}` | {' '.join(words[:max_words])}{' …' if len(words) > max_words else ''} |")
        blocks.append('\n'.join(rows))
    return '\n\n'.join(blocks)


def main():
    """Write docs/ figures and docs/results.md for the given run folders."""
    parser = argparse.ArgumentParser(description='Build README results from finished runs.')
    parser.add_argument('runs', nargs='+', help='run folders to include')
    parser.add_argument('--temperature', default='0.7', help='sampling temperature shown in the samples tables')
    args = parser.parse_args()
    os.makedirs(DOCS_DIR, exist_ok=True)

    results, histories, samples = collect(args.runs)
    plot_training_curves(histories, os.path.join(DOCS_DIR, 'training_curves.png'))
    markdown = f"{results_table(results, histories)}\n\n{samples_tables(samples, args.temperature)}\n"
    with open(os.path.join(DOCS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(markdown)


if __name__ == '__main__':
    main()

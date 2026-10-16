import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from config import VOCAB_SIZE
from models.routers import vocab_partition


def plot_training_curves(histories, path):
    """Train and validation cross-entropy per epoch for several models ({name: history}) in one figure."""
    sns.set_theme(style='whitegrid', context='notebook')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for (name, history), color in zip(histories.items(), sns.color_palette('deep', len(histories))):
        epochs = [h['epoch'] for h in history]
        for ax, key in zip(axes, ('train_ce', 'val_ce')):
            ax.plot(epochs, [h[key] for h in history], label=name, color=color, linewidth=1.8)
    for ax, title in zip(axes, ('Training cross-entropy', 'Validation cross-entropy')):
        ax.set(title=title, xlabel='Epoch', ylabel='Cross-entropy (nats)')
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def plot_expert_usage(model, loader, device, title, path, tokenizer, top_tokens=30):
    """Plot expert utilisation and token→expert routing on a loader; return usage and affinity per expert."""
    model.eval()
    num_experts = len(model.experts)
    token_map = torch.zeros(VOCAB_SIZE, num_experts, device=device)
    sequence_usage = []
    for ids, _ in loader:
        ids = ids.to(device)
        weights = model.route(ids).weights
        token_map.index_add_(0, ids.flatten(), weights.flatten(0, 1))
        sequence_usage.append(weights.mean(dim=1).cpu())

    sequence_usage = torch.cat(sequence_usage).numpy()
    token_map = token_map.cpu().numpy()
    counts = token_map.sum(axis=1)
    share = token_map / np.maximum(counts[:, None], 1e-8)
    partition = getattr(model.router, 'token_to_expert', None)  # learned routers: compare against id ranges
    assignment = (vocab_partition(num_experts) if partition is None else partition).cpu().numpy()
    seen = counts > 0
    usage = sequence_usage.mean(axis=0)
    affinity = np.array([share[seen & (assignment == e), e].mean() for e in range(num_experts)])

    sns.set_theme(style='whitegrid', context='notebook')
    palette = sns.color_palette('deep', num_experts)
    experts = [f'E{e}' for e in range(num_experts)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    def bars(ax, values, title, ylabel):
        """Annotated per-expert bar chart with a uniform reference line."""
        ax.bar(experts, values, color=palette, edgecolor='black', linewidth=0.6)
        ax.axhline(1 / num_experts, color='grey', linestyle='--', linewidth=1, label='uniform')
        for x, value in enumerate(values):
            ax.annotate(f'{value:.3f}', (x, value), ha='center', va='bottom', fontsize=10)
        ax.set(title=title, xlabel='Expert', ylabel=ylabel, ylim=(0, 1))
        ax.legend(loc='upper right')

    bars(axes[0, 0], usage, 'Mean routing weight', 'Weight')
    bars(axes[1, 1], affinity, 'Affinity to assigned tokens', 'Mean routing share')

    sns.violinplot(data=sequence_usage, ax=axes[0, 1], palette=palette, cut=0, inner='quartile')
    axes[0, 1].set(title='Per-sequence routing distribution', xticks=range(num_experts),
                   xticklabels=experts, xlabel='Expert', ylabel='Weight', ylim=(0, 1))

    top = np.argsort(counts)[-top_tokens:][::-1]
    labels = [repr(tokenizer.decode([t]))[1:-1] for t in top]
    sns.heatmap(share[top], ax=axes[1, 0], cmap='mako', vmin=0, vmax=1, annot=num_experts <= 4, fmt='.2f', annot_kws={'size': 7},
                xticklabels=experts, yticklabels=labels, cbar_kws={'label': 'Routing share'})
    axes[1, 0].set(title=f'Routing share of the {top_tokens} most frequent tokens', xlabel='Expert', ylabel='Token')
    axes[1, 0].tick_params(axis='y', labelsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'usage': usage, 'affinity': affinity}

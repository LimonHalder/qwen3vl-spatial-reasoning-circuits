"""All paper figures: layer-wise VAP curves, the head-importance grid, and
the bootstrap Jaccard-overlap heatmap.

Every function takes the already-computed DataFrame/array and an
`out_path`-style filename, and returns the matplotlib Figure so it can also
be displayed inline in a notebook.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg


def plot_layer_curves(layer_summary, token_groups=cfg.TOKEN_GROUPS, save=True):
    """One figure per reasoning level: overwrite rate vs. layer, one line
    per token group. Returns a list of Figures (one per level)."""
    figures = []

    for level in range(1, 6):
        sub = layer_summary[layer_summary["level"] == level]

        fig = plt.figure(figsize=(10, 5))

        for group in token_groups:
            x = sub[sub["group"] == group].sort_values("layer")
            if len(x):
                plt.plot(x["layer"], x["overwrite_rate"], marker="o", label=group)

        plt.xlabel("Decoder layer")
        plt.ylabel("Overwrite rate (%)")
        plt.title(f"Layer-wise VAP — Reasoning Level L{level}")
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(cfg.out_path(f"layer_vap_L{level}.png"), dpi=160)

        figures.append(fig)

    return figures


def plot_head_importance_grid(head_results_df, candidate_layers_by_level, num_heads, save=True):
    """Multi-panel figure: rows = reasoning level, columns = token group,
    shared colorbar and color scale — the head-level causal-importance
    heatmap for the paper."""
    all_groups = sorted({g for lv in candidate_layers_by_level.values() for g in lv})
    n_rows, n_cols = 5, len(all_groups)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), squeeze=False)
    vmax = head_results_df["importance"].abs().quantile(0.98)
    im = None

    for r, level in enumerate(range(1, 6)):
        level_groups = candidate_layers_by_level[level]
        for c, group in enumerate(all_groups):
            ax = axes[r][c]
            layers = level_groups.get(group, [])
            if not layers:
                ax.axis("off")
                continue

            sub = head_results_df[
                (head_results_df["level"] == level) & (head_results_df["group"] == group)
            ]
            matrix = np.full((len(layers), num_heads), np.nan)
            for i, layer in enumerate(layers):
                for _, row in sub[sub["layer"] == int(layer)].iterrows():
                    matrix[i, int(row["head"])] = row["importance"]

            im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers, fontsize=6)
            ax.set_xticks([])
            if r == 0:
                ax.set_title(group, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"L{level}", fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="Causal importance (Δ logit-gap)")
    fig.suptitle("Head-level causal importance across reasoning levels and token groups", y=1.01)

    if save:
        fig.savefig(cfg.out_path("head_importance_grid.pdf"), dpi=300, bbox_inches="tight")

    return fig


def plot_overlap_heatmap(overlap_mean, overlap_lo, overlap_hi, top_k_heads=cfg.TOP_K_HEADS, n_boot=cfg.N_BOOT, save=True):
    """5x5 Jaccard-overlap heatmap with bootstrap 95% CI annotations."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(overlap_mean, vmin=0, vmax=1, cmap="viridis")

    for i in range(5):
        for j in range(5):
            ax.text(
                j, i,
                f"{overlap_mean[i, j]:.2f}\n[{overlap_lo[i, j]:.2f},{overlap_hi[i, j]:.2f}]",
                ha="center", va="center", fontsize=6, color="white",
            )

    ax.set_xticks(range(5)); ax.set_xticklabels([f"L{x}" for x in range(1, 6)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"L{x}" for x in range(1, 6)])
    ax.set_title(f"Top-{top_k_heads} head Jaccard overlap (bootstrap 95% CI, B={n_boot})")
    fig.colorbar(im, label="Jaccard index")
    fig.tight_layout()

    if save:
        fig.savefig(cfg.out_path("head_overlap_bootstrap.pdf"), dpi=300)

    return fig

"""Candidate-layer selection, Jaccard overlap with bootstrap CIs, and the
core-vs-difficulty-specific circuit summary table.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Candidate layer selection (feeds the head-level VAP sweep)
# ---------------------------------------------------------------------------
def select_candidate_layers(layer_summary, top_layers_per_level=cfg.TOP_LAYERS_PER_LEVEL, verbose=True):
    """Rank layers per (level, token group) by overwrite_rate then
    mean_gap_closure, and keep the top `top_layers_per_level`.

    Returns {level: {group: [layer ids]}}.
    """
    candidate_layers_by_level = {}

    for level in range(1, 6):
        sub = layer_summary[layer_summary["level"] == level].copy()
        candidate_layers_by_level[level] = {}

        for group in sorted(sub["group"].dropna().unique()):
            group_sub = sub[sub["group"] == group].copy()

            top = (
                group_sub.sort_values(
                    ["overwrite_rate", "mean_gap_closure"], ascending=[False, False]
                ).head(top_layers_per_level)
            )

            candidate_layers_by_level[level][group] = top["layer"].astype(int).tolist()

            if verbose:
                print(f"L{level} | {group}: {candidate_layers_by_level[level][group]}")

    return candidate_layers_by_level


def save_candidate_layers(candidate_layers_by_level, path):
    with open(path, "w") as f:
        json.dump(candidate_layers_by_level, f, indent=2)


# ---------------------------------------------------------------------------
# Top heads / Jaccard overlap
# ---------------------------------------------------------------------------
def top_heads(df, level, k=cfg.TOP_K_HEADS):
    """Set of (layer, head) tuples ranked by mean importance for `level`."""
    sub = df[df["level"] == level].sort_values("importance", ascending=False).head(k)
    return set(zip(sub["layer"].astype(int), sub["head"].astype(int)))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def bootstrap_head_overlap(head_df, head_results_df, top_k=cfg.TOP_K_HEADS, n_boot=cfg.N_BOOT, seed=cfg.SEED):
    """Compute pairwise Jaccard overlap between each level's top-k heads,
    with a bootstrap (resampling scenes with replacement) 95% CI.

    Returns:
        top_heads_by_level: {level: set of (layer, head)}
        overlap_mean:        5x5 np.ndarray
        overlap_lo, overlap_hi: 5x5 np.ndarray (2.5th / 97.5th percentiles)
    """
    rng = np.random.default_rng(seed)

    top_heads_by_level = {lv: top_heads(head_results_df, lv, k=top_k) for lv in range(1, 6)}

    sample_ids = head_df["sample_id"].unique()
    boot_overlap = np.zeros((n_boot, 5, 5))

    for b in range(n_boot):
        boot_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        boot_summary = (
            head_df[head_df["sample_id"].isin(boot_ids)]
            .groupby(["level", "layer", "head"])["importance"].mean()
            .reset_index()
        )
        boot_top = {lv: top_heads(boot_summary, lv, k=top_k) for lv in range(1, 6)}

        for i, a in enumerate(range(1, 6)):
            for j, bb in enumerate(range(1, 6)):
                boot_overlap[b, i, j] = jaccard(boot_top[a], boot_top[bb])

    overlap_mean = boot_overlap.mean(axis=0)
    overlap_lo, overlap_hi = np.percentile(boot_overlap, [2.5, 97.5], axis=0)

    return top_heads_by_level, overlap_mean, overlap_lo, overlap_hi


def overlap_matrix_to_df(overlap_mean):
    return pd.DataFrame(
        overlap_mean,
        index=[f"L{x}" for x in range(1, 6)],
        columns=[f"L{x}" for x in range(1, 6)],
    )


# ---------------------------------------------------------------------------
# Core vs. difficulty-specific head summary table
# ---------------------------------------------------------------------------
def build_circuit_summary_table(top_heads_by_level):
    """Which heads are common across all reasoning levels ("core circuit")
    vs. specific to a subset of levels.

    Returns a DataFrame with columns: layer, head, n_levels, levels, category.
    """
    head_counts = Counter(h for heads in top_heads_by_level.values() for h in heads)

    rows = []
    for (layer, head), count in head_counts.items():
        levels_present = [lv for lv, heads in top_heads_by_level.items() if (layer, head) in heads]
        rows.append({
            "layer": layer,
            "head": head,
            "n_levels": count,
            "levels": ",".join(f"L{l}" for l in levels_present),
            "category": (
                "core (>=4/5)" if count >= 4
                else "shared (>=3/5)" if count >= 3
                else "level-specific"
            ),
        })

    circuit_table = (
        pd.DataFrame(rows)
        .sort_values(["n_levels", "layer", "head"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return circuit_table


def save_circuit_summary(circuit_table, top_heads_by_level, csv_path, tex_path, json_path):
    circuit_table.to_csv(csv_path, index=False)

    with open(tex_path, "w") as f:
        f.write(circuit_table.to_latex(
            index=False,
            caption=(
                "Attention heads causally implicated (top-10 by importance) at each "
                "reasoning level, and how many of the five levels each head appears in."
            ),
            label="tab:circuit",
        ))

    with open(json_path, "w") as f:
        json.dump({
            "top_heads_by_level": {str(lv): sorted(list(h)) for lv, h in top_heads_by_level.items()},
            "core_heads": circuit_table[circuit_table["category"] == "core (>=4/5)"][["layer", "head"]].values.tolist(),
        }, f, indent=2)

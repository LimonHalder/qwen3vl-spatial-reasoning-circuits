"""Causal-patching core: baseline evaluation, layer-level VAP (vocabulary
activation patching), and head-level VAP.

All sweep functions take an explicit `ctx` bundle (model, processor, device,
decoder_layers, image_token_id, ...) rather than reading module-level
globals, so they're safe to import and call from a notebook or a script.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import config as cfg
from . import model_utils as mu
from .benchmark import scene_to_positions, make_image


@dataclass
class ModelContext:
    """Everything the patching sweeps need, bundled after model load."""
    model: object
    processor: object
    device: str
    decoder_layers: object
    num_layers: int
    image_token_id: int
    num_heads: int = field(default=0)
    head_dim: int = field(default=0)
    hidden_size: int = field(default=0)

    @classmethod
    def build(cls, model, processor, device=cfg.DEVICE, sample_inputs=None):
        decoder_layers = mu.get_decoder_layers(model)
        image_token_id, _ = mu.resolve_image_token_id(processor, sample_inputs=sample_inputs)

        text_cfg = mu.get_text_config(model)
        num_heads = int(text_cfg.num_attention_heads)
        hidden_size = int(text_cfg.hidden_size)
        head_dim = int(getattr(text_cfg, "head_dim", hidden_size // num_heads))

        return cls(
            model=model,
            processor=processor,
            device=device,
            decoder_layers=decoder_layers,
            num_layers=len(decoder_layers),
            image_token_id=image_token_id,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )

    def build_inputs(self, image, question):
        return mu.build_inputs(self.processor, self.device, image, question)

    def token_groups(self, input_ids_row, system_text, question_text):
        return mu.token_groups(self.processor, self.image_token_id, input_ids_row, system_text, question_text)


# ---------------------------------------------------------------------------
# Shared forward / decode helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_logits(ctx: ModelContext, inputs):
    return ctx.model(**inputs, use_cache=False).logits[0, -1, :]


def pred_yes_no(ctx: ModelContext, logits):
    tok = ctx.processor.tokenizer
    token_id = logits.argmax().item()
    decoded = tok.decode([token_id]).strip().lower()

    if decoded.startswith("yes"):
        return "Yes", token_id
    if decoded.startswith("no"):
        return "No", token_id
    return decoded, token_id


# ---------------------------------------------------------------------------
# 1. Baseline evaluation
# ---------------------------------------------------------------------------
def run_baseline_evaluation(ctx: ModelContext, benchmark):
    """Check the model distinguishes clean vs. corrupted scenes at each level.

    Returns (baseline_df, summary_df).
    """
    rows = []

    for scene in benchmark:
        clean_pos = scene_to_positions(scene, corrupt=False)

        for q_index, q in enumerate(scene["queries"]):
            corrupt_pos = scene_to_positions(scene, corrupt=True, query_index=q_index)

            clean_inputs = ctx.build_inputs(make_image(clean_pos), q["question"])
            corrupt_inputs = ctx.build_inputs(make_image(corrupt_pos), q["question"])

            clean_logits = forward_logits(ctx, clean_inputs)
            corrupt_logits = forward_logits(ctx, corrupt_inputs)

            clean_pred, _ = pred_yes_no(ctx, clean_logits)
            corrupt_pred, _ = pred_yes_no(ctx, corrupt_logits)

            rows.append({
                "sample_id": scene["sample_id"],
                "level": q["level"],
                "gt_clean": q["clean_answer"],
                "gt_corrupt": q["corrupted_answer"],
                "pred_clean": clean_pred,
                "pred_corrupt": corrupt_pred,
                "clean_correct": clean_pred == q["clean_answer"],
                "corrupt_correct": corrupt_pred == q["corrupted_answer"],
                "model_distinguishes": (
                    clean_pred in {"Yes", "No"}
                    and corrupt_pred in {"Yes", "No"}
                    and clean_pred != corrupt_pred
                ),
            })

    baseline_df = pd.DataFrame(rows)

    summary = (
        baseline_df.groupby("level")
        .agg(
            clean_accuracy=("clean_correct", "mean"),
            corrupt_accuracy=("corrupt_correct", "mean"),
            distinguish_rate=("model_distinguishes", "mean"),
        )
        .reset_index()
    )
    summary["clean_accuracy"] *= 100
    summary["corrupt_accuracy"] *= 100
    summary["distinguish_rate"] *= 100

    return baseline_df, summary


# ---------------------------------------------------------------------------
# 2. Layer x token-group VAP
# ---------------------------------------------------------------------------
class LayerPatcher:
    """Forward-hook context manager that overwrites a decoder layer's output
    hidden states at `token_positions` with `replacement_values`."""

    def __init__(self, layer_module, token_positions, replacement_values):
        self.layer_module = layer_module
        self.token_positions = token_positions
        self.replacement_values = replacement_values
        self.handle = None

    def __enter__(self):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            for pos, val in zip(self.token_positions, self.replacement_values):
                hs[0, pos, :] = val.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        self.handle = self.layer_module.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()


@torch.no_grad()
def _run_and_cache(ctx: ModelContext, inputs):
    out = ctx.model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states, out.logits


@torch.no_grad()
def _patched_forward_layer(ctx: ModelContext, clean_inputs, layer_idx, token_positions, corrupt_hidden):
    replacement_values = corrupt_hidden[token_positions]
    with LayerPatcher(ctx.decoder_layers[layer_idx], token_positions, replacement_values):
        out = ctx.model(**clean_inputs, use_cache=False)
    return out.logits[0, -1, :]


def run_layer_vap_sweep(ctx: ModelContext, benchmark, token_groups=cfg.TOKEN_GROUPS, verbose=True):
    """Patch decoder-layer hidden states at each token group and measure how
    much the clean answer shifts toward the corrupted answer.

    Returns (layer_df, layer_summary_df). `layer_summary_df` has one row per
    (level, group, layer) with overwrite_rate (%) and mean_gap_closure.
    """
    records = []

    for scene in benchmark:
        clean_pos = scene_to_positions(scene, corrupt=False)

        for q_index, q in enumerate(scene["queries"]):
            corrupt_pos = scene_to_positions(scene, corrupt=True, query_index=q_index)

            clean_inputs = ctx.build_inputs(make_image(clean_pos), q["question"])
            corrupt_inputs = ctx.build_inputs(make_image(corrupt_pos), q["question"])

            if clean_inputs["input_ids"].shape != corrupt_inputs["input_ids"].shape:
                continue

            clean_hs, clean_logits = _run_and_cache(ctx, clean_inputs)
            corrupt_hs, corrupt_logits = _run_and_cache(ctx, corrupt_inputs)

            clean_pred, clean_tok = pred_yes_no(ctx, clean_logits[0, -1, :])
            corrupt_pred, corrupt_tok = pred_yes_no(ctx, corrupt_logits[0, -1, :])

            if clean_pred not in {"Yes", "No"} or corrupt_pred not in {"Yes", "No"} or clean_pred == corrupt_pred:
                continue

            groups = ctx.token_groups(clean_inputs["input_ids"][0], cfg.SYSTEM_PROMPT, q["question"])

            base_gap = (
                clean_logits[0, -1, clean_tok] - clean_logits[0, -1, corrupt_tok]
            ).item()

            for layer_idx in range(ctx.num_layers):
                corrupt_hidden = corrupt_hs[layer_idx + 1][0]

                for group_name in token_groups:
                    positions = groups[group_name]
                    if not positions:
                        continue

                    patched_logits = _patched_forward_layer(
                        ctx, clean_inputs, layer_idx, positions, corrupt_hidden
                    )
                    patched_pred, _ = pred_yes_no(ctx, patched_logits)
                    patched_gap = (patched_logits[clean_tok] - patched_logits[corrupt_tok]).item()

                    records.append({
                        "sample_id": scene["sample_id"],
                        "level": q["level"],
                        "layer": layer_idx,
                        "group": group_name,
                        "overwrite": int(patched_pred == corrupt_pred),
                        "gap_closure": base_gap - patched_gap,
                    })

            if verbose:
                print(f"Completed sample {scene['sample_id']} level {q['level']}")

    layer_df = pd.DataFrame(records)

    layer_summary = (
        layer_df.groupby(["level", "group", "layer"])
        .agg(
            overwrite_rate=("overwrite", "mean"),
            mean_gap_closure=("gap_closure", "mean"),
            n=("overwrite", "size"),
        )
        .reset_index()
    )
    layer_summary["overwrite_rate"] *= 100

    return layer_df, layer_summary


# ---------------------------------------------------------------------------
# 3. Head-level VAP
# ---------------------------------------------------------------------------
class HeadPatcher:
    """Forward-pre-hook context manager that overwrites a single attention
    head's pre-o_proj activations at `token_positions`."""

    def __init__(self, attn_module, head_idx, head_dim, token_positions, replacement_values):
        self.attn_module = attn_module
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.token_positions = token_positions
        self.replacement_values = replacement_values
        self.handle = None

    def __enter__(self):
        start, end = self.head_idx * self.head_dim, (self.head_idx + 1) * self.head_dim

        def pre_hook(module, args):
            x = args[0]
            shape = x.shape
            if x.dim() == 4:
                b, s, nh, hd = x.shape
                x = x.reshape(b, s, nh * hd)
            x = x.clone()
            for pos, val in zip(self.token_positions, self.replacement_values):
                x[0, pos, start:end] = val.to(x.dtype)
            if len(shape) == 4:
                x = x.reshape(shape)
            return (x,) + args[1:]

        self.handle = self.attn_module.o_proj.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


@torch.no_grad()
def _capture_preoproj_all_layers(ctx: ModelContext, inputs):
    captured, handles = {}, []

    for layer_idx, layer in enumerate(ctx.decoder_layers):
        attn = mu.get_attn_module(layer)

        def make_hook(li):
            def hook(module, inp, out):
                x = inp[0].detach()
                if x.dim() == 4:
                    b, s, nh, hd = x.shape
                    x = x.reshape(b, s, nh * hd)
                captured[li] = x[0]
            return hook

        handles.append(attn.o_proj.register_forward_hook(make_hook(layer_idx)))

    ctx.model(**inputs, use_cache=False)

    for h in handles:
        h.remove()

    return captured


@torch.no_grad()
def _patched_forward_head(ctx: ModelContext, clean_inputs, layer_idx, head_idx, token_positions, corrupt_preoproj):
    start, end = head_idx * ctx.head_dim, (head_idx + 1) * ctx.head_dim
    replacement_values = corrupt_preoproj[token_positions, start:end]
    attn = mu.get_attn_module(ctx.decoder_layers[layer_idx])
    with HeadPatcher(attn, head_idx, ctx.head_dim, token_positions, replacement_values):
        out = ctx.model(**clean_inputs, use_cache=False)
    return out.logits[0, -1, :]


def run_head_vap_sweep(ctx: ModelContext, benchmark, candidate_layers_by_level, verbose=True):
    """Causal patch sweep over individual attention heads, restricted to the
    candidate layers selected per (level, token group).

    Returns (head_df, head_results_df). `head_results_df` is aggregated to
    one row per (level, group, layer, head).
    """
    head_records = []
    skipped_missing_group = 0

    for level in range(1, 6):
        for group, candidate_layers in candidate_layers_by_level[level].items():
            if not candidate_layers:
                continue

            iterator = tqdm(benchmark, desc=f"L{level}|{group}", leave=False) if verbose else benchmark

            for scene in iterator:
                q_index = level - 1
                q = scene["queries"][q_index]

                clean_pos = scene_to_positions(scene, corrupt=False)
                corrupt_pos = scene_to_positions(scene, corrupt=True, query_index=q_index)
                clean_inputs = ctx.build_inputs(make_image(clean_pos), q["question"])
                corrupt_inputs = ctx.build_inputs(make_image(corrupt_pos), q["question"])

                clean_logits = forward_logits(ctx, clean_inputs)
                corrupt_logits = forward_logits(ctx, corrupt_inputs)
                clean_pred, clean_tok = pred_yes_no(ctx, clean_logits)
                corrupt_pred, corrupt_tok = pred_yes_no(ctx, corrupt_logits)

                if clean_pred not in {"Yes", "No"} or corrupt_pred not in {"Yes", "No"} or clean_pred == corrupt_pred:
                    continue

                groups = ctx.token_groups(clean_inputs["input_ids"][0], cfg.SYSTEM_PROMPT, q["question"])
                positions = groups.get(group, [])
                if not positions:
                    skipped_missing_group += 1
                    continue

                base_gap = (clean_logits[clean_tok] - clean_logits[corrupt_tok]).item()
                corrupt_preoproj = _capture_preoproj_all_layers(ctx, corrupt_inputs)

                for layer_idx in candidate_layers:
                    layer_idx = int(layer_idx)
                    if layer_idx not in corrupt_preoproj:
                        continue

                    for head_idx in range(ctx.num_heads):
                        patched_logits = _patched_forward_head(
                            ctx, clean_inputs, layer_idx, head_idx, positions, corrupt_preoproj[layer_idx]
                        )
                        patched_gap = (patched_logits[clean_tok] - patched_logits[corrupt_tok]).item()
                        patched_pred, _ = pred_yes_no(ctx, patched_logits)
                        gap_closure = (base_gap - patched_gap) / base_gap if abs(base_gap) > 1e-12 else np.nan

                        head_records.append({
                            "level": level,
                            "group": group,
                            "sample_id": scene["sample_id"],
                            "layer": layer_idx,
                            "head": head_idx,
                            "importance": base_gap - patched_gap,
                            "overwrite": int(patched_pred == corrupt_pred),
                            "gap_closure": gap_closure,
                        })

    if verbose:
        print(f"Sweep complete. {skipped_missing_group} (level, group, sample) rows skipped — group absent from prompt.")

    head_df = pd.DataFrame(head_records)

    head_results_df = (
        head_df.groupby(["level", "group", "layer", "head"])
        .agg(
            importance=("importance", "mean"),
            overwrite_rate=("overwrite", "mean"),
            mean_gap_closure=("gap_closure", "mean"),
            n=("importance", "size"),
        )
        .reset_index()
    )

    return head_df, head_results_df

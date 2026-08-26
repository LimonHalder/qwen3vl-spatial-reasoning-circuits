"""Optional HeadLens: project a selected attention head's isolated residual
contribution onto the vocabulary space for qualitative inspection.

This is *not* a causal or probabilistic measure — it just shows which
vocabulary directions a head's output aligns with under the final norm +
LM head. Use it only after the causal (VAP) sweep has already told you which
heads matter.
"""

import torch

from . import model_utils as mu
from .patching import ModelContext


@torch.no_grad()
def headlens_decode(ctx: ModelContext, lm_head, final_norm, inputs, layer_idx, head_idx, token_position, top_k=10):
    """Return the top_k (token_string, logit) pairs a given head's residual
    contribution points toward, at `token_position` in `inputs`.
    """
    tok = ctx.processor.tokenizer
    attn = mu.get_attn_module(ctx.decoder_layers[layer_idx])

    captured = {}

    def hook(module, inp, out):
        x = inp[0].detach()
        if x.dim() == 4:
            b, s, nh, hd = x.shape
            x = x.reshape(b, s, nh * hd)
        captured["x"] = x[0]

    handle = attn.o_proj.register_forward_hook(hook)
    ctx.model(**inputs, use_cache=False)
    handle.remove()

    start = head_idx * ctx.head_dim
    end = (head_idx + 1) * ctx.head_dim

    x_h = captured["x"][token_position, start:end]
    w_h = attn.o_proj.weight[:, start:end]

    contribution = w_h @ x_h
    v = contribution.unsqueeze(0)

    if final_norm is not None:
        v = final_norm(v)

    logits = lm_head(v)[0]
    values, indices = logits.topk(top_k)

    return [
        (tok.decode([idx]).strip(), round(float(value), 3))
        for idx, value in zip(indices.tolist(), values.tolist())
    ]


def run_headlens_for_scene(ctx: ModelContext, scene, head_results_df, top_n_heads=5, top_k=10):
    """Convenience wrapper: for each reasoning level, decode the top
    `top_n_heads` (by importance) heads at the answer-generating position.

    Returns {level: [(layer, head, importance, decoded), ...]}.
    """
    from .benchmark import scene_to_positions, make_image
    from . import config as cfg

    lm_head = mu.get_lm_head(ctx.model)
    final_norm = mu.get_final_norm(ctx.model)

    clean_pos = scene_to_positions(scene, corrupt=False)
    results = {}

    for level in range(1, 6):
        q = scene["queries"][level - 1]
        inputs = ctx.build_inputs(make_image(clean_pos), q["question"])
        groups = ctx.token_groups(inputs["input_ids"][0], cfg.SYSTEM_PROMPT, q["question"])
        answer_position = groups["last_prompt_token"][0]

        sub = (
            head_results_df[head_results_df["level"] == level]
            .sort_values("importance", ascending=False)
            .head(top_n_heads)
        )

        level_results = []
        for _, row in sub.iterrows():
            layer, head = int(row["layer"]), int(row["head"])
            decoded = headlens_decode(ctx, lm_head, final_norm, inputs, layer, head, answer_position, top_k=top_k)
            level_results.append((layer, head, float(row["importance"]), decoded))

        results[level] = level_results

    return results

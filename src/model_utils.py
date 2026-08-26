"""Qwen3-VL loading, chat-template input construction, and token-group
localization (which input positions correspond to the image, the system
prompt, the instruction text, etc).
"""

from collections import Counter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from . import config as cfg

SYSTEM_PROMPT = cfg.SYSTEM_PROMPT

_CANDIDATE_IMAGE_TOKENS = [
    "<|image_pad|>",
    "<|vision_pad|>",
    "<image>",
    "<|img_pad|>",
]


def get_decoder_layers(model):
    for path in [
        "model.language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not locate decoder layers. Run print(model) and update get_decoder_layers().")


def get_attn_module(layer):
    for attr in ["self_attn", "attn", "attention"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError("Could not find attention submodule.")


def get_text_config(model):
    cfg_obj = getattr(model, "config", None)
    for attr in ["text_config", "llm_config", "language_config"]:
        if cfg_obj is not None and hasattr(cfg_obj, attr):
            return getattr(cfg_obj, attr)
    return cfg_obj


def get_lm_head(model):
    for path in ["lm_head", "model.lm_head", "language_model.lm_head"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Could not find lm_head.")


def get_final_norm(model):
    for path in [
        "model.language_model.norm",
        "language_model.model.norm",
        "model.model.norm",
        "model.norm",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def load_model_and_processor(model_id=cfg.MODEL_ID, device=cfg.DEVICE):
    """Load Qwen3-VL in bfloat16 and its processor. Returns (model, processor)."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_inputs(processor, device, image, question, system=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in inputs.items()}


def resolve_image_token_id(processor, sample_inputs=None):
    """Find the input_ids token id used for image patches.

    Tries a list of known special-token strings first; if none resolve,
    falls back to the most frequent token id in `sample_inputs` (the image
    placeholder is by far the most repeated token in a VLM prompt).
    """
    tok = processor.tokenizer

    for cand in _CANDIDATE_IMAGE_TOKENS:
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != tok.unk_token_id:
            return tid, cand

    if sample_inputs is None:
        raise ValueError(
            "No known image-token string matched; pass sample_inputs to fall back "
            "on the most-frequent-token heuristic."
        )

    counts = Counter(sample_inputs["input_ids"][0].tolist())
    fallback_id = counts.most_common(1)[0][0]
    return fallback_id, tok.decode([fallback_id])


def find_subseq(seq, sub):
    if not sub:
        return []
    L = len(sub)
    for i in range(len(seq) - L + 1):
        if seq[i:i + L] == sub:
            return list(range(i, i + L))
    return []


def token_groups(processor, image_token_id, input_ids_row, system_text, question_text):
    """Return {group_name: [token positions]} for a single tokenized prompt."""
    tok = processor.tokenizer

    ids = input_ids_row.tolist() if torch.is_tensor(input_ids_row) else list(input_ids_row)
    n = len(ids)

    image_idx = [i for i, token_id in enumerate(ids) if token_id == image_token_id]
    last_image_idx = [image_idx[-1] + 1] if image_idx and image_idx[-1] + 1 < n else []
    last_prompt_idx = [n - 1]

    sys_ids = tok(system_text, add_special_tokens=False)["input_ids"]
    q_ids = tok(question_text, add_special_tokens=False)["input_ids"]

    return {
        "image_tokens": image_idx,
        "instruction_tokens": find_subseq(ids, q_ids),
        "system_tokens": find_subseq(ids, sys_ids),
        "last_image_token": last_image_idx,
        "last_prompt_token": last_prompt_idx,
    }

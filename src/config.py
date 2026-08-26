"""Global constants for the Qwen3-VL spatial reasoning circuit benchmark.

Everything that used to be scattered across notebook cells as bare globals
(SEED, MODEL_ID, grid geometry, sweep hyperparameters, output paths) lives
here so every other module in `src/` imports it from one place.
"""

import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility / device / model
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def set_seed(seed: int = SEED) -> None:
    """Seed python/numpy/torch RNGs. Call once at the top of the driver notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Scene / grid parameters
# ---------------------------------------------------------------------------
IMG_SIZE = 336
CELL_SIZE = 84
GRID_W, GRID_H = 4, 3
SHAPE_SIZE = 34

COLORS = ["red", "blue", "green", "orange", "purple", "yellow"]
SHAPES = ["circle", "square"]
OBJECT_DESCRIPTORS = [(c, s) for c in COLORS for s in SHAPES]  # 12 unique objects
GRID_POSITIONS = [(x, y) for y in range(GRID_H) for x in range(GRID_W)]

REL_DXDY = {
    "left": (-1, 0),
    "right": (1, 0),
    "above": (0, -1),
    "below": (0, 1),
}

# Each level specifies where the relevant roles are in the clean scene.
# For odd sample IDs we use the "Yes" polarity; for even IDs the final
# relation is false.
QUERY_POSITIONS = {
    1: {
        "yes_subject": (0, 1), "no_subject": (3, 1),
        "anchor": (3, 1), "no_anchor": (0, 1),
        "steps": [], "final_rel": "left",
    },
    2: {
        "yes_subject": (2, 0), "no_subject": (2, 2),
        "anchor": (2, 2), "no_anchor": (2, 0),
        "steps": [], "final_rel": "above",
    },
    3: {
        "yes_subject": (0, 1), "no_subject": (2, 1),
        "anchor": (1, 2),
        "steps": ["above"], "final_rel": "left",
    },
    4: {
        "yes_subject": (0, 1), "no_subject": (2, 1),
        "anchor": (0, 2),
        "steps": ["right", "above"], "final_rel": "left",
    },
    5: {
        "yes_subject": (0, 0), "no_subject": (3, 0),
        "anchor": (1, 0),
        "steps": ["below", "right", "above"], "final_rel": "left",
    },
}

SYSTEM_PROMPT = "You are a helpful assistant."

TOKEN_GROUPS = [
    "image_tokens",
    "instruction_tokens",
    "system_tokens",
    "last_image_token",
    "last_prompt_token",
]

# ---------------------------------------------------------------------------
# Sweep hyperparameters
# ---------------------------------------------------------------------------
TOP_LAYERS_PER_LEVEL = 4   # candidate decoder layers carried into the head sweep
TOP_K_HEADS = 10           # heads used for Jaccard-overlap / circuit-summary analysis
N_BOOT = 200                # bootstrap resamples for the overlap confidence intervals

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
# Defaults to ./data (see README). Override with the QWEN3VL_OUTPUT_DIR env
# var, e.g. to point back at /kaggle/working on Kaggle.
OUTPUT_DIR = os.environ.get(
    "QWEN3VL_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out_path(filename: str) -> str:
    """Resolve `filename` against OUTPUT_DIR, creating the dir if needed."""
    return os.path.join(OUTPUT_DIR, filename)

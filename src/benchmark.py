"""4x3 grid scene + 5-level query generation, matched corruption search, and
save/load helpers for the spatial reasoning benchmark.

Public entry points:
    build_benchmark(n_samples, seed)   -> list[scene dict]
    save_benchmark(benchmark, path)
    load_benchmark(path)               -> list[scene dict]
    make_image(obj_positions)          -> PIL.Image
    scene_to_positions(scene, corrupt, query_index)
"""

import itertools
import json
import random

from PIL import Image, ImageDraw

from . import config as cfg

IMG_SIZE = cfg.IMG_SIZE
CELL_SIZE = cfg.CELL_SIZE
GRID_W, GRID_H = cfg.GRID_W, cfg.GRID_H
SHAPE_SIZE = cfg.SHAPE_SIZE
COLORS = cfg.COLORS
SHAPES = cfg.SHAPES
OBJECT_DESCRIPTORS = cfg.OBJECT_DESCRIPTORS
GRID_POSITIONS = cfg.GRID_POSITIONS
REL_DXDY = cfg.REL_DXDY
QUERY_POSITIONS = cfg.QUERY_POSITIONS


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_shape(draw, shape, color, cx, cy, size=SHAPE_SIZE):
    h = size // 2
    if shape == "circle":
        draw.ellipse([cx - h, cy - h, cx + h, cy + h], fill=color)
    else:
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=color)


def make_image(obj_positions):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    for (color, shape), (gx, gy) in obj_positions.items():
        cx = gx * CELL_SIZE + CELL_SIZE // 2
        cy = gy * CELL_SIZE + CELL_SIZE // 2
        draw_shape(draw, shape, color, cx, cy)
    return img


def scene_to_positions(scene, corrupt=False, query_index=0):
    """Return {(color, shape): (gx, gy)} for the clean or corrupted layout."""
    if corrupt:
        source = scene["queries"][query_index]["corrupt_objects"]
    else:
        source = scene["objects"]

    return {
        (item["color"], item["shape"]): tuple(item["position"])
        for item in source
    }


# ---------------------------------------------------------------------------
# Query evaluation
# ---------------------------------------------------------------------------
def pos_to_obj(obj_positions):
    return {p: obj for obj, p in obj_positions.items()}


def eval_query(obj_positions, query_record):
    p2obj = pos_to_obj(obj_positions)
    subject = query_record["subject"]
    anchor = query_record["anchor"]

    if subject not in obj_positions or anchor not in obj_positions:
        return None

    cur = anchor

    for rel in query_record["steps"]:
        x, y = obj_positions[cur]
        dx, dy = REL_DXDY[rel]
        nxt = (x + dx, y + dy)
        if nxt not in p2obj:
            return None
        cur = p2obj[nxt]

    a = obj_positions[subject]
    b = obj_positions[cur]

    return {
        "left": a[0] < b[0],
        "right": a[0] > b[0],
        "above": a[1] < b[1],
        "below": a[1] > b[1],
    }[query_record["final_rel"]]


def object_name(obj):
    return f"{obj[0]} {obj[1]}"


def make_query_text(q):
    if q["level"] == 1:
        return (
            f"Is the {object_name(q['subject'])} to the left of the "
            f"{object_name(q['anchor'])}? Answer with Yes or No only."
        )

    if q["level"] == 2:
        return (
            f"Is the {object_name(q['subject'])} above the "
            f"{object_name(q['anchor'])}? Answer with Yes or No only."
        )

    cur = f"the {object_name(q['anchor'])}"

    for rel in q["steps"]:
        if rel == "above":
            cur = f"the object immediately above {cur}"
        elif rel == "below":
            cur = f"the object immediately below {cur}"
        elif rel == "right":
            cur = f"the object immediately to the right of {cur}"
        elif rel == "left":
            cur = f"the object immediately to the left of {cur}"

    return (
        f"Is the {object_name(q['subject'])} to the left of {cur}? "
        f"Answer with Yes or No only."
    )


# ---------------------------------------------------------------------------
# Corruption search
# ---------------------------------------------------------------------------
def swap_positions(obj_positions, p1, p2):
    inv = {p: obj for obj, p in obj_positions.items()}
    inv[p1], inv[p2] = inv[p2], inv[p1]
    return {obj: p for p, obj in inv.items()}


def find_flipping_corruption(obj_positions, query_record):
    """Find a single pairwise object-location swap that flips the answer."""
    clean = eval_query(obj_positions, query_record)

    for p1, p2 in itertools.combinations(GRID_POSITIONS, 2):
        candidate = swap_positions(obj_positions, p1, p2)
        corrupted = eval_query(candidate, query_record)
        if corrupted is not None and corrupted != clean:
            return (
                candidate,
                {"type": "swap", "positions": [list(p1), list(p2)]},
                clean,
                corrupted,
            )

    raise RuntimeError("Could not find a one-swap corruption.")


def make_query_record(obj_positions, level, polarity_yes=True):
    spec = QUERY_POSITIONS[level]

    if level in (1, 2):
        subject_pos = spec["yes_subject"] if polarity_yes else spec["no_subject"]
        anchor_pos = spec["anchor"] if polarity_yes else spec["no_anchor"]
    else:
        subject_pos = spec["yes_subject"] if polarity_yes else spec["no_subject"]
        anchor_pos = spec["anchor"]

    p2obj = pos_to_obj(obj_positions)

    q = {
        "level": level,
        "subject": p2obj[subject_pos],
        "anchor": p2obj[anchor_pos],
        "steps": list(spec["steps"]),
        "final_rel": spec["final_rel"],
        "reasoning_steps": level,
    }

    q["question"] = make_query_text(q)
    q["ground_truth_clean"] = eval_query(obj_positions, q)
    return q


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------
def build_benchmark(n_samples=10, seed=cfg.SEED):
    """Build `n_samples` scenes, each with clean/corrupted query pairs for L1-L5."""
    rng = random.Random(seed)
    samples = []

    for sample_id in range(1, n_samples + 1):
        objs = OBJECT_DESCRIPTORS[:]
        rng.shuffle(objs)

        clean_positions = {objs[i]: GRID_POSITIONS[i] for i in range(len(objs))}

        scene = {
            "sample_id": sample_id,
            "grid": [GRID_W, GRID_H],
            "objects": [
                {"color": o[0], "shape": o[1], "position": list(clean_positions[o])}
                for o in objs
            ],
            "queries": [],
        }

        for level in range(1, 6):
            polarity_yes = (sample_id % 2 == 1)

            q = make_query_record(clean_positions, level, polarity_yes=polarity_yes)

            corrupt_positions, corruption_meta, clean_gt, corrupt_gt = (
                find_flipping_corruption(clean_positions, q)
            )

            q_record = {
                **q,
                "clean_answer": "Yes" if clean_gt else "No",
                "corrupted_answer": "Yes" if corrupt_gt else "No",
                "corruption": corruption_meta,
                "corrupt_objects": [
                    {"color": o[0], "shape": o[1], "position": list(corrupt_positions[o])}
                    for o in objs
                ],
            }

            scene["queries"].append(q_record)

        samples.append(scene)

    return samples


def benchmark_to_rows(benchmark):
    """Flatten a benchmark into one row per (scene, query) for a DataFrame/CSV."""
    rows = []
    for scene in benchmark:
        for q in scene["queries"]:
            rows.append({
                "sample_id": scene["sample_id"],
                "level": q["level"],
                "reasoning_steps": q["reasoning_steps"],
                "question": q["question"],
                "clean_answer": q["clean_answer"],
                "corrupted_answer": q["corrupted_answer"],
                "corruption": json.dumps(q["corruption"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_benchmark(benchmark, path):
    with open(path, "w") as f:
        json.dump(benchmark, f, indent=2)


def load_benchmark(path):
    with open(path) as f:
        return json.load(f)

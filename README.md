# Qwen3-VL Spatial Reasoning Circuit Benchmark

This repository contains a controlled benchmark and circuit-analysis pipeline for studying how **Qwen3-VL** performs spatial reasoning, using activation patching (VAP) and head-level causal analysis.

The notebook extends earlier VAP/HeadLens code with a **10-scene × 5-query benchmark**. Each scene is evaluated across five reasoning levels:

- **L1 — Direct:** one-step left/right relation
- **L2 — Direct:** one-step above/below relation
- **L3–L5 — Compositional:** multi-hop relations built on top of L1/L2

## 📄 Paper

[https://drive.google.com/file/d/1N8-I3WCZzZBJZakcfJkQkIT_gQ-rNLAp/view?usp=drive_link]

## 📁 Repository Structure

```
qwen3vl-spatial-reasoning-circuits/
├── notebooks/
│   └── qwen3vl_spatial_reasoning_circuit_benchmark.ipynb   # thin driver notebook (imports src/, runs pipeline, plots)
├── src/
│   ├── __init__.py
│   ├── config.py         # global constants: seed, model id, scene/grid params, sweep params, output dir
│   ├── benchmark.py       # 4x3 grid scene + 5-level query generation, corruption search, save/load
│   ├── model_utils.py     # Qwen3-VL loading, chat-template inputs, token-group localization
│   ├── patching.py        # baseline eval, layer-level VAP, head-level VAP (the causal-patching core)
│   ├── analysis.py        # candidate-layer selection, Jaccard overlap + bootstrap CI, circuit tables
│   ├── headlens.py         # optional HeadLens vocabulary-projection decoding
│   └── plotting.py         # all paper figures (layer curves, head-importance grid, overlap heatmap)
├── data/                  # generated CSVs/JSON/figures land here (gitignored except .gitkeep)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

The notebook itself is now just a driver: it calls into `src/` for every step and focuses on running the pipeline top-to-bottom and displaying results/plots. This keeps the reusable logic (benchmark generation, patching, statistics) testable and importable outside the notebook — e.g. `from src.benchmark import build_benchmark`.

## 🧠 What the Pipeline Does

1. **Benchmark design** (`src/benchmark.py`) — builds a 4×3 grid scene with 12 unique objects so spatial relations are unambiguous, and generates matched clean/corrupted query pairs across 5 reasoning levels.
2. **Model loading** (`src/model_utils.py`) — loads Qwen3-VL (`Qwen3VLForConditionalGeneration`) via Hugging Face Transformers.
3. **Baseline evaluation** (`src/patching.py:run_baseline_evaluation`) — checks the model correctly distinguishes clean vs. corrupted scenes before doing causal analysis.
4. **Layer × token-group VAP** (`src/patching.py:run_layer_vap_sweep`) — patches decoder-layer hidden states at different token groups (e.g. last prompt token) and measures how much the answer shifts toward the corrupted answer.
5. **Head-level VAP** (`src/patching.py:run_head_vap_sweep`) — a causal patch sweep over individual attention heads, testing whether the same heads matter across all 5 reasoning levels.
6. **Circuit stability analysis** (`src/analysis.py:bootstrap_head_overlap`) — computes Jaccard overlap (with bootstrap confidence intervals) between the top causal heads at each difficulty level, to see whether the model reuses a shared "circuit" or recruits new heads for harder queries.
7. **Core vs. difficulty-specific heads** (`src/analysis.py:build_circuit_summary_table`) — summarizes which heads are common across all levels ("core circuit") vs. specific to certain reasoning levels, exported as CSV/LaTeX.
8. **Optional HeadLens** (`src/headlens.py`) — projects a selected head's residual contribution onto the vocabulary space for qualitative (non-probabilistic) inspection.
9. **Figures** (`src/plotting.py`) — every plot used in steps 3–7 above, saved to `data/` as PNG/PDF.

## ⚙️ Setup

```bash
git clone <this-repo-url>
cd qwen3vl-spatial-reasoning-circuits
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebooks/qwen3vl_spatial_reasoning_circuit_benchmark.ipynb
```

> **Note:** This notebook expects a GPU (it loads Qwen3-VL in `bfloat16` with `device_map="auto"`). It was originally developed/run on Kaggle; some cells reference Kaggle input paths (`/kaggle/input/...`) for pre-computed CSVs (e.g. `layer_vap_by_difficulty.csv`). Update these paths to point at your own data location before running locally — or just delete those cells, since the driver notebook now recomputes everything via `src/patching.py` rather than reloading a pre-computed CSV. By default all generated files land in `data/` (override with the `QWEN3VL_OUTPUT_DIR` env var, e.g. to point back at `/kaggle/working`).

## 📊 Outputs

Running the full notebook produces:
- Layer-wise VAP curves per reasoning level (L1–L5)
- Head-level causal importance heatmaps/figures
- Jaccard overlap analysis of top circuits across difficulty levels
- A core-vs-difficulty-specific head summary table (CSV + LaTeX)

## 📌 Citation

If you use this benchmark or code, please cite the paper linked above.

## 📝 License

Add a license of your choice (e.g. MIT) here.

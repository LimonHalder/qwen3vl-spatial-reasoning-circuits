"""Qwen3-VL spatial reasoning circuit benchmark — reusable pipeline code.

Modules:
    config      - global constants (seed, model id, scene/grid params, sweep params, output dir)
    benchmark   - 4x3 grid scene + 5-level query generation, corruption search, save/load
    model_utils - Qwen3-VL loading, chat-template inputs, token-group localization
    patching    - baseline eval, layer-level VAP, head-level VAP (the causal-patching core)
    analysis    - candidate-layer selection, Jaccard overlap + bootstrap CI, circuit tables
    headlens    - optional HeadLens vocabulary-projection decoding
    plotting    - all paper figures (layer curves, head-importance grid, overlap heatmap)
"""

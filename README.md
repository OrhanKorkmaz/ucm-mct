# UCM-MCT: A Neutral Interlingua for LLM Hidden States

**Micro Cognitive Translators (MCT)** let frozen, heterogeneous LLMs exchange
hidden states directly — no text round-trip — through a jointly learned
**neutral hub** that no model owns. N models connect with 2N tiny translators
(~1-5M params each) instead of N·(N−1) pairwise bridges.

All results below were produced on a single MacBook (M2, 16 GB, MPS) across
three independent model families: **Llama-3.2-1B**, **Qwen2.5-0.5B**,
**SmolLM2-360M**, over a 10,000-concept bilingual (EN/TR) benchmark.

## Headline results

| Claim | Metric | Result |
|---|---|---|
| Geometry survives translation | Procrustes rank correlation | **ρ = 0.995** |
| Identity survives translation | top-1 retrieval, 10k pool | up to **88%** cross-model |
| Function survives injection | KL-ratio vs random control | **0.17** (agreement 81%) |
| Compression (BatchTopK SAE + aux-k) | top-1 drop @8× | **−3.5 pts**, 0 dead features |
| Depth translation (L2 → hub → L3) | cosine | **0.778** |
| Channel speed vs text round-trip | ms/message | **0.32 vs 286 (~900×)** |
| Third family joins the hub | effect on existing routes | **+15 pts** (positive synergy) |

**A measured limit (new finding):** translation fidelity is
language-dependent. Both models encode Turkish concepts near-perfectly in
isolation (self-consistency ≈ 0.99), yet cross-model relative fidelity is
**EN 1.01 (lossless) vs TR 0.39** — the shared "Platonic" core is thick where
training data overlaps and thin elsewhere.

## How it works

```
model A (frozen)          neutral hub (768d)          model B (frozen)
  layer lA  --E_A-->  jointly trained space   --D_B-->  layer lB
                      InfoNCE + reconstruction
                      + cross-anchor + cycle-consistency
```

- **MCT** = semi-orthogonal core (Procrustes init + soft orthogonality
  penalty) + gated residual MLP, trained on z-standardized activations.
- **Functional alignment**: a tuned-lens-style KL-distillation phase
  (receiver frozen, gradients flow only to the translator) turns
  vector-space closeness into functional compatibility — the single most
  important recipe in this repo.
- **Injection**: single-position (last token) replace with norm calibration.
- **SAE bridge**: BatchTopK + auxiliary dead-feature revival; a message is
  the (index, value) pairs of active features — 64-128 bytes on the wire.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy scipy transformers accelerate safetensors matplotlib wordfreq
bash scripts/run_all.sh                           # extraction -> CKA -> H1-H5 -> report
.venv/bin/python scripts/14_library_fairness.py   # relative-fidelity finding
.venv/bin/python scripts/22_three_family_mesh.py  # 3-family hub
.venv/bin/python scripts/23_latency_bench.py      # 906x speed measurement
.venv/bin/python scripts/24_batchtopk_h4.py       # SAE compression sweep
.venv/bin/python scripts/25_canary_drift.py       # drift canary + closed-form repair
```

Outputs land in `results/*.json` + `results/REPORT.md` (figures included).
Every number in the tables above maps to a JSON produced by a script here.

Hardware notes that will save you a day: Llama needs **bfloat16** (fp16
overflows), right-padding is mandatory (left-pad NaNs propagate through
causal attention), Qwen's massive-activation dimensions require per-dimension
z-standardization, and torch's hard `orthogonal` parametrization breaks on
the first AdamW step — use a soft penalty instead. Gated `meta-llama`
weights are loaded via the identical `unsloth/Llama-3.2-1B` mirror.

## What's intentionally not here (yet)

Ongoing work on cross-model capability grafting, sequence-level translation,
and a subliminal-channel firewall will be released with the next report.

## Citation

```bibtex
@misc{korkmaz2026ucmmct,
  author = {Korkmaz, Orhan},
  title  = {UCM-MCT: A Neutral Interlingua for LLM Hidden States},
  year   = {2026},
  url    = {https://github.com/USERNAME/ucm-mct}
}
```

License: Apache-2.0. Contributions and replications welcome — especially on
other model families and low-resource languages.

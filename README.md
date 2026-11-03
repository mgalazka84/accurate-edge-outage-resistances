# Accurate endpoint resistances after edge outages

Python code and recorded experiments accompanying the research draft
**Componentwise accurate endpoint resistances for all single-edge outages in low-cycle-rank graphs**, by **Marek Gałązka and Hanna Wdowicka**.

The code computes the effective resistance between the endpoints of each original edge after that edge is removed. Each deletion is a separate query on the original graph. Bridges return infinite resistance and a zero leverage-score complement.

This repository contains the implementation, verification programs, synthetic inputs and recorded numerical results. The manuscript and submission correspondence are maintained separately.

## Quick start

Use Python 3.12. The core solver and exact verification require only the standard library.

```bash
python code/outage.py
python code/verify.py
```

The example is a triangle with edge resistances `1, 1e16, 1e16`. Removing its first edge leaves endpoint resistance `2e16` and a leverage-score complement of about `5e-17`.

For the numerical comparisons and figures, install the recorded dependency versions:

```bash
python -m pip install -r code/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/experiments.py
python code/figures.py
```

On Windows, set the same three thread-count environment variables using the shell's syntax before running the experiments. The figure program writes PDFs and PNGs to `manuscript/`, a generated directory excluded from Git.

**Reproduction overwrites the generated files in `results/`.** Keep an untouched checkout when comparing with the recorded measurements. Timings depend on hardware and system load.

## Files

| File or directory | Purpose |
| --- | --- |
| `code/outage.py` | Bridge detection, chain compression, positive elimination and reconstruction |
| `code/baselines.py` | Exact/high-precision references and dense/sparse numerical baselines |
| `code/verify.py` | Exhaustive exact checks, seeded cases and targeted regressions |
| `code/experiments.py` | Accuracy and timing experiments |
| `code/figures.py` | Figures, explanatory diagrams, tables and summary from saved data |
| `code/README.txt` | Detailed input contract and reproduction notes |
| `results/accuracy_inputs.json` | All 100 synthetic graphs with their actual floating-point weights |
| `results/accuracy.csv` | All 1684 accuracy queries and six implementations |
| `results/verification.json` | Exact-validation record |
| `results/precision_checks.json` | Selected reference checks at higher precision |
| `results/timings.csv`, `results/rank_timings.csv` | Recorded runs, including unfavorable dense-graph cases |
| `results/environment.json` | Recorded Python, library, processor and thread settings |

## Recorded checks

- 875 small graphs: 3756 finite outage responses agree with independent exact rational solves, with 1278 disconnected outages classified separately.
- Additional long-cycle and excluded-sum regression examples.
- 100 synthetic accuracy graphs and 1684 finite outage queries, with 160-digit reference arithmetic and selected 240-digit checks.
- Timing experiments include subdivisions up to 98,304 edges and dense graphs where the proposed method is slower.

The stored records describe completed computational checks. They do not establish novelty or replace verification of the mathematical arguments.

## Input and scope

Edges are triples `(u, v, resistance)`, with vertices numbered `0` through `n-1`. The graph must be connected and loop-free, and resistances must be positive and finite. Parallel edges are allowed. The numerical guarantee assumes no overflow or underflow in intermediate operations.

The output concerns the failed edge's own endpoints. It is not an all-pairs solver after each failure, a sequential-update routine, or an alternating-current power-flow model. Series reduction and positive star-mesh elimination are classical ingredients; the draft analyzes their combination and the error of the reconstructed output vector.

## Status and attribution

Research code accompanying an author-review draft, prepared on 3 November 2026. Contact: Marek Gałązka, `galazka@amu.edu.pl`.

An OpenAI language-model assistant supported implementation of the verification, experimental and plotting programs. The draft manuscript and cover letter describe the broader assistance used in preparing the research draft.

No software license has been selected for this repository.

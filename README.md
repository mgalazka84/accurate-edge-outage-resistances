# Accurate endpoint resistances after edge outages

Code and recorded experiments accompanying **Componentwise accurate all-edge outage resistances in graphs with cubic cycle-rank complexity**, by **Marek Gałązka and Hanna Wdowicka**.

For each original edge, the code computes the effective resistance between its endpoints after that edge is removed. Each deletion is a separate query. Bridges return infinite resistance and a zero leverage-score complement. The manuscript and submission correspondence are maintained separately from this public repository.

## Shared positive Schur reductions

`code/joint_outage.py` adds `all_outages_joint(n, edges)`. It shares background
networks among balanced groups of outage queries. Active query edges remain
absent; a child adds sibling edges and positively eliminates vertices outside
its own endpoints. The deterministic arithmetic work is
`O(n + m + sum_i r_i^3)`, with the componentwise error bound proved in the
manuscript. More precisely, a kernel with v vertices and k edges costs
O(k v^2 + k log(k+1)); a complete graph therefore costs O(v^4). The
original `all_outages` remains the independent-query baseline.

Reproduce the added checks and figures with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/joint_experiments.py
python code/joint_figures.py
```

The shared experiments reuse archived inputs and references, and write
`results/joint_*.csv` and `results/joint_*.json`. They include 875 full-graph
rational checks (3756 finite outages), 110 additional multigraphs (2310 exact
kernel queries), 301 parallel-edge reference queries, the original 1684-query
weight suite, 744 rank-sweep queries, and 313 finite IEEE outages. The largest
relative errors in the three floating-point suites are below 1.23e-15. All
reported errors satisfy their pathwise bounds. `joint_figures.py` writes two
PDF figures and a table fragment to `figures/`; these generated files need not
be tracked in the code repository.

The 128-vertex ladder uses 782,008 kernel arithmetic operations with sharing,
versus 43,856,872 for independent positive solves. Recorded complete-profile
medians in the matched HRA run are approximately 0.05180 and 1.918 seconds,
with compressed SF-LDL taking 2.375 seconds on the same inputs. Both counted
kernels use Python scalar arithmetic; SF-LDL uses NumPy array updates.
These are implementation-specific measurements.

Recursive Schur methods have prior art: Durfee et al., STOC 2017
(DOI 10.1145/3055399.3055463), and the active-edge exclusion recursion in
Felix Broberg's 2020 KTH thesis, *Prediction Assisted Fully Dynamic All-Pairs
Effective Resistance*, Chapter 5. The accompanying manuscript analyzes the
cubic deterministic specialization and its full working-precision error bound;
it does not claim the first offline Schur recursion.


## Matched HRA comparison: ladders and complete graphs

Run after the original figure scripts (the last two commands replace Figures 8
and 10 and Table 7):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/hra_comparison.py
python code/positioning_figures.py
python code/joint_figures.py
```

`hra_comparison_timings.csv` compares independent positive elimination, shared
positive elimination and compressed SF-LDL on circular ladders through 128
vertices and complete graphs K4 through K24. The complete-graph runs also
include dense and sparse original-factor updates. All methods use the same
saved inputs, one warm-up and three timed complete-profile runs. Timings exclude
high-precision reference computations; all repetitions are retained.

For K24, Algorithm 1 / Algorithm 2 / SF-LDL medians are 0.1542 / 0.01770 /
0.2655 seconds. Sharing is 15.00 times faster than this SF-LDL implementation,
but 7.07 / 9.86 times slower than dense / sparse updates. On the 128-vertex
ladder it is 45.84 times faster than SF-LDL. Both positive approaches remain
accurate: all 1118 shared outputs satisfy their error bounds, and the maximum
SF-LDL relative error is below 1.52e-15. Complete graphs reuse 746 archived
100-digit direct references; the 372 ladder references use 160-digit updates,
checked by 15 independent 240-digit direct deleted-network solves.

The results include saved inputs, accuracy records, precision checks and
`hra_comparison_environment.json`. The original four-method complete-graph
run remains in `positioning_complete_timings.csv`; its different medians are
not combined with the new run. Figure 8's subdivision panel retains the
original baseline and `positioning_subdivision_timings.csv` measurements.
`positioning_experiments.py` reproduces those archived inputs and references.

The work theorem gives O(k v^2 + k log(k+1)) per kernel, O(v^4) on complete
graphs and O(sum_i r_i^3) in cycle rank. Separate SF-LDL solves cost O(k v^3).
The tested dense factor-and-incidence route also costs O(v^3 + k v^2); the
unbenchmarked full-inverse identity costs O(v^3 + k). Neither asymptotic bound
alone explains the observed constant-factor timing gaps. Depth-first auxiliary
matrix storage is O(v^2 [1 + log_+(p/v)]) for p nonloop kernel edges, plus
O(k) query/input/output storage. Here log_+(x)=max(0,log2(x)).

`hra_sparse_extreme.json` reproduces the largest finite sparse-update error
from Table 3. Its computed denominator is about 0.198 and the true complement
nearly one: this particular error starts in the original-network solve, not
in division by a nearly zero denominator.

`positioning_error_extrema.json` identifies the common error/bound extremum in
Sections 7.3 and 7.8: `theta_s16_j0`, zero-based edge index 6, endpoints (5,1).
Its resistance error/bound is 0.26902461010506 for both methods (E=7).

## Quick start

Use Python 3.12. The core solver needs only the standard library:

```bash
python code/outage.py
```

The example is a triangle with resistances `1, 1e16, 1e16`. Removing the first edge leaves resistance `2e16` and a leverage-score complement of about `5e-17`.

Install the recorded dependencies for the verification scripts, numerical comparisons and plots:

```bash
python -m pip install -r code/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/experiments.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/revision_experiments.py --stage all
python code/figures.py
python code/revision_figures.py
```

The additional stages can be run separately: `validate`, `accuracy`, `rank`, `grids`, or `timings`. They also work with the archived original inputs and results, without rerunning `experiments.py`. No external data download is needed. On Windows, set the thread-count environment variables using the shell's syntax. Plots and tables are generated in `manuscript/`, which is excluded from Git.

**Reproduction overwrites generated files in `results/`.** Preserve an untouched checkout for comparison. Timings depend on hardware and system load; all measured timing repetitions are retained.

## Files

| Path | Purpose |
| --- | --- |
| `code/outage.py` | Bridge detection, chain compression, positive elimination and reconstruction |
| `code/baselines.py` | Original six-method comparison and exact/high-precision references |
| `code/verify.py` | Exhaustive exact checks and targeted regressions |
| `code/experiments.py` | Original accuracy and scaling experiments |
| `code/revision_baselines.py` | Full/compressed subtraction-free LDL, Decimal 35/80 and refinement |
| `code/revision_experiments.py` | Additional validation, accuracy, kernel-rank, IEEE and timing studies |
| `code/figures.py`, `code/revision_figures.py` | Figures, explanatory diagrams, tables and summaries |
| `code/hra_comparison.py` | Matched SF-LDL, shared and conventional timings; extra reference checks |
| `results/hra_*` | Inputs, raw timings, 1118 accuracy checks and sparse-update diagnostic |
| `code/README.txt` | Input contract, reproduction details and failure conventions |
| `results/accuracy_inputs.json`, `results/accuracy.csv` | 100 saved graphs and 1684 queries for the original six methods |
| `results/revision_accuracy.csv` | Six additional methods on those same queries |
| `results/revision_rank*.json`, `results/revision_rank.csv` | 20 graphs and 744 queries with growing kernel size |
| `results/revision_pglib*` | Parsed IEEE inputs, reference checks, accuracy and timings |
| `results/*timings.csv` | Recorded timing repetitions, including unfavorable regimes |
| `results/environment.json`, `results/revision_environment.json` | Actual versions, processor and thread settings |
| `data/pglib/` | Four attributed PGLib IEEE source cases, upstream license and provenance |

## Recorded checks and findings

- Original exact verification: 875 small graphs, 3756 finite queries and 1278 disconnected outages, plus long-cycle and excluded-sum regressions.
- Additional baseline validation: 62 integer-weight graphs and 329 finite queries, including parallel edges and two cycles sharing an anchor.
- Accuracy: twelve implementations on 100 saved graphs and 1684 finite queries, with 160-digit references and selected 240-digit checks.
- Kernel growth: 20 distinct cubic graphs through 64 kernel vertices, 744 finite queries and 60 selected direct 240-digit reference checks.
- PGLib IEEE 14-, 30-, 57- and 118-bus cases: 327 active branches, 14 bridges and 313 finite queries, with 40 selected direct 240-digit checks.
- Scaling includes subdivisions through 98,304 edges and complete graphs where repeated positive elimination is substantially slower.

Both the proposed positive solver and the independently implemented subtraction-free LDL variants maintain high relative accuracy in the extreme-weight suite. Conventional 80-digit direct solves also succeed throughout; 35 digits do not suffice for every extreme case. Sparse binary64 updates work well on all four IEEE cases and are faster on the larger ones. The analytic error bound becomes conservative as the kernel grows. These records support a structural and accuracy-based choice, not universal superiority.

The SF-LDL baseline implements the principle of Orera and Peña, *Mathematics* 12(22), 3491 (2024), https://doi.org/10.3390/math12223491. It is an independent implementation, not the authors' software. The refinement baseline corrects direct deleted-network Cholesky solves; it does not implement the rank-one SMIR method.

## Inputs, data and scope

Edges are `(u, v, resistance)`, with vertices numbered `0` through `n-1`. Inputs must be connected and loop-free, with strictly positive finite resistances. Parallel edges are supported. The theorem assumes no overflow or underflow, including in intermediate arithmetic.

PGLib cases are converted to positive DC sensitivity Laplacians using resistance `x * tap`, with a stored zero tap interpreted as one. Every active branch is retained. All active reactances are positive and all phase shifts zero in these cases. This is a DC sensitivity experiment on benchmark networks, not AC power-flow or measured operational validation. Exact upstream identity, file hashes and adaptation are recorded in `data/pglib/PROVENANCE.json`.

The output concerns each failed edge's own endpoints. It is not an all-pairs outage solver or a sequential-update routine. Series reduction, star-mesh elimination and HRA factorization are established techniques. The manuscript's claim concerns the quantitative bound and cost of the complete all-edge computation after chain reduction.

## Status and attribution

Research materials revised in response to supplied comments on 4 November 2026. Computational checks do not establish novelty or replace the human authors' verification of the arguments. Contact: Marek Gałązka, `galazka@amu.edu.pl`.

An OpenAI language-model assistant supported implementation of the verification, experimental and plotting programs.

The original software and accompanying documentation are licensed under the **MIT License**; see [`LICENSE`](LICENSE). The copied PGLib case data retain their upstream attribution and **CC BY 4.0** license. Third-party files in `data/pglib/` retain their upstream terms, documented in `data/pglib/LICENSE`; the repository-root MIT License does not replace those terms.

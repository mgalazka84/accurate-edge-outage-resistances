VERTEX PARTITIONS
See ../README.md for Algorithm 3, the C++ implementations, commands, data and range diagnostics. The following sections describe the baseline solvers.

REPRODUCIBILITY MATERIAL — FIRST SUBMISSION, 3 NOVEMBER 2026

Componentwise accurate all-edge outage resistances in graphs with cubic cycle-rank complexity — Marek Gałązka and Hanna Wdowicka
https://github.com/mgalazka84/accurate-edge-outage-resistances
Contact: Marek Gałązka, Adam Mickiewicz University, galazka@amu.edu.pl

INPUT CONTRACT
Edge triples are (u, v, RESISTANCE). Vertices are integers 0,...,n-1. The graph
must be connected and loop-free, with strictly positive finite weights. Parallel
edges retain separate identities. Compressed self-loops represent closed chains.
Each edge deletion is separate; the output is the failed edge's own endpoint
resistance and leverage-score complement. Bridges return infinity and zero.
The numerical guarantee excludes overflow, underflow and subnormal intermediate
arithmetic. No scaled or arbitrary-exponent solver is implemented.

REPRODUCTION FROM THE ARCHIVE/REPOSITORY ROOT
Use Python 3.12; recorded dependencies are in code/requirements.txt.
Only outage.py is independent of third-party packages. The verification script
also imports the numerical-baseline module, so install its dependencies first.

    python -m pip install -r code/requirements.txt
    python code/outage.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/verify.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/experiments.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/revision_experiments.py --stage all
    python code/figures.py
    python code/revision_figures.py

On Windows set the three environment variables using the shell's own syntax.
Paths are resolved relative to the scripts. Figures and table fragments are
written to manuscript/. The two plotting scripts read saved data; they do not
rerun numerical experiments. The additional experiment stages can be selected
separately with --stage validate|accuracy|rank|grids|timings. The accuracy stage
uses the archived accuracy_inputs.json and accuracy.csv from the original study.
The rank stage is the most costly of the added reference computations.

Experiments overwrite generated result files: keep an untouched copy for exact
comparison. Timing stages use one warm-up and three measured complete runs;
individual repetitions and medians are retained. Accuracy-stage runtimes are
single diagnostic measurements, not the repeated benchmark used in the paper.
No timings are extrapolated beyond the stated baseline size caps.

METHODS
The six original implementations are compressed/full positive elimination,
compressed/full Cholesky, dense original-factor endpoint update and sparse LU
endpoint update. Six additions are full/compressed subtraction-free LDL,
conventional Decimal 35/80 direct solves, a Decimal 35 original-factor update,
and binary64 Cholesky with up to five corrections and Decimal 80 edge residuals.
SF-LDL independently implements the principle of Orera–Peña (2024), DOI
10.3390/math12223491; it is not their software. The direct refinement baseline
is not the Sherman–Morrison iterative-refinement algorithm (SMIR).

REFERENCE AND FAILURE CONVENTIONS
Binary64 input weights are converted exactly to Decimal. The original suite
uses 160-digit direct deleted-network Gaussian reference solves, with 20 repeats
at 240 digits. The kernel-rank and IEEE studies reuse a 160-digit original-network
LU for reference identities, then check selected outages by independent direct
240-digit solves (60 and 40 checks respectively). Those repeats agree below
relative error 1e-100. Decimal baseline errors are evaluated before conversion to
binary64. Fraction arithmetic independently checks small integer-weight graphs.

Infinity in a relative-error column records a failed, nonpositive or nonfinite
numerical output. CSV tables retain both finite error values and failure counts.
Only the original weight-accuracy display caps finite errors at one and assigns
failures display error one before calculating medians. The triangle diagnostic
plots a zero complement with its mathematical relative error of one. Errors below
the range of the kernel-rank log plot are labelled explicitly instead of being
replaced by arbitrary positive surrogate values.

RECORDED DATA
verification.json: 875 graphs, 3756 finite responses and 1278 bridge outages,
plus long-cycle and exclusive-sum regressions.
accuracy_inputs.json / accuracy.csv: 100 graphs, 1684 queries, original six methods.
precision_checks.json: 20 independent higher-precision reference checks.
triangle.csv: cancellation diagnostic.
timings.csv / rank_timings.csv: subdivision and complete-graph measurements.
environment.json / summary.json: original environment and aggregate results.
revision_validation.json: 62 integer-weight graphs, 329 finite checks for new methods.
revision_accuracy.csv: added six methods on the same 1684 original queries.
revision_accuracy_runtime.csv: single-pass accuracy-stage timing diagnostics.
revision_rank_inputs.json / revision_rank.csv: 20 distinct graphs, 744 queries.
revision_rank_precision_checks.json: 60 independent 240-digit checks.
revision_pglib_inputs.json: parsed active branches, taps, bus maps and graph edges.
revision_pglib_accuracy.csv / revision_pglib_summary.json: 313 finite queries,
327 active branches in total, 14 bridges in the four IEEE cases.
revision_pglib_precision_checks.json: 40 independent 240-digit checks.
revision_pglib_timings.csv / revision_timings.csv: added repeated timing runs.
revision_environment.json / revision_summary.json: environment and aggregates.

PGLIB DATA
The original IEEE 14-, 30-, 57- and 118-bus files, upstream README and LICENSE are
in data/pglib/. The source commit, file hashes and exact adaptation are documented
in PROVENANCE.json. No runtime network access is needed. Each active branch uses
r = x * tap, interpreting stored tap zero as one. All active x are positive and
phase shifts zero in these cases; no active branch is dropped. The experiment
uses the positive DC sensitivity Laplacian, not an AC model or operational data.
The case data are CC BY 4.0 and retain original notices. The original software and
accompanying documentation are licensed under MIT; see the repository-root LICENSE.
The files in data/pglib/ retain their upstream terms, which are not replaced by MIT.

INTERPRETATION
HRA factorization and 80-digit direct solves are accurate competitors. Ordinary
sparse updates work well on all four IEEE cases and are faster on the larger ones.
The useful regime for compression has long degree-two chains and a small cyclic
core. The bound becomes conservative as the kernel grows. Absolute timings mix
Python loops and compiled arithmetic and are implementation-specific.
Computational validation does not replace human verification of the proof,
novelty, attribution and final submission content.

SHARED SCHUR EXTENSION

Shared positive Schur reductions

code/joint_outage.py adds all_outages_joint(n, edges). It shares background
networks among balanced groups of outage queries. Active query edges remain
absent; a child adds sibling edges and positively eliminates vertices outside
its own endpoints. The deterministic arithmetic work is
O(n + m + sum_i r_i^3), with the componentwise error bound proved in the
manuscript. More precisely, a kernel with v vertices and k edges costs
O(k v^2 + k log(k+1)); a complete graph therefore costs O(v^4). The
original all_outages remains the independent-query baseline.

Reproduce the added checks and figures with:

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/joint_experiments.py
python code/joint_figures.py

The shared experiments reuse archived inputs and references, and write
results/joint_*.csv and results/joint_*.json. They include 875 full-graph
rational checks (3756 finite outages), 110 additional multigraphs (2310 exact
kernel queries), 301 parallel-edge reference queries, the original 1684-query
weight suite, 744 rank-sweep queries, and 313 finite IEEE outages. The largest
relative errors in the three floating-point suites are below 1.23e-15. All
reported errors satisfy their pathwise bounds. joint_figures.py writes two
PDF figures and a table fragment to figures/; these generated files need not
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


Matched HRA comparison: ladders and complete graphs

Run after the original figure scripts (the last two commands replace Figures 8
and 10 and Table 7):

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/hra_comparison.py
python code/positioning_figures.py
python code/joint_figures.py

hra_comparison_timings.csv compares independent positive elimination, shared
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
hra_comparison_environment.json. The original four-method complete-graph
run remains in positioning_complete_timings.csv; its different medians are
not combined with the new run. Figure 8's subdivision panel retains the
original baseline and positioning_subdivision_timings.csv measurements.
positioning_experiments.py reproduces those archived inputs and references.

The work theorem gives O(k v^2 + k log(k+1)) per kernel, O(v^4) on complete
graphs and O(sum_i r_i^3) in cycle rank. Separate SF-LDL solves cost O(k v^3).
The tested dense factor-and-incidence route also costs O(v^3 + k v^2); the
unbenchmarked full-inverse identity costs O(v^3 + k). Neither asymptotic bound
alone explains the observed constant-factor timing gaps. Depth-first auxiliary
matrix storage is O(v^2 [1 + log_+(p/v)]) for p nonloop kernel edges, plus
O(k) query/input/output storage. Here log_+(x)=max(0,log2(x)).

hra_sparse_extreme.json reproduces the largest finite sparse-update error
from Table 3. Its computed denominator is about 0.198 and the true complement
nearly one: this particular error starts in the original-network solve, not
in division by a nearly zero denominator.

positioning_error_extrema.json identifies the common error/bound extremum in
Sections 7.3 and 7.8: theta_s16_j0, zero-based edge index 6, endpoints (5,1).
Its resistance error/bound is 0.26902461010506 for both methods (E=7).


COMPLETE-GRAPH REFERENCE PRECISION

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/complete_precision_check.py

All 746 direct 100-digit references agree with 240-digit repeats to relative
difference below 2.01e-98. Per-query values and a summary are retained in
results/complete_precision_check.csv and results/complete_precision_summary.json.
Timing ratios use unrounded CSV medians; K24 shared/sparse is
9.863582299394875..., correctly rounded to 9.86.

Vertex-partition validation and matched comparisons (one BLAS thread)
-------------------------------------------------------------------
From the root of the computational archive:
  python code/vertex_study.py original exact rings grid
  python code/vertex_reference_checks.py suites
  python code/vertex_reference_checks.py ladders
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/vertex_matched_study.py grids
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/vertex_matched_study.py families
  python code/vertex_figures.py --output figures/vertex_compiled.pdf

These commands read the archived inputs and per-query references under results/.
The ladder checks independently factor the original deleted networks twice on
each of 2304 edges. The compiled matched runs count scalar arithmetic in a
separate pass. Grid kernel methods take the same compressed input and return
every bypass. The separate original-grid SuperLU times include both reconstructed
outputs on the full network, with preidentified bridges. Path traces report the
actual pivot-degree cost, level count and largest degree. Loop bypasses have no
elimination path. Per-run results retain unrounded errors and times.

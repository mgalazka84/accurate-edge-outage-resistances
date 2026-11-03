REPRODUCIBILITY MATERIAL — MAJOR REVISION, 3 NOVEMBER 2026

Componentwise accurate endpoint resistances for all single-edge outages in
low-cycle-rank graphs — Marek Gałązka and Hanna Wdowicka
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
The case data are CC BY 4.0 and retain original notices. This does not grant a
license for this repository's original code, for which none has been selected.

INTERPRETATION
HRA factorization and 80-digit direct solves are accurate competitors. Ordinary
sparse updates work well on all four IEEE cases and are faster on the larger ones.
The useful regime for compression has long degree-two chains and a small cyclic
core. The bound becomes conservative as the kernel grows. Absolute timings mix
Python loops and compiled arithmetic and are implementation-specific.
Computational validation does not replace human verification of the proof,
novelty, attribution and final submission content.

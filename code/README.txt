ONLINE RESOURCE 1 - REPRODUCIBILITY MATERIAL

Manuscript: Componentwise accurate endpoint resistances for all single-edge
outages in low-cycle-rank graphs
Target journal: Numerical Algorithms
Draft authors: Marek Gałązka and Hanna Wdowicka
Corresponding author: Marek Gałązka, Faculty of Mathematics and Computer Science,
Adam Mickiewicz University, Poznań, Poland; galazka@amu.edu.pl
Version date: 2026-11-03

This is an author-review research draft, not a record of independent peer review.
The core techniques (series reduction and positive star-mesh elimination) are
classical and attributed in the manuscript. No software license has been selected for this repository.

CONTENTS AND INPUT CONTRACT
outage.py: core graph reduction, positive kernel solver, reconstruction, and
          explicit error exponents. No third-party dependency.
baselines.py: dense and sparse nodal baselines and exact/high-precision references.
verify.py: exhaustive exact validation and targeted regression checks.
experiments.py: seeded accuracy, triangle, subdivision and dense-rank experiments.
figures.py: figures, tables and summaries from the recorded CSV files.
            Also draws the introductory outage diagram and the worked chain example.
results/: original saved experimental results, including inputs and environment.

Input edge triples are (u, v, RESISTANCE). Vertices are integers 0,...,n-1.
The graph must be connected, loop-free, with strictly positive finite weights.
Parallel edges are supported. Kernel self-loops arise internally and are treated
according to the reconstruction identity. Each single-edge deletion is separate.
The main guarantee assumes no overflow or underflow in intermediate operations.
The implementation is not an unbounded-exponent or interval-arithmetic solver.

QUICK EXAMPLE
From the code directory:
    python outage.py
The triangle with resistances 1, 1e16, 1e16 has endpoint outage resistance 2e16
and leverage-score complement approximately 5e-17 on its first edge.

REPRODUCTION
Use Python 3.12 and the dependency versions recorded in requirements.txt and
results/environment.json. Run from the extracted archive root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/verify.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/experiments.py
    python code/figures.py

On Windows set the same environment variables using the shell's native syntax
before running the three Python commands. Relative output paths are resolved
from the scripts, not from the current shell directory.

The experiments overwrite their generated result files. Preserve an untouched
copy of the supplied recorded results if exact comparison is needed.
Graphs/weights are generated with Python random.Random and saved explicitly in
results/accuracy_inputs.json. The figure script creates the manuscript directory
if needed. Timings are medians of three runs after one warm-up; all runs are saved.
Exact timings will vary with machine and load. No baseline timing is extrapolated.

REFERENCE AND FAILURE HANDLING
Grounded Decimal Gaussian elimination uses 160 decimal digits and exactly imports
the binary64 input values. Twenty reference queries are repeated at 240 digits.
Exact Fraction arithmetic independently checks the small integer-weight suite.
CSV infinity in an error column denotes a failed/nonpositive/nonfinite result.
The failure counts and maxima over finite outputs are both retained. In one
display only, errors are capped at 1 and failures assigned 1 before medians are
computed; the raw CSV is unchanged. In the scalar diagnostic triangle.csv, a zero
subtraction is stored and plotted with its mathematical relative error of 1;
the other diagnostics classify nonpositive resistance outputs as failures.

DATA FILES
verification.json: exact test counts and bound checks.
accuracy.csv: per-outage references, all method errors, and analytic bounds.
accuracy_inputs.json: all 100 graphs with the actual binary64 weights.
precision_checks.json: the 20 high-precision repeat comparisons.
triangle.csv: scalar cancellation diagnostic.
timings.csv: fixed-cycle-rank subdivision timing measurements.
rank_timings.csv: unfavorable dense-graph timing measurements.
summary.json: aggregate errors and failure counts.
environment.json: actual versions, processor and thread settings.

The exact validation checks 875 graphs, 3756 finite outage responses and 1278
disconnected outages, in addition to a long-cycle and exclusive-sum regression.
The accuracy experiment contains 100 graphs and 1684 finite outage queries.

LIMITATIONS
No empirical power-grid or other application data are included. The solver
computes endpoint outage responses, not arbitrary-pair responses after failure.
The benchmark does not reproduce every stabilized Sherman-Morrison variant.
The mathematical arguments and novelty require human author verification.

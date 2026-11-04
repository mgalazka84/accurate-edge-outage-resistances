"""Matched HRA timings on the archived ladder and complete-graph inputs.

Run with one BLAS thread. No earlier results are overwritten. Timings include
compression and reconstruction. Decimal references are outside timed regions.
"""
import csv
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import scipy
from baselines import decimal_reference, sm_profile, sparse_sm_profile
from experiments import median_time
from joint_outage import all_outages_joint
from outage import all_outages
from revision_baselines import compressed_sf_ldl, decimal_update_profile
from revision_experiments import error, ladder, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def diagnose_sparse_extreme():
    """Record the actual cause of the largest finite sparse-update error."""
    original = list(csv.DictReader((RES/'accuracy.csv').open()))
    row = max(original, key=lambda r: float(r['sparse_update'])
              if math.isfinite(float(r['sparse_update'])) else -1)
    g = next(g for g in json.loads((RES/'accuracy_inputs.json').read_text())
             if g['case'] == row['case'])
    eid = int(row['eid'])
    rho, delta = sparse_sm_profile(g['n'], g['edges'])
    write_json('hra_sparse_extreme.json', dict(case=g['case'], eid=eid,
        recorded_relative_error=float(row['sparse_update']),
        reproduced_relative_error=error(rho[eid], Decimal(row['reference'])),
        computed_rho=float(rho[eid]), computed_delta=float(delta[eid]),
        computed_original_resistance=float(rho[eid]*delta[eid]),
        reference_rho=row['reference'], reference_delta=row['defect_reference'],
        interpretation='The original-network solve is inaccurate. The computed denominator is about 0.198, not nearly zero; the reference complement is nearly one.'))


def main():
    diagnose_sparse_extreme()
    timings, checks, precision_checks, inputs = [], [], [], []
    dense_refs = {(r['case'], int(r['eid'])): Decimal(r['reference'])
                  for r in csv.DictReader((RES/'positioning_complete_accuracy.csv').open())}
    graphs = json.loads((RES/'positioning_complete_inputs.json').read_text())
    for g in graphs:
        g['family'] = 'complete'
    graphs += [dict(case=f'ladder_n{n}', family='ladder', n=n, seed=1107590+n,
                    edges=ladder(n, 4, 1107590+n)) for n in [8, 16, 32, 64, 128]]
    for g in graphs:
        n, edges = g['n'], g['edges']
        methods = [('compressed_positive', lambda: all_outages(n, edges).bypass),
                   ('shared_positive', lambda: all_outages_joint(n, edges).bypass),
                   ('sf_ldl_compressed', lambda: compressed_sf_ldl(n, edges))]
        if g['family'] == 'complete':
            methods += [('sherman_morrison', lambda: sm_profile(n, edges)),
                        ('sparse_update', lambda: sparse_sm_profile(n, edges))]
        for name, fn in methods:
            med, raw = median_time(fn)
            timings.append(dict(case=g['case'], family=g['family'], n=n,
                                m=len(edges), rank=len(edges)-n+1, method=name,
                                median_seconds=med, run1=raw[0], run2=raw[1], run3=raw[2]))
        if g['family'] == 'complete':
            refs = [dense_refs[g['case'], eid] for eid in range(len(edges))]
        else:
            refs = decimal_update_profile(n, edges, precision=160)
            for eid in sorted({0, len(edges)//2, len(edges)-1}):
                u, v, _ = edges[eid]
                direct = decimal_reference(n, edges, u, v, omit=eid, precision=240)
                discrepancy = error(refs[eid], direct)
                assert discrepancy < 1e-100, (g['case'], eid, discrepancy)
                precision_checks.append(dict(case=g['case'], eid=eid,
                    reference160=str(refs[eid]), reference240=str(direct),
                    relative_difference=discrepancy))
        sf = compressed_sf_ldl(n, edges)
        shared = all_outages_joint(n, edges)
        for eid, reference in enumerate(refs):
            sf_error, shared_error = error(sf[eid], reference), error(shared.bypass[eid], reference)
            bound = math.expm1(shared.error_exponents[eid] * -math.log1p(-2**-53))
            assert sf_error < 1e-13, (g['case'], eid, sf_error)
            assert shared_error <= bound, (g['case'], eid, shared_error, bound)
            checks.append(dict(case=g['case'], family=g['family'], eid=eid,
                reference=str(reference), sf_ldl_rho=sf[eid], sf_ldl_error=sf_error,
                shared_rho=shared.bypass[eid], shared_error=shared_error,
                shared_exponent=shared.error_exponents[eid], shared_bound=bound))
        inputs.append(g)
        write_csv('hra_comparison_timings.csv', timings)
        write_csv('hra_comparison_accuracy.csv', checks)
        write_json('hra_comparison_precision_checks.json', precision_checks)
        write_json('hra_comparison_inputs.json', inputs)
        print(g['case'], {r['method']: r['median_seconds'] for r in timings if r['case']==g['case']}, flush=True)
    env = dict(date=datetime.now(timezone.utc).isoformat(), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        threads={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        timing='one warm-up and three complete-profile runs; median; references outside timed regions',
        implementations='positive solvers: Python loops; SF-LDL: Python with vectorized NumPy updates and dot products; dense/sparse update: SciPy compiled factorizations and solves',
        references='complete: archived 100-digit direct nodal solves; ladders: Decimal(160) original-factor update, checked by three Decimal(240) direct deleted-network solves per graph',
        checked_outputs=len(checks), max_sf_ldl_error=max(r['sf_ldl_error'] for r in checks),
        max_shared_error=max(r['shared_error'] for r in checks))
    write_json('hra_comparison_environment.json', env)
    print(json.dumps(env, indent=2), flush=True)


if __name__ == '__main__':
    main()

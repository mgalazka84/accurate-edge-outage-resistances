"""Matched timings for Algorithm 2 and the original Figure 8 inputs.

Retains all previous result files. Use one BLAS thread. Complete-graph
methods are rerun together; subdivision adds only the missing shared curve.
"""
import csv
import json
import math
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from baselines import sm_profile, sparse_sm_profile, decimal_reference
from experiments import median_time, relerr, subdivided, write_csv
from joint_outage import all_outages_joint
from outage import all_outages

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def record_extrema():
    old = list(csv.DictReader((RES/'accuracy.csv').open()))
    shared = list(csv.DictReader((RES/'joint_accuracy.csv').open()))
    x = max(old, key=lambda r: float(r['positive'])/float(r['analytic_bound']))
    y = max(shared, key=lambda r: float(r['error_to_bound']))
    inputs = json.loads((RES/'accuracy_inputs.json').read_text())
    graph = next(g for g in inputs if g['case'] == x['case'])
    assert (x['case'],x['eid']) == (y['case'],y['eid'])
    out = dict(
        independent=dict(case=x['case'], zero_based_eid=int(x['eid']),
                         rho_error=float(x['positive']), bound=float(x['analytic_bound']),
                         exponent=int(x['exponent']),
                         ratio=float(x['positive'])/float(x['analytic_bound'])),
        shared=dict(case=y['case'], zero_based_eid=int(y['eid']),
                    rho_error=float(y['relative_error']), bound=float(y['bound']),
                    exponent=int(y['exponent']), ratio=float(y['error_to_bound'])),
        endpoints=graph['edges'][int(x['eid'])][:2],
        note='The same resistance output attains both reported maxima; 0.270 is an upper bound, not an attained value.')
    (RES/'positioning_error_extrema.json').write_text(json.dumps(out,indent=2)+'\n')


def main():
    rows, inputs, checks = [], [], []
    for n in [4, 6, 8, 10, 12, 16, 20, 24]:
        rng = random.Random(72000 + n)
        edges = [(u, v, 10.0**rng.uniform(-2, 2))
                 for u in range(n) for v in range(u+1, n)]
        inputs.append(dict(case=f'K{n}', seed=72000+n, n=n, edges=edges))
        methods = [('compressed_positive', lambda: all_outages(n, edges)),
                   ('shared_positive', lambda: all_outages_joint(n, edges)),
                   ('sherman_morrison', lambda: sm_profile(n, edges)),
                   ('sparse_update', lambda: sparse_sm_profile(n, edges))]
        for name, fn in methods:
            med, raw = median_time(fn)
            rows.append(dict(n=n, m=len(edges), rank=len(edges)-n+1,
                             method=name, median_seconds=med,
                             run1=raw[0], run2=raw[1], run3=raw[2]))
        # Independent high-precision reference for every dense-graph output.
        refs = [decimal_reference(n, edges, u, v, omit=eid, precision=100)
                for eid, (u,v,_) in enumerate(edges)]
        prof = all_outages_joint(n, edges)
        for eid, reference in enumerate(refs):
            error = relerr(prof.bypass[eid], reference)
            bound = math.expm1(prof.error_exponents[eid] * -math.log1p(-2**-53))
            assert error <= bound
            checks.append(dict(case=f'K{n}', eid=eid, reference=str(reference),
                               rho=prof.bypass[eid], relative_error=error,
                               exponent=prof.error_exponents[eid], bound=bound))
        print(f'Complete graph n={n}: matched timings and {len(edges)} checks', flush=True)
    write_csv('positioning_complete_timings.csv', rows)
    write_csv('positioning_complete_accuracy.csv', checks)
    (RES/'positioning_complete_inputs.json').write_text(json.dumps(inputs, separators=(',', ':')))

    rows=[]
    for length in [1,4,16,64,256,1024,4096,16384]:
        n, edges=subdivided('K4',length,4,42000+length)
        med, raw=median_time(lambda: all_outages_joint(n,edges))
        rows.append(dict(length=length,n=n,m=len(edges),rank=3,
                         method='shared_positive',median_seconds=med,
                         run1=raw[0],run2=raw[1],run3=raw[2]))
        print(f'Shared subdivision length={length}, m={len(edges)}',flush=True)
    write_csv('positioning_subdivision_timings.csv',rows)
    env=dict(date=datetime.now(timezone.utc).isoformat(),python=sys.version,
             platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
             threads={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
             timing='one warm-up, three measured complete-profile runs; median',
             inputs='complete graphs: seed 72000+n, half-width 2; subdivision: seed 42000+length, half-width 4',
             figure='complete panel reruns all four methods; subdivision retains original baselines',
             checks=len(checks),max_relative_error=max(x['relative_error'] for x in checks))
    (RES/'positioning_environment.json').write_text(json.dumps(env,indent=2)+'\n')
    record_extrema()


if __name__ == '__main__':
    main()

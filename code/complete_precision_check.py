"""Check every archived 100-digit complete-graph reference at 240 digits.

This checks precision convergence of the direct deleted-network nodal solve.
It does not replace the separate rational verification or change any timings.
"""
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

from baselines import decimal_reference
from revision_experiments import error, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def main():
    original = {(r['case'], int(r['eid'])): r for r in
                csv.DictReader((RES/'positioning_complete_accuracy.csv').open())}
    comparisons = {(r['case'], int(r['eid'])): r for r in
                   csv.DictReader((RES/'hra_comparison_accuracy.csv').open())
                   if r['family'] == 'complete'}
    graphs = json.loads((RES/'positioning_complete_inputs.json').read_text())
    rows = []
    for g in graphs:
        n, edges = g['n'], g['edges']
        for eid, (u, v, _) in enumerate(edges):
            old = original[g['case'], eid]
            comparison = comparisons[g['case'], eid]
            reference = decimal_reference(n, edges, u, v, omit=eid, precision=240)
            with localcontext() as ctx:
                ctx.prec = 240
                discrepancy = abs(Decimal(old['reference'])/reference - 1)
            assert discrepancy < Decimal('1e-90'), (g['case'], eid, discrepancy)
            shared_error = error(float(comparison['shared_rho']), reference)
            sf_error = error(float(comparison['sf_ldl_rho']), reference)
            assert shared_error <= float(comparison['shared_bound'])
            assert sf_error < 5.66e-16
            rows.append(dict(case=g['case'], eid=eid, reference240=str(reference),
                             relative_difference100_240=str(discrepancy),
                             shared_error240=shared_error, sf_ldl_error240=sf_error))
        write_csv('complete_precision_check.csv', rows)
        print(g['case'], len(edges), 'reference checks passed', flush=True)
    summary = dict(status='PASS', date=datetime.now(timezone.utc).isoformat(),
        graphs=len(graphs), queries=len(rows), original_precision=100, check_precision=240,
        input_half_width=2, input_note='Archived binary64 resistances converted exactly to Decimal',
        max_relative_reference_difference=max(float(r['relative_difference100_240']) for r in rows),
        max_shared_error240=max(r['shared_error240'] for r in rows),
        max_sf_ldl_error240=max(r['sf_ldl_error240'] for r in rows),
        note='Precision convergence of all direct nodal references; not a formal interval certificate.')
    write_json('complete_precision_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

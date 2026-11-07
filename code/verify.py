"""Exact exhaustive and randomized validation, producing a machine-readable log."""
import itertools, json, math, random, time
from fractions import Fraction
from pathlib import Path
from outage import all_outages, exclusive_sums
from baselines import grounded_generic


def connected(n, edges, omit=None):
    adj = [[] for _ in range(n)]
    for i, (u, v, _) in enumerate(edges):
        if i != omit:
            adj[u].append(v); adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == n


def main(solver=None, result_name="verification.json"):
    solver = all_outages if solver is None else solver
    start = time.perf_counter()
    cases, queries, bridges, maxerr, maxratio = 0, 0, 0, 0.0, 0.0
    def check(n, edges):
        nonlocal cases, queries, bridges, maxerr, maxratio
        exact_edges = [(u,v,Fraction(r)) for u,v,r in edges]
        float_edges = [(u,v,float(r)) for u,v,r in edges]
        exact = solver(n, exact_edges)
        actual = solver(n, float_edges)
        for k in actual.kernels:
            if k.rank >= 2:
                assert k.n <= 2*k.rank-2
                assert len(k.edges) <= 3*k.rank-3
        for eid,(u,v,r) in enumerate(edges):
            if not connected(n, edges, eid):
                assert eid in exact.bridges and math.isinf(actual.bypass[eid])
                bridges += 1
                continue
            ref = grounded_generic(n, exact_edges, u,v,omit=eid)
            assert exact.bypass[eid] == ref, (n,edges,eid,exact.bypass[eid],ref)
            assert exact.defect[eid] == Fraction(r)/(Fraction(r)+ref)
            # Integer inputs in this suite are exactly representable in binary64.
            err = abs(float(Fraction.from_float(actual.bypass[eid])/ref-1))
            err_d = abs(float(Fraction.from_float(actual.defect[eid])/exact.defect[eid]-1))
            bound = math.expm1(actual.error_exponents[eid]*(-math.log1p(-2**-53)))
            bound_d = math.expm1((actual.error_exponents[eid]+2)*(-math.log1p(-2**-53)))
            assert err <= bound, (err,bound,n,edges,eid)
            assert err_d <= bound_d, (err_d,bound_d,n,edges,eid)
            maxerr = max(maxerr, err, err_d)
            maxratio = max(maxratio, err/bound, err_d/bound_d)
            queries += 1
        cases += 1

    counts = {}
    for n in range(2,6):
        pairs = list(itertools.combinations(range(n),2))
        c = 0
        for mask in range(1,1<<len(pairs)):
            edges = [(u,v,1+(7*j+n)%11) for j,(u,v) in enumerate(pairs) if mask>>j&1]
            if connected(n,edges):
                check(n,edges); c += 1
        counts[str(n)] = c
    special = [
        (2,[(0,1,1),(0,1,7)]),
        (3,[(0,1,1),(0,1,2),(1,2,3)]),
        (5,[(0,1,1),(1,2,3),(2,0,5),(0,3,7),(3,4,9),(4,0,11)]),
        (6,[(0,1,1),(1,2,3),(2,0,5),(2,3,7),(3,4,9),(4,5,11),(5,3,13)])]
    for n,edges in special: check(n,edges)
    rng = random.Random(11075)
    for _ in range(100):
        n = rng.randrange(3,10)
        pairs = {(v,rng.randrange(v)) for v in range(1,n)}
        pairs = {tuple(sorted(e)) for e in pairs}
        for _ in range(rng.randrange(1,6)):
            pairs.add(tuple(sorted(rng.sample(range(n),2))))
        check(n,[(u,v,rng.randrange(1,100)) for u,v in sorted(pairs)])
    assert exclusive_sums([1e30,1.0,2.0])[0] == 3.0
    long_cycle = [(i,(i+1)%10001,1.0) for i in range(10001)]
    p = solver(10001,long_cycle)
    assert all(x == 10000.0 for x in p.bypass)
    result = dict(status='PASS', exhaustive_connected_graphs=counts,
                  tested_graphs=cases, finite_outage_queries=queries,
                  disconnected_outages=bridges, max_relative_error=maxerr,
                  max_error_to_proven_bound=maxratio,
                  long_cycle_edges=10001,
                  elapsed_seconds=time.perf_counter()-start)
    out = Path(__file__).resolve().parents[1]/'results'/result_name
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()

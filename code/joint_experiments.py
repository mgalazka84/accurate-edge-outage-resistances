"""Reproduce shared-Schur correctness, accuracy and scaling experiments.

Uses the archived inputs and high-precision references. Generates only joint_*
results; preserves all baseline records. No external data are fetched.
"""
import csv, io, json, math, platform, random, statistics, sys, time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from outage import all_outages, Kernel, height
from joint_outage import all_outages_joint, joint_kernel
from baselines import grounded_generic
from experiments import subdivided
from revision_experiments import atomic, error, ladder, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def validate():
    import verify
    saved = (OUT/'verification.json').read_bytes()
    verify.all_outages = all_outages_joint
    try:
        verify.main()
        atomic(OUT/'joint_verification.json', (OUT/'verification.json').read_bytes())
    finally:
        atomic(OUT/'verification.json', saved)
    rng = random.Random(1107503)
    cases, queries = 0, 0
    for n in range(2, 13):
        for repetition in range(10):
            pairs = [(i, (i+1) % n) for i in range(n)]
            pairs += [tuple(rng.sample(range(n), 2)) for _ in range(2*n)]
            rng.shuffle(pairs)
            edges = [(a, b, Fraction(rng.randrange(1, 30))) for a,b in pairs]
            kernel = Kernel(n, edges, [[i] for i in range(len(edges))])
            values, exponents = joint_kernel(kernel)
            for eid, (a,b,_) in enumerate(edges):
                ref = grounded_generic(n, edges, a, b, omit=eid)
                assert values[eid] == ref, (n,repetition,eid)
                queries += 1
            cases += 1
    write_json('joint_kernel_validation.json',dict(status='PASS',graphs=cases,
        exact_finite_queries=queries,seed=1107503,description='Uncompressed cyclic multigraphs; shuffled query order; independent rational nodal solves'))
    count, worst = 0, 0.0
    for multiplicity in [2,3,7,32,257]:
        weights=[float(2**((i*17)%97)) for i in range(multiplicity)]
        e=[(0,1,w) for w in weights]
        vals,exps=joint_kernel(Kernel(2,e,[[i] for i in range(multiplicity)]))
        uniform=2+2*height(multiplicity)
        for j,val in enumerate(vals):
            ref=1/sum((1/Fraction(w) for i,w in enumerate(weights) if i!=j),Fraction(0))
            err=abs(float(Fraction(val)/ref-1))
            bound=math.expm1(exps[j]*(-math.log1p(-2**-53)))
            assert err<=bound and exps[j]<=uniform,(multiplicity,j)
            worst=max(worst,err);count+=1
    write_json('joint_parallel_validation.json',dict(status='PASS',
        parallel_multiplicities=[2,3,7,32,257],exact_reference_queries=count,
        max_relative_error=worst,description='Power-of-two resistances; direct parallel-conductance oracle; pathwise and uniform exponent checks'))


def accuracy():
    summaries=[]
    for tag, inputfile, referencefile in [
        ('accuracy','accuracy_inputs.json','accuracy.csv'),
        ('rank','revision_rank_inputs.json',None),
        ('pglib','revision_pglib_inputs.json','revision_pglib_accuracy.csv')]:
        graphs=json.loads((OUT/inputfile).read_text())
        lookup = {(r['case'], int(r['eid'])):r for r in csv.DictReader((OUT/referencefile).open())} if referencefile else {}
        rows=[]
        for g in graphs:
            p=all_outages_joint(g['n'],g['edges'])
            refs=None
            if tag=='rank':
                from revision_baselines import decimal_update_profile
                refs=decimal_update_profile(g['n'],g['edges'],160)
            for eid,(_,_,resistance) in enumerate(g['edges']):
                if eid in p.bridges:
                    assert math.isinf(p.bypass[eid]) and p.defect[eid]==0
                    continue
                ref=refs[eid] if refs is not None else Decimal(lookup[g['case'],eid]['reference'])
                with localcontext() as ctx:
                    ctx.prec=200
                    dref=Decimal(resistance)/(Decimal(resistance)+ref)
                er,ed=error(p.bypass[eid],ref),error(p.defect[eid],dref)
                e=p.error_exponents[eid]
                b=math.expm1(e*(-math.log1p(-2**-53)))
                bd=math.expm1((e+2)*(-math.log1p(-2**-53)))
                assert er<=b and ed<=bd,(tag,g['case'],eid,er,ed,b,bd)
                rows.append(dict(case=g['case'],eid=eid,rho_reference=str(ref),
                    delta_reference=str(dref),rho=p.bypass[eid],delta=p.defect[eid],
                    relative_error=er,defect_error=ed,exponent=e,bound=b,
                    error_to_bound=max(er/b,ed/bd)))
        write_csv('joint_'+tag+'.csv',rows)
        summaries.append(dict(suite=tag,graphs=len(graphs),queries=len(rows),
            max_rho_error=max(r['relative_error'] for r in rows),
            max_delta_error=max(r['defect_error'] for r in rows),
            max_error_to_bound=max(r['error_to_bound'] for r in rows)))
        print(summaries[-1],flush=True)
    write_json('joint_accuracy_summary.json',summaries)


class CountedFloat:
    """Instrument the unchanged independent solver's positive arithmetic."""
    counts=0
    def __init__(self,value): self.value=float(value)
    @staticmethod
    def value_of(x): return x.value if isinstance(x,CountedFloat) else x
    def __add__(self,other):
        CountedFloat.counts+=1; return CountedFloat(self.value+self.value_of(other))
    __radd__=__add__
    def __mul__(self,other):
        CountedFloat.counts+=1; return CountedFloat(self.value*self.value_of(other))
    __rmul__=__mul__
    def __truediv__(self,other):
        CountedFloat.counts+=1; return CountedFloat(self.value/self.value_of(other))
    def __rtruediv__(self,other):
        CountedFloat.counts+=1; return CountedFloat(self.value_of(other)/self.value)
    def __gt__(self,other): return self.value>self.value_of(other)
    def __eq__(self,other): return self.value==self.value_of(other)
    def __bool__(self): return bool(self.value)


def timings():
    from outage import positive_elimination
    rows=[]
    for n in [8,16,32,64,128]:
        edges=ladder(n,4,1107590+n)
        k=Kernel(n,edges,[[i] for i in range(len(edges))])
        stats={}; joint_kernel(k,stats=stats)
        CountedFloat.counts=0
        ce=[(a,b,CountedFloat(r)) for a,b,r in edges]
        for eid,(a,b,_) in enumerate(edges):
            positive_elimination(n,ce,a,b,omit=eid)
        independent_ops=CountedFloat.counts
        methods={'independent':lambda:all_outages(n,edges),
                 'shared':lambda:all_outages_joint(n,edges)}
        for name,fn in methods.items():
            fn(); runs=[]
            for _ in range(3):
                begin=time.perf_counter(); fn(); runs.append(time.perf_counter()-begin)
            rows.append(dict(family='ladder',n=n,m=len(edges),rank=len(edges)-n+1,
                method=name,seconds=statistics.median(runs),run1=runs[0],run2=runs[1],run3=runs[2],
                kernel_arithmetic_ops=independent_ops if name=='independent' else stats['arithmetic_ops']))
        write_csv('joint_timings.csv',rows)
        print('Shared scaling n=',n,'ops=',independent_ops,stats['arithmetic_ops'],flush=True)
    for g in json.loads((OUT/'revision_pglib_inputs.json').read_text()):
        for name,fn in [('independent',all_outages),('shared',all_outages_joint)]:
            fn(g['n'],g['edges']); runs=[]
            for _ in range(3):
                start=time.perf_counter();fn(g['n'],g['edges']);runs.append(time.perf_counter()-start)
            rows.append(dict(family=g['case'],n=g['n'],m=len(g['edges']),rank=len(g['edges'])-g['n']+1,
                method=name,seconds=statistics.median(runs),run1=runs[0],run2=runs[1],run3=runs[2],kernel_arithmetic_ops=''))
    write_csv('joint_timings.csv',rows)
    write_json('joint_environment.json',dict(python=sys.version,platform=platform.platform(),
        timing='one warm-up, three timed complete profiles; same Python scalar kernels',
        operations='addition, multiplication, division only; kernel stage; diagnostic counts, not all graph work'))


if __name__=='__main__':
    validate(); accuracy(); timings()

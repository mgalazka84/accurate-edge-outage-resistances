"""Reproduce numerical tables/figures. Run with BLAS threads fixed to one."""
import csv, json, math, os, platform, random, statistics, sys, time
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import scipy
from outage import all_outages, compress, exclusive_sums, positive_elimination
from baselines import grounded_float, decimal_reference, sm_profile, sparse_sm_profile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'


def write_csv(name, rows):
    if rows:
        with (RESULTS/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def relerr(x, reference):
    if not math.isfinite(float(x)) or x <= 0:
        return math.inf
    with localcontext() as ctx:
        ctx.prec=180
        return float(abs(Decimal.from_float(float(x))/reference-1))


def family(name):
    if name == 'theta': return 2,[(0,1),(0,1),(0,1)]
    if name == 'K4': return 4,[(u,v) for u in range(4) for v in range(u+1,4)]
    if name == 'K33': return 6,[(u,v) for u in range(3) for v in range(3,6)]
    if name == 'prism': return 6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
    if name == 'Petersen':
        return 10,[(i,(i+1)%5) for i in range(5)]+[(i,i+5) for i in range(5)]+[(i+5,(i+2)%5+5) for i in range(5)]
    raise ValueError(name)


def subdivided(name, length, spread, seed, variable=False):
    n, pairs = family(name)
    rng=random.Random(seed)
    edges=[]
    for u,v in pairs:
        count=rng.randrange(1,length+1) if variable else length
        nodes=[u]+list(range(n,n+count-1))+[v]
        n += count-1
        for a,b in zip(nodes,nodes[1:]):
            resistance = 10.0**rng.uniform(-spread,spread) if spread else 1.0
            edges.append((a,b,resistance))
    return n,edges


def compressed_chol(n, edges):
    bridges,kernels=compress(n,edges)
    values=[math.inf]*len(edges)
    for k in kernels:
        for j,(s,t,_) in enumerate(k.edges):
            try:
                val=grounded_float(k.n,k.edges,s,t,omit=j) if s!=t else 0.0
            except (np.linalg.LinAlgError,ValueError): val=math.nan
            for eid,rest in zip(k.chains[j],exclusive_sums([edges[e][2] for e in k.chains[j]])):
                values[eid]=rest+val
    return values


def triangle_experiment():
    rows=[]
    for exponent in range(2,33,2):
        scale=10.0**exponent
        edges=[(0,1,1.0),(1,2,scale),(2,0,scale)]
        p=all_outages(3,edges)
        with localcontext() as ctx:
            ctx.prec=180
            ref=Decimal(2)*Decimal.from_float(scale)
            dref=Decimal(1)/(1+ref)
            oracle_R=float(ref/(1+ref))
        sm,sd=sm_profile(3,edges)
        cancellation=1.0-oracle_R
        with localcontext() as ctx:
            ctx.prec=180
            cancellation_error=float(abs(Decimal.from_float(cancellation)/dref-1))
        row=dict(exponent=exponent,delta_reference=str(dref),
                 positive_defect=p.defect[0],oracle_subtraction=cancellation,
                 positive_defect_relerr=relerr(p.defect[0],dref),
                 oracle_subtraction_relerr=cancellation_error,
                 sm_bypass_relerr=relerr(sm[0],ref),
                 positive_bypass_relerr=relerr(p.bypass[0],ref))
        rows.append(row)
    write_csv('triangle.csv',rows)


def accuracy_experiment():
    rows=[]; graphs=[]; precision_checks=[]
    names=['theta','K4','K33','prism','Petersen']
    counter=0
    for name in names:
        for spread in [0,8,16,32]:
            for replicate in range(5):
                seed=1107500+counter
                n,edges=subdivided(name,3,spread,seed,variable=True)
                case=f'{name}_s{spread}_j{replicate}'
                graphs.append(dict(case=case,n=n,edges=edges))
                p=all_outages(n,edges)
                cholc=compressed_chol(n,edges)
                sm,_=sm_profile(n,edges)
                sparse_sm,_=sparse_sm_profile(n,edges)
                for eid,(u,v,r) in enumerate(edges):
                    ref=decimal_reference(n,edges,u,v,omit=eid,precision=160)
                    if eid==0 and replicate==0:
                        ref2=decimal_reference(n,edges,u,v,omit=eid,precision=240)
                        with localcontext() as ctx:
                            ctx.prec=240
                            discrep=float(abs(ref/ref2-1))
                        assert discrep < 1e-80, (case,discrep)
                        precision_checks.append(dict(case=case,relative_difference=discrep))
                    with localcontext() as ctx:
                        ctx.prec=180
                        dr=Decimal.from_float(r)/(Decimal.from_float(r)+ref)
                    try: chol=grounded_float(n,edges,u,v,omit=eid)
                    except (np.linalg.LinAlgError,ValueError): chol=math.nan
                    full=positive_elimination(n,edges,u,v,omit=eid)
                    bound=math.expm1(p.error_exponents[eid]*(-math.log1p(-2**-53)))
                    error=relerr(p.bypass[eid],ref)
                    error_d=relerr(p.defect[eid],dr)
                    assert error <= bound, (case,eid,error,bound)
                    assert error_d <= math.expm1((p.error_exponents[eid]+2)*(-math.log1p(-2**-53)))
                    rows.append(dict(case=case,family=name,spread=spread,replicate=replicate,
                                     n=n,m=len(edges),eid=eid,r=r,reference=str(ref),
                                     defect_reference=str(dr),positive=error,
                                     positive_defect=error_d,full_positive=relerr(full,ref),
                                     compressed_cholesky=relerr(cholc[eid],ref),
                                     full_cholesky=relerr(chol,ref),sherman_morrison=relerr(sm[eid],ref),
                                     sparse_update=relerr(sparse_sm[eid],ref),
                                     analytic_bound=bound,exponent=p.error_exponents[eid]))
                counter += 1
            print(f'Accuracy {name}, spread {spread}: {counter}/100 graphs',flush=True)
    write_csv('accuracy.csv',rows)
    (RESULTS/'accuracy_inputs.json').write_text(json.dumps(graphs,separators=(',',':')))
    (RESULTS/'precision_checks.json').write_text(json.dumps(precision_checks,indent=2))


def median_time(fn, repeats=3):
    fn()  # warmup
    values=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); values.append(time.perf_counter()-start)
    return statistics.median(values),values


def timing_experiment():
    rows=[]
    for length in [1,4,16,64,256,1024,4096,16384]:
        n,edges=subdivided('K4',length,4,42000+length)
        methods={'compressed_positive':lambda:all_outages(n,edges),
                 'compressed_cholesky':lambda:compressed_chol(n,edges)}
        if length<=16:
            methods['full_positive']=lambda:[positive_elimination(n,edges,u,v,omit=i) for i,(u,v,_) in enumerate(edges)]
            methods['full_cholesky']=lambda:[grounded_float(n,edges,u,v,omit=i) for i,(u,v,_) in enumerate(edges)]
        if length<=256:
            methods['sherman_morrison']=lambda:sm_profile(n,edges)
        if length<=1024:
            methods['sparse_update']=lambda:sparse_sm_profile(n,edges)
        for method,fn in methods.items():
            median,raw=median_time(fn)
            rows.append(dict(length=length,n=n,m=len(edges),rank=3,method=method,
                             median_seconds=median,run1=raw[0],run2=raw[1],run3=raw[2]))
        print(f'Timing subdivision length {length}, n={n}, m={len(edges)}',flush=True)
    write_csv('timings.csv',rows)
    # A separate rank sweep deliberately includes graphs without useful compression.
    rows=[]
    for n in [4,6,8,10,12,16,20,24]:
        rng=random.Random(72000+n)
        edges=[(u,v,10.0**rng.uniform(-2,2)) for u in range(n) for v in range(u+1,n)]
        for method,fn in [('compressed_positive',lambda:all_outages(n,edges)),
                          ('sherman_morrison',lambda:sm_profile(n,edges)),
                          ('sparse_update',lambda:sparse_sm_profile(n,edges))]:
            median,raw=median_time(fn)
            rows.append(dict(n=n,m=len(edges),rank=len(edges)-n+1,method=method,
                             median_seconds=median,run1=raw[0],run2=raw[1],run3=raw[2]))
    write_csv('rank_timings.csv',rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
             scipy=scipy.__version__,processor=platform.processor(),
             threads={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
             date='2026-11-03',reference='decimal, 160 decimal digits; selected checks at 240')
    try:
        env['cpu_model']=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    except (OSError,StopIteration): pass
    (RESULTS/'environment.json').write_text(json.dumps(env,indent=2))
    triangle_experiment()
    accuracy_experiment()
    timing_experiment()


if __name__=='__main__': main()

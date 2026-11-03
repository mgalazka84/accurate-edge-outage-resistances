"""Reproduce added baseline, kernel-rank and PGLib experiments.

Run after experiments.py, or reuse the archived original accuracy.csv inputs.
No data are fetched at runtime: attributed PGLib source files are included.
Each stage atomically saves its results. Numerical failures are retained.
"""
import argparse, csv, hashlib, io, json, math, os, platform, random, re
import statistics, sys, time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy
from outage import all_outages, compress, find_bridges, positive_elimination
from baselines import decimal_reference, grounded_generic, sparse_sm_profile
from experiments import subdivided, median_time
from revision_baselines import (sf_ldl, compressed_sf_ldl, full_profile,
    decimal_direct, decimal_update_profile, refined_cholesky)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
PGLIB_COMMIT='dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3'


def atomic(path,data):
    if isinstance(data,str): data=data.encode()
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.complete')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def write_json(name,obj):
    atomic(OUT/name,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def write_csv(name,rows):
    b=io.StringIO(); w=csv.DictWriter(b,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows); atomic(OUT/name,b.getvalue())


def error(value,reference):
    with localcontext() as ctx:
        ctx.prec=240
        x=value if isinstance(value,Decimal) else Decimal(float(value))
        ref=reference if isinstance(reference,Decimal) else Decimal(reference.numerator)/Decimal(reference.denominator)
        return float(abs(x/ref-1)) if x.is_finite() and x>0 else math.inf


def additional_profiles(n,edges):
    return {
        'sf_ldl_full':lambda:full_profile(n,edges,sf_ldl),
        'sf_ldl_compressed':lambda:compressed_sf_ldl(n,edges),
        'decimal35_direct':lambda:full_profile(n,edges,lambda *a,**k:decimal_direct(*a,**k,precision=35)),
        'decimal80_direct':lambda:full_profile(n,edges,lambda *a,**k:decimal_direct(*a,**k,precision=80)),
        'decimal35_update':lambda:decimal_update_profile(n,edges,35),
        'cholesky_ir':lambda:full_profile(n,edges,refined_cholesky)}


def validate():
    rng=random.Random(1107531); cases=[]
    for _ in range(60):
        n=rng.randrange(3,10)
        pairs={(i,rng.randrange(i)) for i in range(1,n)}
        pairs={tuple(sorted(p)) for p in pairs}
        for j in range(rng.randrange(1,9)):
            pairs.add(tuple(sorted(rng.sample(range(n),2))))
        cases.append((n,[(v,w,float(rng.randrange(1,40))) for v,w in sorted(pairs)]))
    cases.extend([(2,[(0,1,1.0),(0,1,7.0)]),
        (5,[(0,1,1.),(1,2,3.),(2,0,5.),(0,3,7.),(3,4,9.),(4,0,11.)])])
    count=0; maxima={name:0.0 for name in additional_profiles(*cases[0])}
    for n,edges in cases:
        bridge=find_bridges(n,edges)
        refs=[grounded_generic(n,edges,s,t,omit=i,factory=Fraction) if i not in bridge else None for i,(s,t,_) in enumerate(edges)]
        for name,fn in additional_profiles(n,edges).items():
            vals=fn()
            for i,ref in enumerate(refs):
                if ref is None:
                    assert math.isinf(float(vals[i])); continue
                err=error(vals[i],ref); maxima[name]=max(maxima[name],err)
                tol=5e-29 if name.startswith('decimal35') else 5e-74 if name.startswith('decimal80') else 5e-14
                assert err<tol,(name,n,i,err)
        count+=len(edges)-len(bridge)
    for exponent in [16,32]:
        e=[(0,1,1.0),(1,2,10.0**exponent),(2,0,10.0**exponent)]
        assert error(sf_ldl(3,e,0,1,omit=0),Decimal(2)*Decimal(e[1][2]))<1e-14
    result=dict(status='PASS',graphs=len(cases),finite_queries=count,max_relative_errors=maxima,
                extra_triangle_exponents=[16,32])
    write_json('revision_validation.json',result); print(json.dumps(result),flush=True)


def accuracy():
    graphs=json.loads((OUT/'accuracy_inputs.json').read_text())
    old={(row['case'],int(row['eid'])):row for row in csv.DictReader((OUT/'accuracy.csv').open())}
    rows=[]; runtime=[]
    for gi,g in enumerate(graphs):
        profiles={}
        for name,fn in additional_profiles(g['n'],g['edges']).items():
            start=time.perf_counter(); profiles[name]=fn(); elapsed=time.perf_counter()-start
            runtime.append(dict(case=g['case'],method=name,seconds=elapsed))
        for i,edge in enumerate(g['edges']):
            src=old[(g['case'],i)]; ref=Decimal(src['reference'])
            row=dict(case=g['case'],eid=i,n=g['n'],m=len(g['edges']),spread=int(src['spread']),reference=src['reference'])
            for name,vals in profiles.items():
                row[name]=error(vals[i],ref)
                row[name+'_value']=str(vals[i])
            rows.append(row)
        if (gi+1)%10==0:
            write_csv('revision_accuracy.csv',rows); write_csv('revision_accuracy_runtime.csv',runtime)
            print('Added accuracy baselines',gi+1,'/',len(graphs),flush=True)


def ladder(n,spread,seed):
    if n==4:
        pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    else:
        q=n//2
        pairs=[(i,(i+1)%q) for i in range(q)]+[(i+q,(i+1)%q+q) for i in range(q)]+[(i,i+q) for i in range(q)]
    rng=random.Random(seed)
    return [(i,j,10.0**rng.uniform(-spread,spread) if spread else 1.0) for i,j in pairs]


def rank_sweep():
    rows=[]; inputs=[]; checks=[]
    for n in [4,8,16,32,64]:
        for spread in [0,8]:
            for replicate in range(1 if spread == 0 else 3):
                seed=11075000+100*n+10*spread+replicate
                edges=ladder(n,spread,seed); label=f'ladder_n{n}_s{spread}_j{replicate}'
                inputs.append(dict(case=label,n=n,edges=edges))
                p=all_outages(n,edges)
                ref=decimal_update_profile(n,edges,160)
                for eid,(s,t,r) in enumerate(edges):
                    err=error(p.bypass[eid],ref[eid])
                    exponent=p.error_exponents[eid]
                    bound=math.expm1(exponent*(-math.log1p(-2**-53)))
                    assert err<=bound,(label,eid,err,bound)
                    rows.append(dict(case=label,n=n,m=len(edges),rank=len(edges)-n+1,spread=spread,replicate=replicate,
                        eid=eid,exponent=exponent,relative_error=err,bound=bound,error_to_bound=err/bound))
                for eid in sorted({0,len(edges)//2,len(edges)-1}):
                    s,t,_=edges[eid]
                    ref2=decimal_reference(n,edges,s,t,omit=eid,precision=240)
                    discrepancy=error(ref[eid],ref2)
                    assert discrepancy<1e-100,(label,eid,discrepancy)
                    checks.append(dict(case=label,eid=eid,reference160=str(ref[eid]),reference240=str(ref2),relative_difference=discrepancy))
                write_csv('revision_rank.csv',rows)
                write_json('revision_rank_inputs.json',inputs)
                write_json('revision_rank_precision_checks.json',checks)
        print('Rank sweep n=',n,'completed',flush=True)


def read_pglib(path):
    text=path.read_text()
    def table(key):
        body=re.search(r'mpc\.'+key+r'\s*=\s*\[(.*?)\];',text,re.S).group(1)
        rows=[]
        for line in body.splitlines():
            line=line.split('%')[0].strip().rstrip(';')
            if line: rows.append([float(x) for x in line.split()])
        return rows
    bus,branch=table('bus'),table('branch')
    ids={int(row[0]):i for i,row in enumerate(bus)}
    edges=[]; indices=[]
    for i,row in enumerate(branch):
        if row[10]<=0: continue
        assert row[3]>0 and row[9]==0,('Unsupported reactance or phase shift',path,i)
        tap=row[8] if row[8] else 1.0
        assert tap>0
        edges.append((ids[int(row[0])],ids[int(row[1])],row[3]*tap)); indices.append(i+1)
    return len(bus),edges,dict(source_branch_rows=len(branch),active_branch_rows=indices,bus_ids=list(ids),
        nonunity_taps=sum(row[8] not in (0.,1.) for row in branch),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def grids():
    rows=[]; timing=[]; summary=[]; inputs=[]; checks=[]
    for size in [14,30,57,118]:
        path=ROOT/'data/pglib'/f'pglib_opf_case{size}_ieee.m'
        n,edges,meta=read_pglib(path); label=path.stem
        p=all_outages(n,edges); bridges=p.bridges
        refs=decimal_update_profile(n,edges,160)
        methods={'compressed_positive':lambda:all_outages(n,edges).bypass,
            'sf_ldl_full':lambda:full_profile(n,edges,sf_ldl),
            'sf_ldl_compressed':lambda:compressed_sf_ldl(n,edges),
            'decimal35_update':lambda:decimal_update_profile(n,edges,35),
            'sparse_update':lambda:sparse_sm_profile(n,edges)[0]}
        vals={}
        for name,fn in methods.items():
            # Four calls: first produces values and warms the path; then 3 timed runs.
            vals[name]=fn(); runs=[]
            for _ in range(3):
                start=time.perf_counter(); fn(); runs.append(time.perf_counter()-start)
            timing.append(dict(case=label,method=name,median_seconds=statistics.median(runs),run1=runs[0],run2=runs[1],run3=runs[2]))
        finite=[i for i in range(len(edges)) if i not in bridges]
        selected=sorted(set(finite[round(j*(len(finite)-1)/min(9,len(finite)-1))] for j in range(min(10,len(finite)))))
        for eid in selected:
            s,t,_=edges[eid]
            direct=decimal_reference(n,edges,s,t,omit=eid,precision=240)
            discrepancy=error(refs[eid],direct)
            assert discrepancy<1e-100,(label,eid,discrepancy)
            checks.append(dict(case=label,eid=eid,reference160=str(refs[eid]),reference240=str(direct),relative_difference=discrepancy))
        for eid in finite:
            row=dict(case=label,eid=eid,source_branch_row=meta['active_branch_rows'][eid],reference=str(refs[eid]))
            for name in methods: row[name]=error(vals[name][eid],refs[eid])
            with localcontext() as ctx:
                ctx.prec=180
                delta=Decimal(edges[eid][2])/(Decimal(edges[eid][2])+refs[eid])
            row['delta_reference']=str(delta); row['positive_defect']=error(p.defect[eid],delta)
            row['exponent']=p.error_exponents[eid]
            row['bound']=math.expm1(row['exponent']*(-math.log1p(-2**-53)))
            assert row['compressed_positive']<=row['bound']
            rows.append(row)
        summary.append(dict(case=label,n=n,m=len(edges),cycle_rank=len(edges)-n+1,bridges=len(bridges),
            kernel_vertices=[k.n for k in p.kernels],kernel_edges=[len(k.edges) for k in p.kernels],
            kernel_ranks=[k.rank for k in p.kernels],max_chain=max(len(ch) for k in p.kernels for ch in k.chains),**meta))
        inputs.append(dict(case=label,n=n,edges=edges))
        write_csv('revision_pglib_accuracy.csv',rows); write_csv('revision_pglib_timings.csv',timing)
        write_json('revision_pglib_summary.json',summary); write_json('revision_pglib_inputs.json',inputs)
        write_json('revision_pglib_precision_checks.json',checks)
        print('PGLib',size,'completed;',len(finite),'finite outages',flush=True)
    provenance=dict(repository='https://github.com/power-grid-lib/pglib-opf',commit=PGLIB_COMMIT,
        data_license='CC BY 4.0',source_files=[p.name for p in sorted((ROOT/'data/pglib').glob('*.m'))],
        adaptation='Active branches; r=x*tau, tau=1 for zero tap field; consecutive bus indices; no phase shifts in selected cases. Original branch resistance, charging and AC operating constraints are not modeled. No active branch is dropped.',
        interpretation='Positive Laplacian of the DC phase-angle sensitivity model; not an AC contingency study or measured operational data.')
    atomic(ROOT/'data/pglib/PROVENANCE.json',json.dumps(provenance,indent=2)+'\n')


def timings():
    rows=[]
    for length in [1,4,16,64,256,1024]:
        n,edges=subdivided('K4',length,4,42000+length)
        methods={'compressed_positive':lambda:all_outages(n,edges).bypass,
            'sf_ldl_compressed':lambda:compressed_sf_ldl(n,edges)}
        if n<=100:
            methods['sf_ldl_full']=lambda:full_profile(n,edges,sf_ldl)
            methods['decimal35_update']=lambda:decimal_update_profile(n,edges,35)
            methods['decimal35_direct']=lambda:full_profile(n,edges,lambda *a,**k:decimal_direct(*a,**k,precision=35))
            methods['decimal80_direct']=lambda:full_profile(n,edges,lambda *a,**k:decimal_direct(*a,**k,precision=80))
            methods['cholesky_ir']=lambda:full_profile(n,edges,refined_cholesky)
        for name,fn in methods.items():
            median,runs=median_time(fn)
            rows.append(dict(length=length,n=n,m=len(edges),method=name,median_seconds=median,run1=runs[0],run2=runs[1],run3=runs[2]))
        write_csv('revision_timings.csv',rows)
        print('Additional timings length=',length,'completed',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['all','validate','accuracy','rank','grids','timings'],default='all'); args=parser.parse_args()
    write_json('revision_environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),processor=platform.processor(),
        threads={x:os.environ.get(x) for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},decimal_precisions=[35,80,160,240],date='2026-11-03'))
    stages={'validate':validate,'accuracy':accuracy,'rank':rank_sweep,'grids':grids,'timings':timings}
    for name,fn in stages.items():
        if args.stage in ('all',name):
            start=time.perf_counter(); fn(); print(name,'elapsed',time.perf_counter()-start,flush=True)


if __name__=='__main__': main()

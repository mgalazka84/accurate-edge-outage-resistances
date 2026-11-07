"""Matched all-kernel measurements, grid degree traces and graph cut checks."""
import argparse
import csv
import itertools
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from joint_outage import joint_kernel
from outage import Kernel,compress,height,exclusive_sums
from signed_comparators import compared
from vertex_outage import compiled,inputs,vertex_paths
from vertex_study import ladder,read_grid,write_csv,write_json,decimal_solve
from decimal import Decimal,localcontext

ROOT=Path(__file__).resolve().parents[1]
UNIT=2**-53
ROUTES={2:"edge",3:"vertex",4:"sfldl",5:"inverse",6:"sparse_ldl"}


def bench(n,edges,modes):
    ar=inputs(edges); runners={}; report={}
    for mode in modes:
        route=compiled if mode<=4 else compared
        values,stats=route(n,ar,mode)
        runners[mode]=lambda f=route,m=mode: f(n,ar,m,False)[0]
        warm=runners[mode]()
        np.testing.assert_array_equal(values,warm)
        report[mode]=dict(v=n,k=len(edges),ops=stats['ops'],times=[],values=values)
    for rep in range(5):
        for mode in modes[rep%len(modes):]+modes[:rep%len(modes)]:
            begin=time.perf_counter_ns();runners[mode]();stop=time.perf_counter_ns()
            report[mode]['times'].append((stop-begin)*1e-9)
    positive=report[3]['values']
    for mode,record in report.items():
        use=positive>0
        record['max_disagreement']=float(np.max(np.abs(record.pop('values')[use]/positive[use]-1)))
        if mode in (2,4):assert record['max_disagreement']<2e-12
        record['median_s']=statistics.median(record['times'])
        record['method']=ROUTES[mode]
    return list(report.values())


def bench_families():
    rows=[]
    for family,sizes in [('complete',[8,16,24,32,48]),('ladder',[32,64,128])]:
        for n in sizes:
            rng=random.Random(455+n)
            edges=ladder(n,4,rng) if family=='ladder' else [(i,j,10.**rng.uniform(-2,2)) for i,j in itertools.combinations(range(n),2)]
            current=[dict(family=family,**rec) for rec in bench(n,edges,[2,3,4,5])]
            rows+=current;write_csv('vertex_matched_families.csv',rows)
            print(family,n,[(r['method'],r['ops'],r['median_s']) for r in current],flush=True)
    from revision_experiments import ladder as table7_ladder
    diagnostic=[]
    for seed,edges in [(1107590+128,table7_ladder(128,4,1107590+128)),
                       (455+128,ladder(128,4,random.Random(455+128)))]:
        stats={}
        joint_kernel(Kernel(128,edges,[[i] for i in range(len(edges))]),stats=stats)
        _,cpp=compiled(128,inputs(edges),2,True)
        assert stats['arithmetic_ops']-stats['reusable_divisions']==cpp['ops']
        diagnostic.append(dict(seed=seed,python_ops=stats['arithmetic_ops'],cpp_ops=cpp['ops'],
                               reused_divisions=stats['reusable_divisions'],pair_updates=stats['pair_updates']))
    write_json('vertex_count_conventions.json',diagnostic)


def components(n,edges,removed):
    adj=[[] for _ in range(n)]
    for i,(u,v,_) in enumerate(edges):
        if i not in removed:
            adj[u].append(v);adj[v].append(u)
    seen=set();sizes=[]
    for i in range(n):
        if i in seen:continue
        stack=[i];seen.add(i);size=0
        while stack:
            u=stack.pop();size+=1
            for v in adj[u]:
                if v not in seen:seen.add(v);stack.append(v)
        sizes.append(size)
    return sorted(sizes)


def original_superlu(n,edges,bridges):
    # Same signed incidence, grounding, ordering and dense RHS blocks as the LODF experiment.
    m=len(edges);u,v,r=inputs(edges)
    B=coo_matrix((np.tile([1.,-1.],m),(np.column_stack([u,v]).ravel(),np.repeat(np.arange(m),2))),shape=(n,m)).tocsc()[:-1,:]
    lu=splu((B.multiply(1/r)@B.T).tocsc())
    rho=np.full(m,np.inf);delta=np.zeros(m)
    connected=np.ones(m,dtype=bool);connected[list(bridges)]=False
    for j in range(0,m,32):
        rhs=B[:,j:j+32].toarray();x=lu.solve(rhs);R=np.sum(rhs*x,axis=0)
        ok=connected[j:j+len(R)]
        d=1-R[ok]/r[j:j+len(R)][ok]
        delta[j:j+len(R)][ok]=d;rho[j:j+len(R)][ok]=R[ok]/d
    return delta,rho,lu.L.nnz,lu.U.nnz


def grid_study():
    ref=json.loads((ROOT/'results/vertex_grids.json').read_text())
    costs=[];examples=[]
    for old in ref:
        case=old['case'];n,edges,source=read_grid(ROOT/'data/pglib'/(case+'.m'))
        bridges,kernels=compress(n,edges)
        ker=max(kernels,key=lambda g:g.n)
        e=source.index(old['selected_row'])
        h=next(i for i,c in enumerate(ker.chains) if e in c)
        degrees,trace=vertex_paths(ker.n,inputs(ker.edges))
        audited,audit=compiled(ker.n,inputs(ker.edges),3)
        np.testing.assert_array_equal(degrees,audited)
        v=ker.n; j=height(v)-1
        pairs=Counter(tuple(sorted((s,t))) for s,t,_ in ker.edges if s!=t)
        mu=max(pairs.values());a=height(mu);H=height(max(map(len,ker.chains)));D=height(v-1)
        def trace_for(edge,idx):
            loop=(ker.edges[idx][0]==ker.edges[idx][1]);Hh=height(len(ker.chains[idx]))
            degree_sum,pivots,levels,max_degree=map(int,trace[idx])
            assert (pivots,levels)==((0,0) if loop else (v-2,j))
            def exponent(q,L):
                return 3+2*Hh if loop else 3+max(2*Hh,H+3+max(a+L+q,2*a))
            uniform=exponent((v-2)*(3+D),j);path=exponent(degree_sum,levels)
            assert path<=uniform
            return dict(outage=source[edge],kernel_edge=idx,loop=loop,chain_length=len(ker.chains[idx]),
                        query_multiplicity=0 if loop else pairs[tuple(sorted(ker.edges[idx][:2]))],
                        degree_sum=degree_sum,pivots=pivots,levels=levels,max_degree=max_degree,
                        uniform_E_delta=uniform,path_E_delta=path,uniform_bound=math.expm1(-uniform*math.log1p(-UNIT)),
                        path_bound=math.expm1(-path*math.log1p(-UNIT)))
        selected=trace_for(e,h)
        monitored=[old['monitor_row']]
        if 'split_flow' in old:monitored.append(old['split_flow']['monitor_row'])
        cuts={}
        for row in monitored:
            f=source.index(row)
            assert components(n,edges,{e})==components(n,edges,{f})==[n]
            sizes=components(n,edges,{e,f})
            cuts[str(row)]=sizes
        runs=[]
        standard,_,nnzl,nnzu=original_superlu(n,edges,bridges)
        assert standard[e]==old['sparse_delta']
        for rep in range(5):
            t=time.perf_counter_ns();original_superlu(n,edges,bridges);runs.append((time.perf_counter_ns()-t)*1e-9)
        extra={}
        if selected['loop']:
            candidates=[]
            for idx,chain in enumerate(ker.chains):
                if ker.edges[idx][0]==ker.edges[idx][1]:continue
                for edge,rest in zip(chain,exclusive_sums([edges[g][2] for g in chain])):
                    r=edges[edge][2];delta=r/(r+(rest+degrees[idx]))
                    candidates.append((delta,edge,idx))
            delta,edge,idx=min(candidates); extra=trace_for(edge,idx)
            with localcontext() as ctx:
                ctx.prec=145
                x,y,r=edges[edge];r=Decimal.from_float(r)
                truth=decimal_solve(n,edges,edge,130)[x]
                check=decimal_solve(n,edges,edge,80)[x]
                reference=r/(r+truth)
                extra.update(delta=float(reference),positive_error=float(abs(Decimal.from_float(delta)/reference-1)),
                             sparse_error=float(abs(Decimal.from_float(float(standard[edge]))/reference-1)),
                             reference_gap=float(abs(check/truth-1)))
        examples.append(dict(case=case,v=v,k=len(ker.edges),padding=2**height(v),H=H,D=D,mu=mu,
                selected=selected,extra_nonloop=extra,cut_sizes=cuts,
                original_SuperLU_times=runs,original_SuperLU_median=statistics.median(runs),Lnnz=nnzl,Unnz=nnzu))
        write_json('vertex_grid_paths.json',examples)
        current=[dict(case=case,**rec) for rec in bench(v,ker.edges,[2,3,6,4])]
        costs+=current;write_csv('vertex_matched_grids.csv',costs)
        print(case,examples[-1],[(r['method'],r['ops'],r['median_s']) for r in current],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('study',choices=['grids','families'])
    opt=p.parse_args()
    if hasattr(os,'sched_getaffinity'):
        # Bind all timed methods and any BLAS workers (launched with one thread) to one CPU.
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    (grid_study if opt.study=='grids' else bench_families)()

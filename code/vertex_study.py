"""Reproduce November vertex/edge/SF-LDL comparisons. Run from any directory.
A C++17 compiler, Python 3, NumPy and SciPy are required; no data are downloaded.
The exact checks and Decimal references do not call the compiled kernels.
"""
from pathlib import Path
import csv, hashlib, heapq, itertools, json, math, os, platform
import random, re, statistics, subprocess, sys, time
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from baselines import grounded_generic
from outage import compress, exclusive_sums, height
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
from vertex_outage import METRICS, FLAGS, compiled, inputs, all_outages_vertex

def write_csv(name,records):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
def write_json(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def err(x,y):return float(abs(D.from_float(float(x))/y-1)) if y else float(abs(D.from_float(float(x))))
def exact_vertex(n,edges):
    """Exact rational vertex recursion for testing the identities, independent of C++."""
    groups={}
    for i,(a,b,r) in enumerate(edges):
        if a!=b:groups.setdefault(tuple(sorted((a,b))),[]).append((i,1/F(r)))
    answers=[F(0)]*len(edges);seen=set()
    def visit(slots,bg):
        if len(slots)==2:
            key=tuple(sorted(slots))
            if key in groups and key not in seen:
                seen.add(key)
                for idx,val in groups[key]:
                    others=sum((x for j,x in groups[key] if j!=idx),F(0))
                    answers[idx]=1/(bg.get(key,0)+others)
            return
        q=len(slots)//4
        for i,j in itertools.combinations(range(4),2):
            child=slots[i*q:(i+1)*q]+slots[j*q:(j+1)*q]; U=set(child)
            if not any(a in U and b in U for a,b in groups.keys()-seen):continue
            c=bg.copy()
            for (a,b),group in groups.items():
                if a in slots and b in slots and not(a in U and b in U):
                    c[a,b]=c.get((a,b),0)+sum((x for _,x in group),F(0))
            for h in slots:
                if h<0 or h in U:continue
                row={a if b==h else b:x for (a,b),x in c.items() if h in (a,b)}
                denom=sum(row.values(),F(0));assert denom>0
                for a,b in itertools.combinations(sorted(row),2):
                    c[a,b]=c.get((a,b),0)+row[a]*(row[b]/denom)
                c={ab:x for ab,x in c.items() if h not in ab}
            visit(child,c)
    if groups:
        cap=1<<max(1,height(n));visit(tuple(range(n))+(-1,)*(cap-n),{})
    assert len(seen)==len(groups)
    return answers

def exact_suite():
    rng=random.Random(11837); records=[]; loops=0
    for n in range(2,11):
        for run in range(5):
            edges=[(i,(i+1)%n,F(rng.randrange(1,25))) for i in range(n)]
            for j in range(n+2):
                a,b=rng.sample(range(n),2);edges.append((a,b,F(rng.randrange(1,27))))
            edges.append((0,1,F(10**12)));edges.append((0,0,F(9)));loops+=1
            rng.shuffle(edges); e3=exact_vertex(n,edges); arrays=inputs(edges)
            comp={mode:compiled(n,arrays,mode)[0] for mode in (2,3,4)}
            for eid,(a,b,r) in enumerate(edges):
                if a==b:
                    assert all(x[eid]==0 for x in comp.values()) and e3[eid]==0;continue
                truth=grounded_generic(n,edges,a,b,eid,F);assert truth==e3[eid]
                row={'v':n,'sample':run,'edge':eid,'exact_beta':str(truth)}
                for mode,values in comp.items(): row[f'error_{mode}']=float(abs(F.from_float(values[eid])/truth-1))
                records.append(row)
    write_csv('vertex_exact.csv',records)
    special=[(0,1,1.),(0,1,2.),(1,2,3.),(2,3,4.),(3,1,5.),(3,4,1.)]
    bridges,ks=compress(5,special); assert bridges=={5}
    for K in ks:
        ref=exact_vertex(K.n,K.edges);calc=compiled(K.n,inputs(K.edges),3)[0]
        for i,chain in enumerate(K.chains):
            for e,rest in zip(chain,exclusive_sums([F(special[j][2]) for j in chain])):
                a,b,r=special[e];truth=grounded_generic(5,special,a,b,e,F)
                assert rest+ref[i]==truth and abs(float(truth)-(float(rest)+calc[i]))<1e-12
    return {'graphs':45,'nonloop_queries':len(records),'loop_checks':loops,'mixed_original_edges':6,
            'binary64_max':{m:max(p[f'error_{m}'] for p in records) for m in (2,3,4)}}

def decimal_solve(n,edges,eid,precision,deleted=True):
    """Signed sparse grounded LDL elimination with minimum-degree ordering.
    Returns potentials for unit injection at the endpoints. Inputs are exact binary64.
    """
    with localcontext() as ctx:
        ctx.prec=precision
        s,ground,_=edges[eid]; adj=[{} for _ in range(n)];diag=[D(0)]*n;rhs=[D(0)]*n;rhs[s]=D(1)
        for j,(x,y,r) in enumerate(edges):
            if (deleted and j==eid) or x==y:continue
            g=D(1)/D.from_float(float(r))
            if x!=ground:diag[x]+=g
            if y!=ground:diag[y]+=g
            if ground not in (x,y):
                adj[x][y]=adj[y][x]=adj[x].get(y,D(0))-g
        queue=[(len(adj[v]),v) for v in range(n) if v!=ground];heapq.heapify(queue)
        keep=[True]*n;keep[ground]=False;history=[]
        while queue:
            sz,v=heapq.heappop(queue)
            if not keep[v] or len(adj[v])!=sz:continue
            row=list(adj[v].items());p=diag[v]
            if p<=0:raise ArithmeticError('Reference precision insufficient')
            history.append((v,p,rhs[v],row));keep[v]=False
            for ii,(x,ax) in enumerate(row):
                mult=ax/p;diag[x]-=mult*ax;rhs[x]-=mult*rhs[v]
                del adj[x][v]
                for y,ay in row[ii+1:]:
                    adj[x][y]=adj[y][x]=adj[x].get(y,D(0))-mult*ay
            for x,_ in row:heapq.heappush(queue,(len(adj[x]),x))
            adj[v].clear()
        sol=[D(0)]*n
        for v,p,b,row in reversed(history):sol[v]=(b-sum((val*sol[x] for x,val in row),D(0)))/p
        return sol

def ladder(n,spread,rng):
    half=n//2
    links=[(i,(i+1)%half) for i in range(half)]+[(half+i,half+(i+1)%half) for i in range(half)]+[(i,i+half) for i in range(half)]
    return [(a,b,10.**rng.uniform(-spread,spread)) for a,b in links]

def large():
    rows=[];per_query=[]
    for n in [128,256,384]:
        for spread in [8,32]:
            ed=ladder(n,spread,random.Random(48600+n*37+spread));ar=inputs(ed)
            beta,mt=compiled(n,ar,3);orig=ar[2];comp=orig/(orig+beta)
            # Kernel audit covers every arithmetic operation of Algorithm 3 on these singleton chains.
            post=np.concatenate((orig+beta,comp))
            positives_zero=int(np.count_nonzero(comp==0))
            sub=int(np.count_nonzero((post>0)&(post<np.finfo(float).tiny)))
            samples=sorted(set([0,len(ed)//3,len(ed)//2,len(ed)-1,int(np.argmin(comp))]))
            for e in samples:
                x,y,r=ed[e]; refs=[decimal_solve(n,ed,e,p)[x] for p in (130,210)]
                with localcontext() as ctx:
                    ctx.prec=215;truth=refs[-1];delta=D.from_float(r)/(D.from_float(r)+truth)
                    per_query.append(dict(v=n,spread=spread,edge=e,rho_ref=str(truth),delta_ref=str(delta),
                      rho_error=err(beta[e],truth),delta_error=err(comp[e],delta),precision_gap=float(abs(refs[0]/truth-1))))
            sample=[p for p in per_query if p['v']==n and p['spread']==spread]
            J=max(0,height(n)-1);q=3+height(max(1,n-1))
            boundE=3+max(J+(n-2)*q,0)+1 # H=0, A=0, one final addition and reciprocal and reconstruction
            rows.append(dict(v=n,spread=spread,queries=len(ed),reference_queries=len(samples),
                max_rho=max(p['rho_error'] for p in sample),max_delta=max(p['delta_error'] for p in sample),
                reference_gap=max(p['precision_gap'] for p in sample),min_delta=float(comp.min()),
                exponent_rho=boundE,**mt,reconstruction_subnormal=sub,reconstruction_zero=positives_zero))
            print('large',rows[-1],flush=True)
    write_csv('vertex_wide.csv',rows);write_csv('vertex_wide_references.csv',per_query);return rows

def timing():
    rows=[]
    for family,sizes in [('complete',[8,16,24,32,48]),('ladder',[32,64,128])]:
        for n in sizes:
            rng=random.Random(455+n);edges=ladder(n,4,rng) if family=='ladder' else [(i,j,10.**rng.uniform(-2,2)) for i,j in itertools.combinations(range(n),2)]
            ar=inputs(edges);values={}
            for mode in (2,3,4):
                vals,stats=compiled(n,ar,mode);values[mode]=vals
                compiled(n,ar,mode,False);times=[]
                for rep in range(5):
                    start=time.perf_counter_ns();compiled(n,ar,mode,False);times.append((time.perf_counter_ns()-start)*1e-9)
                rows.append(dict(family=family,v=n,k=len(edges),algorithm=mode,median_s=statistics.median(times),**stats,raw_seconds=' '.join(map(str,times))))
            assert np.max(np.abs(values[2]/values[3]-1))<1e-12 and np.max(np.abs(values[4]/values[3]-1))<1e-12
            print('timing',family,n,flush=True)
    write_csv('vertex_compiled_timings.csv',rows);return rows

def read_grid(path):
    t=path.read_text()
    def tab(k):
        b=re.search(r'^mpc\.'+k+r'\s*=\s*\[(.*?)\];',t,re.S|re.M)[1]
        return [[float(x) for x in l.split('%')[0].strip().strip(';').split()] for l in b.splitlines() if l.split('%')[0].strip()]
    buses=tab('bus');ids={int(row[0]):i for i,row in enumerate(buses)};edges=[];source=[]
    for rownum,row in enumerate(tab('branch'),1):
        if row[10]<=0:continue
        r=row[3]*(row[8] or 1.)
        if r<=0:raise ValueError('Entire case lies outside the positive model')
        edges.append((ids[int(row[0])],ids[int(row[1])],r));source.append(rownum)
    return len(buses),edges,source

def grid_case(path):
    n,edges,source=read_grid(path);m=len(edges);bridges,kernels=compress(n,edges)
    u,w,r=inputs(edges);B=coo_matrix((np.tile([1.,-1.],m),(np.column_stack([u,w]).ravel(),np.repeat(np.arange(m),2))),shape=(n,m)).tocsc()[:-1,:]
    lu=splu((B.multiply(1/r)@B.T).tocsc());standard=np.full(m,np.nan)
    for j in range(0,m,32):
        rhs=B[:,j:j+32].toarray();x=lu.solve(rhs)
        standard[j:j+32]=1-np.sum(rhs*x,axis=0)/r[j:j+32]
    rho=np.full(m,np.inf);audit={key:0 for key in METRICS}
    for ker in kernels:
        b,stats=compiled(ker.n,inputs(ker.edges),3)
        for met in METRICS:audit[met]+=stats[met]
        for h,chain in enumerate(ker.chains):
            for e,rest in zip(chain,exclusive_sums([edges[i][2] for i in chain])):rho[e]=float(rest)+b[h]
    positive=r/(r+rho);e=min((j for j in range(m) if j not in bridges),key=lambda j:positive[j])
    s,t,res=edges[e]
    refs=[decimal_solve(n,edges,e,p) for p in (80,130)]
    with localcontext() as ctx:
        ctx.prec=135;truth=refs[-1][s];dref=D.from_float(res)/(D.from_float(res)+truth)
        ref_flow=[(refs[-1][a]-refs[-1][b])/D.from_float(rr) if j!=e else D(0) for j,(a,b,rr) in enumerate(edges)]
        monitor=max((j for j in range(m) if j!=e),key=lambda j:abs(ref_flow[j]));lodf=ref_flow[monitor]
        # Polarities follow incidence (from,to). The independent H_fe comes from the intact Decimal system.
        original=decimal_solve(n,edges,e,130,False);a,b,z=edges[monitor]
        numerator=(original[a]-original[b])/D.from_float(z)
        rhs=B[:,e].toarray().ravel();xb=np.append(lu.solve(rhs),0.)
        numer64=float((xb[a]-xb[b])/z)
        result=dict(case=path.stem,n=n,m=m,bridges=len(bridges),kernel_sizes=[(g.n,len(g.edges),max(map(len,g.chains))) for g in kernels],
              selected_row=source[e],monitor_row=source[monitor],positive_delta=float(positive[e]),reference_delta=str(dref),
              sparse_delta=float(standard[e]),positive_delta_error=err(positive[e],dref),sparse_delta_error=err(standard[e],dref),
              lodf=str(lodf),reference_precision_gap=float(abs(refs[0][s]/truth-1)),check_LODF_identity=float(abs(numerator/dref/lodf-1)),
              fixed_num_error_positive=float(abs(numerator/D.from_float(float(positive[e]))/lodf-1)),
              fixed_num_error_sparse=float(abs(numerator/D.from_float(float(standard[e]))/lodf-1)),
              common_binary64_numerator=numer64,full_ratio_error_positive=err(numer64/positive[e],lodf),full_ratio_error_sparse=err(numer64/standard[e],lodf),audit=audit,
              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # A second monitor measures an incomplete redistribution on a different branch.
        partial = [j for j in range(m) if j != e and D("0.05") <= abs(ref_flow[j]) <= D("0.95")]
        if partial:
            j = min(partial, key=lambda j: (abs(abs(ref_flow[j])-D("0.5")), j))
            x, y, resistance = edges[j]
            exact_num = (original[x]-original[y])/D.from_float(resistance)
            common_num = float((xb[x]-xb[y])/resistance)
            target = ref_flow[j]
            result["split_flow"] = dict(monitor_row=source[j], lodf=str(target),
                high_precision_identity_gap=float(abs(exact_num/dref/target-1)),
                common_binary64_numerator=common_num,
                full_ratio_error_sparse=err(common_num/standard[e], target),
                full_ratio_error_positive=err(common_num/positive[e], target))
            assert result["split_flow"]["high_precision_identity_gap"] < 1e-90
        assert result['reference_precision_gap']<1e-65 and result['check_LODF_identity']<1e-90
    print('grid',path.name,result['selected_row'],result['reference_delta'],result['positive_delta_error'],result['sparse_delta_error'],flush=True)
    return result

def parallel_ring():
    results=[]
    for n in [128,256,384]:
        gen=random.Random(745+n);edges=[]
        for i in range(n):
            edges.extend((i,(i+1)%n,10.**gen.uniform(-32,32)) for j in range(2))
        profile,stats=all_outages_vertex(n,edges)
        with localcontext() as ctx:
            ctx.prec=180
            R=[D.from_float(r) for a,b,r in edges]
            links=[1/(1/R[i]+1/R[i+1]) for i in range(0,len(R),2)]
            rr=[];dd=[]
            for e in range(len(R)):
                j=e//2;path=sum(links[:j]+links[j+1:]);other=R[e^1]
                bypass=1/(1/path+1/other);d=R[e]/(R[e]+bypass)
                rr.append(err(profile.bypass[e],bypass));dd.append(err(profile.defect[e],d))
            results.append(dict(v=n,queries=len(R),max_rho_error=max(rr),max_delta_error=max(dd),min_delta=min(profile.defect),
                    global_rho_exponent=max(profile.error_exponents),max_error_to_bound=max(max(rr)/math.expm1(-max(profile.error_exponents)*math.log1p(-2**-53)),max(dd)/math.expm1(-(2+max(profile.error_exponents))*math.log1p(-2**-53))),**stats))
    write_csv('vertex_parallel_rings.csv',results)
def inverse_comparison():
    from scipy.linalg import cho_factor,cho_solve
    rows=[]
    for n in [8,16,24,32,48]:
        rng=random.Random(455+n);ed=[(i,j,10.**rng.uniform(-2,2)) for i,j in itertools.combinations(range(n),2)]
        a,b,r=inputs(ed);ref,_=compiled(n,(a,b,r),3)
        def full_inverse():
            g=1/r
            L=np.zeros((n,n));np.add.at(L,(a,a),g);np.add.at(L,(b,b),g)
            L[a,b]=-g;L[b,a]=-g
            Q=np.zeros_like(L)
            Q[:-1,:-1]=cho_solve(cho_factor(L[:-1,:-1],check_finite=False),np.eye(n-1),check_finite=False)
            R=Q[a,a]+Q[b,b]-2*Q[a,b]
            return R/(1-R/r)
        for name,fn in [('vertex',lambda:compiled(n,(a,b,r),3,False)[0]),('inverse',full_inverse)]:
            values=fn();ts=[]
            for rep in range(7):
                t=time.perf_counter_ns();fn();ts.append((time.perf_counter_ns()-t)*1e-9)
            rows.append(dict(v=n,method=name,median_s=statistics.median(ts),raw_seconds=' '.join(map(str,ts)),max_disagreement=float(max(abs(values/ref-1)))))
    write_csv('vertex_inverse_timings.csv',rows)


def archived_exact():
    """Use the existing independent enumeration, oracles and per-query bound checks."""
    from verify import main as verify_all
    from outage import Profile
    def profile(n, edges):
        if not edges or not isinstance(edges[0][2], F):
            return all_outages_vertex(n, edges)[0]
        bridges, kernels = compress(n, edges)
        resistances = [math.inf] * len(edges)
        complements = [F(0)] * len(edges)
        for ker in kernels:
            betas = exact_vertex(ker.n, ker.edges)
            for chain, beta in zip(ker.chains, betas):
                amounts = [edges[i][2] for i in chain]
                for i, other in zip(chain, exclusive_sums(amounts)):
                    resistances[i] = other + beta
                    complements[i] = edges[i][2] / (edges[i][2] + resistances[i])
        return Profile(resistances, complements, bridges, kernels, [0]*len(edges))
    verify_all(profile, "vertex_original_verification.json")


def main():
    if len(sys.argv)>1:sections=set(sys.argv[1:])
    else:sections={'exact','timing','large','grid','range','rings','inverse','original'}
    if 'exact' in sections: print(ex:=exact_suite(),flush=True);write_json('vertex_exact_summary.json',ex)
    if 'original' in sections:archived_exact()
    if 'timing' in sections:timing()
    if 'large' in sections:large()
    if 'grid' in sections:write_json('vertex_grids.json',[grid_case(ROOT/'data/pglib'/f) for f in ('pglib_opf_case1354_pegase.m','pglib_opf_case2383wp_k.m')])
    if 'rings' in sections:parallel_ring()
    if 'inverse' in sections:inverse_comparison()
    if 'range' in sections:
        stress=[(0,1,1e-250),(1,2,1e200),(1,3,1e200),(0,2,1.),(0,3,1.),(2,3,1.)]
        vals,metrics=compiled(4,inputs(stress),3)
        assert metrics['positive_to_zero']+metrics['subnormal']>0
        _, fast = compiled(4, inputs(stress), 3, audit=False)
        assert fast['hw_underflow'] == 1
        try:
            all_outages_vertex(4, stress)
        except FloatingPointError:
            strict_caught = True
        else:
            strict_caught = False
        assert strict_caught
        write_json('vertex_range_check.json',
                   dict(**metrics, unaudited_hw_flag=fast['hw_underflow'],
                        full_solver_raised=strict_caught))
    write_json('vertex_environment.json',dict(compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],flags=FLAGS,platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=__import__('scipy').__version__,thread_settings={name:os.getenv(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unspecified')))
if __name__=='__main__':main()

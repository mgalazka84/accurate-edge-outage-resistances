"""Additional baselines for the supplied major-revision comments.

SF-LDL is our implementation of the subtraction-free Laplacian factorization
principle in Orera--Pena (2024), Algorithm 1 / Theorem 1. The ground is placed
last; pivots are recomputed from off-diagonal conductances, and positive
triangular solves recover the diagonal of the grounded inverse. This is not
the authors' software. Decimal routines are conventional nodal elimination,
not subtraction-free elimination. IR is direct Cholesky refinement, not SMIR.
"""
from decimal import Decimal, localcontext
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from outage import pair_sum, find_bridges, compress, exclusive_sums
from baselines import decimal_reference


def sf_ldl(n, edges, s, t, omit=None):
    if s == t:
        return 0.0
    order = [i for i in range(n) if i != t] + [t]
    ix = {v:i for i,v in enumerate(order)}
    groups = {}
    for eid,(a,b,r) in enumerate(edges):
        if eid == omit or a == b:
            continue
        a,b = sorted((ix[a],ix[b]))
        groups.setdefault((a,b),[]).append(1.0/r)
    c = np.zeros((n,n))
    for (a,b),vals in groups.items():
        c[a,b] = c[b,a] = pair_sum(vals)
    p = np.zeros((n-1,n-1))  # p = I-L on the strictly lower triangle
    d = np.zeros(n-1)
    for k in range(n-1):
        v = c[k,k+1:].copy()
        d[k] = pair_sum(v.tolist())
        if not d[k] > 0 or not np.isfinite(d[k]):
            raise ArithmeticError('Nonpositive/nonfinite SF-LDL pivot')
        multipliers = v/d[k]
        p[k+1:,k] = multipliers[:-1]
        # Store each undirected entry once, then mirror, avoiding asymmetry
        # caused by evaluating the two possible products in different orders.
        ii,jj = np.triu_indices(len(v),1)
        new = c[k+1+ii,k+1+jj] + multipliers[ii]*v[jj]
        c[k+1+ii,k+1+jj] = new
        c[k+1+jj,k+1+ii] = new
    y = np.zeros(n-1)
    source = ix[s]
    for i in range(n-1):
        y[i] = (1.0 if i == source else 0.0) + float(p[i,:i] @ y[:i])
    x = np.zeros(n-1)
    for i in range(n-2,-1,-1):
        x[i] = y[i]/d[i] + float(p[i+1:,i] @ x[i+1:])
    return float(x[source])


def compressed_sf_ldl(n, edges):
    bridges,kernels = compress(n,edges)
    rho = [math.inf]*len(edges)
    for ker in kernels:
        for h,(s,t,_) in enumerate(ker.edges):
            beta = sf_ldl(ker.n,ker.edges,s,t,omit=h) if s != t else 0.0
            vals = exclusive_sums([edges[e][2] for e in ker.chains[h]])
            for e,rest in zip(ker.chains[h],vals):
                rho[e] = rest + beta
    return rho


def full_profile(n, edges, solver):
    bridges = find_bridges(n,edges)
    values = []
    for eid,(s,t,_) in enumerate(edges):
        if eid in bridges:
            values.append(math.inf)
        else:
            try:
                values.append(solver(n,edges,s,t,omit=eid))
            except (ArithmeticError,ValueError,np.linalg.LinAlgError):
                values.append(math.nan)
    return values


def decimal_direct(n,edges,s,t,omit=None,precision=35):
    return decimal_reference(n,edges,s,t,omit,precision)


def decimal_update_profile(n,edges,precision=35):
    """One ordinary Decimal LU, reused for all endpoint identities.

    The node n-1 is grounded. Inputs are exact conversions of binary64
    resistance values. Arithmetic, resistance formation, and complement
    subtraction all use the requested precision; no binary64 round trip.
    """
    bridges = find_bridges(n,edges)
    q = n-1
    with localcontext() as ctx:
        ctx.prec = precision
        a = [[Decimal(0) for _ in range(q)] for _ in range(q)]
        for v,w,r in edges:
            g = Decimal(1)/Decimal(r)
            if v<q: a[v][v] += g
            if w<q: a[w][w] += g
            if v<q and w<q:
                a[v][w] -= g; a[w][v] -= g
        try:
            for j in range(q):
                if a[j][j] <= 0:
                    raise ArithmeticError('Nonpositive conventional LU pivot')
                for i in range(j+1,q):
                    f = a[i][j]/a[j][j]
                    a[i][j] = f
                    if f:
                        for k in range(j+1,q):
                            a[i][k] -= f*a[j][k]
        except ArithmeticError:
            return [Decimal('Infinity') if i in bridges else Decimal('NaN') for i in range(len(edges))]
        out = []
        for eid,(s,t,r) in enumerate(edges):
            if eid in bridges:
                out.append(Decimal('Infinity')); continue
            b = [Decimal(int(i==s)-int(i==t)) for i in range(q)]
            for i in range(q):
                b[i] -= sum(a[i][j]*b[j] for j in range(i))
            x = [Decimal(0)]*q
            for i in range(q-1,-1,-1):
                x[i] = (b[i]-sum(a[i][j]*x[j] for j in range(i+1,q)))/a[i][i]
            resistance = (x[s] if s<q else Decimal(0))-(x[t] if t<q else Decimal(0))
            defect = 1-resistance/Decimal(r)
            out.append(+(resistance/defect) if defect>0 and resistance>0 else Decimal('NaN'))
        return out


def refined_cholesky(n,edges,s,t,omit=None,steps=5):
    """Binary64 factors/corrections, Decimal(80) residuals from edge data.

    At most five corrections; stop only when the binary64 iterate is unchanged.
    A failed factorization is reported as failure rather than repaired.
    """
    keep = [v for v in range(n) if v != t]
    ix = {v:i for i,v in enumerate(keep)}
    q=n-1
    a = np.zeros((q,q)); rhs=np.zeros(q); rhs[ix[s]]=1
    active=[]
    for eid,(v,w,r) in enumerate(edges):
        if eid==omit or v==w: continue
        g=1.0/r
        if v!=t: a[ix[v],ix[v]]+=g
        if w!=t: a[ix[w],ix[w]]+=g
        if v!=t and w!=t:
            a[ix[v],ix[w]]-=g; a[ix[w],ix[v]]-=g
        active.append((v,w,r))
    fac=cho_factor(a,lower=True,check_finite=False)
    x=cho_solve(fac,rhs,check_finite=False)
    with localcontext() as ctx:
        ctx.prec=80
        conductances=[(v,w,Decimal(1)/Decimal(r)) for v,w,r in active]
        for _ in range(steps):
            if not np.all(np.isfinite(x)): return math.nan
            xd=[Decimal(float(val)) for val in x]
            residual=[Decimal(int(v==s)) for v in keep]
            for v,w,g in conductances:
                voltage=(xd[ix[v]] if v!=t else Decimal(0))-(xd[ix[w]] if w!=t else Decimal(0))
                current=g*voltage
                if v!=t: residual[ix[v]]-=current
                if w!=t: residual[ix[w]]+=current
            correction=cho_solve(fac,np.array([float(z) for z in residual]),check_finite=False)
            new=x+correction
            if np.array_equal(new,x): break
            x=new
    return float(x[ix[s]])

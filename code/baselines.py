"""Independent grounded-system references and comparison implementations."""
from decimal import Decimal, localcontext
from fractions import Fraction
from math import inf
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from outage import find_bridges


def grounded_float(n, edges, s, t, omit=None):
    if s == t:
        return 0.0
    keep = [v for v in range(n) if v != t]
    ix = {v: j for j, v in enumerate(keep)}
    a = np.zeros((n-1, n-1))
    for eid, (v, w, r) in enumerate(edges):
        if eid == omit or v == w:
            continue
        g = 1.0 / r
        if v != t:
            a[ix[v], ix[v]] += g
        if w != t:
            a[ix[w], ix[w]] += g
        if v != t and w != t:
            a[ix[v], ix[w]] -= g
            a[ix[w], ix[v]] -= g
    rhs = np.zeros(n-1)
    rhs[ix[s]] = 1.0
    x = cho_solve(cho_factor(a, lower=True, check_finite=False), rhs, check_finite=False)
    return float(x[ix[s]])


def grounded_generic(n, edges, s, t, omit=None, factory=Fraction):
    """Gaussian elimination of a grounded nodal matrix; independent of compression."""
    if s == t:
        return factory(0)
    keep = [v for v in range(n) if v != t]
    ix = {v: j for j, v in enumerate(keep)}
    a = [[factory(0) for _ in keep] for _ in keep]
    for eid, (v, w, r) in enumerate(edges):
        if eid == omit or v == w:
            continue
        g = factory(1) / factory(r)
        if v != t:
            a[ix[v]][ix[v]] += g
        if w != t:
            a[ix[w]][ix[w]] += g
        if v != t and w != t:
            a[ix[v]][ix[w]] -= g
            a[ix[w]][ix[v]] -= g
    rhs = [factory(0) for _ in keep]
    rhs[ix[s]] = factory(1)
    size = len(keep)
    for j in range(size):
        if a[j][j] == 0:
            raise ArithmeticError('Singular grounded reference matrix')
        for i in range(j+1, size):
            f = a[i][j] / a[j][j]
            if f == 0:
                continue
            for k in range(j+1, size):
                a[i][k] -= f*a[j][k]
            rhs[i] -= f*rhs[j]
    x = [factory(0) for _ in keep]
    for i in range(size-1, -1, -1):
        x[i] = (rhs[i] - sum(a[i][j]*x[j] for j in range(i+1, size))) / a[i][i]
    return x[ix[s]]


def decimal_reference(n, edges, s, t, omit=None, precision=100):
    with localcontext() as ctx:
        ctx.prec = precision
        return +grounded_generic(n, edges, s, t, omit, Decimal)


def sm_profile(n, edges):
    """One original-graph grounded Cholesky, then endpoint rank-one identities.

    Each incidence solve has the same factorization. Returns per-edge errors
    as NaN rather than repairing nonpositive computed denominators.
    """
    ground = n-1
    a = np.zeros((n-1, n-1))
    for v, w, r in edges:
        g = 1.0/r
        if v != ground:
            a[v,v] += g
        if w != ground:
            a[w,w] += g
        if v != ground and w != ground:
            a[v,w] -= g
            a[w,v] -= g
    bridges = find_bridges(n, edges)
    rho, delta = [], []
    try:
        factor = cho_factor(a, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return [np.nan]*len(edges), [np.nan]*len(edges)
    for eid, (v, w, r) in enumerate(edges):
        if eid in bridges:
            rho.append(inf); delta.append(0.0); continue
        b = np.zeros(n-1)
        if v != ground: b[v] += 1.0
        if w != ground: b[w] -= 1.0
        x = cho_solve(factor, b, check_finite=False)
        resistance = float(b @ x)
        defect = 1.0 - resistance/r
        delta.append(defect)
        rho.append(resistance/defect if defect > 0 else np.nan)
    return rho, delta


def sparse_sm_profile(n, edges):
    """One sparse SuperLU factorization followed by endpoint incidence solves."""
    rows, cols, data = [], [], []
    ground=n-1
    for v,w,r in edges:
        g=1.0/r
        if v!=ground: rows.append(v);cols.append(v);data.append(g)
        if w!=ground: rows.append(w);cols.append(w);data.append(g)
        if v!=ground and w!=ground:
            rows.extend([v,w]);cols.extend([w,v]);data.extend([-g,-g])
    a=coo_matrix((data,(rows,cols)),shape=(n-1,n-1)).tocsc()
    bridges=find_bridges(n,edges)
    try: factor=splu(a)
    except RuntimeError: return [np.nan]*len(edges),[np.nan]*len(edges)
    rho,delta=[],[]
    b=np.zeros(n-1)
    for eid,(v,w,r) in enumerate(edges):
        if eid in bridges:
            rho.append(inf);delta.append(0.0);continue
        b.fill(0.0)
        if v!=ground:b[v]=1.0
        if w!=ground:b[w]=-1.0
        x=factor.solve(b)
        resistance=(x[v] if v!=ground else 0)-(x[w] if w!=ground else 0)
        defect=1.0-resistance/r
        delta.append(defect)
        rho.append(resistance/defect if defect>0 and resistance>0 else np.nan)
    return rho,delta

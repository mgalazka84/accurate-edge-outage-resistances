"""All single-edge endpoint outage resistances by series compression.

Research implementation, Python 3.10+. No dependencies for the main algorithm.
Input weights are RESISTANCES, not conductances. Graphs may have parallel edges
but must be connected and have no self-loops. Each outage is evaluated separately
from the original graph; this is not a sequential-deletion algorithm.
"""
from dataclasses import dataclass
from math import ceil, log2, inf, isfinite


def height(n):
    return (max(1, n) - 1).bit_length()


def pair_sum(values):
    values = list(values)
    if not values:
        return 0
    while len(values) > 1:
        values = [values[i] + values[i + 1] if i + 1 < len(values)
                  else values[i] for i in range(0, len(values), 2)]
    return values[0]


def exclusive_sums(values):
    """All sums with one input excluded, O(n) work, addition depth <=2 ceil(log2 n)."""
    n = len(values)
    p = 1 << height(n)
    tree = [0] * (2 * p)
    tree[p:p+n] = values
    for i in range(p - 1, 0, -1):
        tree[i] = tree[2*i] + tree[2*i+1]
    outside = [0] * (2 * p)
    for i in range(1, p):
        outside[2*i] = outside[i] + tree[2*i+1]
        outside[2*i+1] = outside[i] + tree[2*i]
    return outside[p:p+n]


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for i, (a, b, r) in enumerate(edges):
        if not (0 <= a < n and 0 <= b < n) or a == b or r <= 0:
            raise ValueError('Expected positive-resistance, loop-free graph')
        adj[a].append((b, i))
        adj[b].append((a, i))
    return adj


def find_bridges(n, edges, adj=None):
    """Iterative Tarjan DFS, including correct handling of parallel edges."""
    adj = adjacency(n, edges) if adj is None else adj
    if n == 0:
        raise ValueError('Empty graph')
    tin, low = [-1]*n, [-1]*n
    parent, pe = [-1]*n, [-1]*n
    clock = 0
    bridges = set()
    tin[0] = low[0] = clock
    stack = [(0, 0)]
    while stack:
        v, pos = stack[-1]
        if pos == len(adj[v]):
            stack.pop()
            if parent[v] != -1:
                p = parent[v]
                low[p] = min(low[p], low[v])
                if low[v] > tin[p]:
                    bridges.add(pe[v])
            continue
        u, eid = adj[v][pos]
        stack[-1] = (v, pos+1)
        if eid == pe[v]:
            continue
        if tin[u] == -1:
            clock += 1
            tin[u] = low[u] = clock
            parent[u], pe[u] = v, eid
            stack.append((u, 0))
        else:
            low[v] = min(low[v], tin[u])
    if any(t == -1 for t in tin):
        raise ValueError('Input graph must be connected')
    return bridges


@dataclass
class Kernel:
    n: int
    edges: list
    chains: list

    @property
    def rank(self):
        return len(self.edges) - self.n + 1


def compress(n, edges):
    adj = adjacency(n, edges)
    bridges = find_bridges(n, edges, adj)
    cadj = [[(v, e) for v, e in nei if e not in bridges] for nei in adj]
    seen_v, used_e, kernels = set(), set(), []
    for root in range(n):
        if root in seen_v or not cadj[root]:
            continue
        comp, todo = [], [root]
        seen_v.add(root)
        while todo:
            v = todo.pop()
            comp.append(v)
            for u, _ in cadj[v]:
                if u not in seen_v:
                    seen_v.add(u)
                    todo.append(u)
        anchors = sorted(v for v in comp if len(cadj[v]) != 2)
        if not anchors:
            anchors = [min(comp)]
        amap = {v: i for i, v in enumerate(anchors)}
        kedges, chains = [], []
        for a in anchors:
            for u, eid in cadj[a]:
                if eid in used_e:
                    continue
                chain = [eid]
                used_e.add(eid)
                prev_e, v = eid, u
                while v not in amap:
                    nex = [(w, e) for w, e in cadj[v] if e != prev_e]
                    assert len(nex) == 1
                    v, prev_e = nex[0]
                    chain.append(prev_e)
                    used_e.add(prev_e)
                resistance = pair_sum(edges[e][2] for e in chain)
                kedges.append((amap[a], amap[v], resistance))
                chains.append(chain)
        kernels.append(Kernel(len(anchors), kedges, chains))
    assert used_e | bridges == set(range(len(edges)))
    return bridges, kernels


def positive_elimination(n, edges, s, t, omit=None):
    """Classical star-mesh elimination, retaining s,t. Generic positive arithmetic.

    Conductance sums use pairwise summation. Self-loops are ignored. Parallel
    edges are combined by positive addition; an omitted edge is skipped, not
    subtracted. No graph compression is performed inside this routine.
    """
    if s == t:
        return 0
    groups = {}
    for eid, (a, b, r) in enumerate(edges):
        if eid == omit or a == b:
            continue
        a, b = min(a, b), max(a, b)
        groups.setdefault((a, b), []).append(1 / r)
    c = [[0] * n for _ in range(n)]
    for (a, b), values in groups.items():
        c[a][b] = c[b][a] = pair_sum(values)
    active = set(range(n))
    # Fixed vertex ordering; no numerical pivot or tolerance decides topology.
    for v in range(n):
        if v in (s, t):
            continue
        nei = [i for i in sorted(active) if i != v and c[v][i] > 0]
        total = pair_sum(c[v][i] for i in nei)
        if total:
            for jpos, j in enumerate(nei):
                for i in nei[:jpos]:
                    increment = c[i][v] * (c[j][v] / total)
                    if increment == 0:
                        raise FloatingPointError('Underflow in positive elimination')
                    c[i][j] = c[j][i] = c[i][j] + increment
        active.remove(v)
    if c[s][t] == 0:
        return inf
    return 1 / c[s][t]


@dataclass
class Profile:
    bypass: list
    defect: list
    bridges: set
    kernels: list
    error_exponents: list


def all_outages(n, edges, kernel_solver=positive_elimination):
    """Return rho_e = R_(G-e)(ends(e)) and delta_e = r_e/(r_e+rho_e)."""
    bridges, kernels = compress(n, edges)
    m = len(edges)
    rho, delta, exponents = [inf]*m, [0]*m, [0]*m
    for kernel in kernels:
        v, k = kernel.n, len(kernel.edges)
        h = max(height(len(ch)) for ch in kernel.chains)
        elim = max(0, v-2) * (3 + height(max(1, v-1)))
        macro_exp = h + height(k) + 2 + elim
        for j, (a, b, _) in enumerate(kernel.edges):
            bpass = kernel_solver(v, kernel.edges, a, b, omit=j) if a != b else 0
            exc = exclusive_sums([edges[e][2] for e in kernel.chains[j]])
            exponent = max(2*h, macro_exp) + 1 if a != b else 2*h + 1
            for eid, rest in zip(kernel.chains[j], exc):
                rho[eid] = rest + bpass
                r = edges[eid][2]
                delta[eid] = r / (r + rho[eid])
                exponents[eid] = exponent
                if not (rho[eid] > 0 and isfinite(rho[eid]) and delta[eid] > 0):
                    raise FloatingPointError('Result outside supported floating-point range')
    return Profile(rho, delta, bridges, kernels, exponents)


if __name__ == '__main__':
    edges = [(0, 1, 1.0), (1, 2, 1e16), (2, 0, 1e16)]
    p = all_outages(3, edges)
    print('bypass resistances:', p.bypass)
    print('leverage-score complements:', p.defect)

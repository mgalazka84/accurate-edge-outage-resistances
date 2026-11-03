"""Subtraction-free shared Schur reductions for all endpoint outages.

The query set is excluded from each node's background network. Its two
children independently add the sibling query edges before eliminating vertices.
Resistance inputs and the original graph are never modified. Fraction inputs
give an exact-arithmetic implementation of the same operations.
"""
from math import inf, isfinite
from outage import compress, exclusive_sums, height, pair_sum, Profile


def joint_kernel(kernel, chain_height=0, stats=None):
    """Return bypass resistances and per-query exponents for one kernel.

Loops have bypass zero. Nonloop edges of a bridgeless kernel have positive
finite bypasses. Vertex order and balanced query splits are deterministic.
"""
    edges = kernel.edges
    query = [i for i, (a, b, _) in enumerate(edges) if a != b]
    beta, exponent = [0] * len(edges), [0] * len(edges)
    if stats is None:
        stats = {}
    for key in ('nodes', 'eliminations', 'pair_updates', 'arithmetic_ops',
                'matrix_entries', 'max_depth'):
        stats.setdefault(key, 0)
    if not query:
        return beta, exponent
    weights = {i: 1 / edges[i][2] for i in query}
    stats['arithmetic_ops'] += len(query)
    vertices = sorted({x for i in query for x in edges[i][:2]})

    def visit(ids, vertices, background, accumulated, depth, eliminated):
        stats['nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        if len(ids) == 1:
            eid = ids[0]
            a, b, _ = edges[eid]
            ix = {v: j for j, v in enumerate(vertices)}
            conductance = background[ix[a]][ix[b]]
            if conductance <= 0:
                raise ArithmeticError('Nonpositive bypass in a bridgeless kernel')
            beta[eid] = 1 / conductance
            exponent[eid] = accumulated + 1
            stats['arithmetic_ops'] += 1
            assert eliminated == kernel.n - 2
            return
        middle = len(ids) // 2
        left, right = ids[:middle], ids[middle:]
        for child, added in ((left, right), (right, left)):
            c = [row[:] for row in background]
            size = len(vertices)
            stats['matrix_entries'] += size * size
            ix = {v: j for j, v in enumerate(vertices)}
            groups = {}
            for eid in added:
                a, b, _ = edges[eid]
                i, j = sorted((ix[a], ix[b]))
                groups.setdefault((i, j), []).append(weights[eid])
            assembly_depth = 0
            for (i, j), values in groups.items():
                assembly_depth = max(assembly_depth, height(len(values)))
                # Sum new original-edge conductances first, then add the
                # background once: its accumulated error pays only one rounding.
                c[i][j] = c[j][i] = c[i][j] + pair_sum(values)
                stats['arithmetic_ops'] += len(values)
            ledger = max(accumulated, chain_height + 1 + assembly_depth) + 1
            retained = {x for eid in child for x in edges[eid][:2]}
            active = list(range(size))
            path_eliminations = eliminated
            for pivot, label in enumerate(vertices):
                if label in retained:
                    continue
                neighbors = [i for i in active if i != pivot and c[pivot][i] > 0]
                # Every component of K-child touches a retained endpoint.
                if not neighbors:
                    raise ArithmeticError('Unexpected zero elimination pivot')
                total = pair_sum(c[pivot][i] for i in neighbors)
                stats['arithmetic_ops'] += len(neighbors) - 1
                for pos, j in enumerate(neighbors):
                    for i in neighbors[:pos]:
                        increment = c[i][pivot] * (c[j][pivot] / total)
                        if increment <= 0:
                            raise FloatingPointError('Underflow in a star-mesh update')
                        c[i][j] = c[j][i] = c[i][j] + increment
                        stats['pair_updates'] += 1
                        stats['arithmetic_ops'] += 3
                active.remove(pivot)
                ledger += 3 + height(len(neighbors))
                stats['eliminations'] += 1
                path_eliminations += 1
            child_vertices = [vertices[i] for i in active]
            child_background = [[c[i][j] for j in active] for i in active]
            visit(child, child_vertices, child_background, ledger,
                  depth + 1, path_eliminations)

    visit(query, vertices, [[0] * len(vertices) for _ in vertices],
          chain_height + 1, 0, 0)
    return beta, exponent


def all_outages_joint(n, edges, stats=None):
    """All endpoint outage resistances, O(n+m+sum r_i^3) arithmetic work."""
    bridges, kernels = compress(n, edges)
    m = len(edges)
    rho, delta, exponents = [inf] * m, [0] * m, [0] * m
    for kernel in kernels:
        h = max(height(len(chain)) for chain in kernel.chains)
        beta, beta_exponents = joint_kernel(kernel, h, stats)
        for j, (a, b, _) in enumerate(kernel.edges):
            exc = exclusive_sums([edges[e][2] for e in kernel.chains[j]])
            exponent = 1 + max(2 * h, beta_exponents[j]) if a != b else 2 * h + 1
            for eid, rest in zip(kernel.chains[j], exc):
                rho[eid] = rest + beta[j]
                delta[eid] = edges[eid][2] / (edges[eid][2] + rho[eid])
                exponents[eid] = exponent
                if not (rho[eid] > 0 and isfinite(rho[eid]) and delta[eid] > 0):
                    raise FloatingPointError('Result outside supported arithmetic range')
    return Profile(rho, delta, bridges, kernels, exponents)

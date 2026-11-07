"""Binary64 all-edge outage resistances using shared vertex partitions.

Each input edge is (u, v, resistance). Parallel edges retain their identities.
The full solver returns (Profile, diagnostics) and checks arithmetic range by
default. The compiled kernel interface takes a compressed, bridgeless kernel.
"""
import ctypes
from functools import lru_cache
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from outage import Profile, compress, exclusive_sums, height

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["ops", "subnormal", "positive_to_zero", "nonfinite", "hw_underflow", "nodes", "leaves"]
FLAGS = ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"]


@lru_cache(maxsize=1)
def _load_kernel():
    directory = tempfile.TemporaryDirectory(prefix="outage_kernels_")
    binary = Path(directory.name) / "kernels.so"
    source = ROOT / "code" / "positive_kernels.cpp"
    subprocess.run(["g++", *FLAGS, str(source), "-o", str(binary)], check=True)
    function = ctypes.CDLL(str(binary)).kernel
    indices = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    values = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    counters = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int] * 4 + [indices, indices, values, values, counters]
    function.restype = ctypes.c_int
    # The temporary directory lives for the lifetime of the loaded library.
    return function, directory


def inputs(edges):
    return (np.array([a for a, b, r in edges], dtype=np.int32),
            np.array([b for a, b, r in edges], dtype=np.int32),
            np.array([r for a, b, r in edges], dtype=np.float64))


def compiled(n, arrays, mode, audit=True):
    """Run edge partitions (2), vertex partitions (3), or separate SF-LDL (4)."""
    starts, ends, resistances = arrays
    size = len(resistances)
    if n <= 0 or len(starts) != size or len(ends) != size:
        raise ValueError("Kernel arrays must describe the same positive-size vertex set")
    values = np.zeros(size, dtype=np.float64)
    counters = np.zeros(len(METRICS), dtype=np.uint64)
    function, _ = _load_kernel()
    status = function(mode, int(audit), n, size, starts, ends, resistances, values, counters)
    metrics = dict(zip(METRICS, map(int, counters)))
    if status:
        raise ArithmeticError(f"Kernel {mode} could not complete: {metrics}")
    return values, metrics


def all_outages_vertex(n, edges, strict_range=True):
    """Return the outage profile and range counters; delta's exponent is rho's plus two."""
    data = [(a, b, float(r)) for a, b, r in edges]
    if any(not math.isfinite(r) or r <= 0 for a, b, r in data):
        raise ValueError("Resistances must be finite and positive")
    bridges, kernels = compress(n, data)
    rho = [math.inf] * len(data)
    delta = [0.0] * len(data)
    exponents = [0] * len(data)
    metrics = dict.fromkeys(METRICS, 0)
    tiny = sys.float_info.min
    metrics["outer_subnormal"] = sum(r < tiny for a, b, r in data)
    metrics["outer_zero"] = 0

    for ker in kernels:
        bypasses, counts = compiled(ker.n, inputs(ker.edges), mode=3)
        for name, value in counts.items():
            metrics[name] += value
        multiplicities = {}
        for a, b, _ in ker.edges:
            if a != b:
                pair = min(a, b), max(a, b)
                multiplicities[pair] = multiplicities.get(pair, 0) + 1
        parallel_depth = height(max(multiplicities.values(), default=1))
        chain_depth = max(height(len(chain)) for chain in ker.chains)
        levels = max(0, height(ker.n) - 1)
        elimination = max(0, ker.n - 2) * (3 + height(max(1, ker.n - 1)))
        bypass_exponent = chain_depth + 3 + max(2 * parallel_depth,
                                                  parallel_depth + levels + elimination)
        for edge, chain, bypass in zip(ker.edges, ker.chains, bypasses):
            a, b, _ = edge
            chain_exponent = 2 * height(len(chain))
            edge_exponent = 1 + (max(chain_exponent, bypass_exponent) if a != b else chain_exponent)
            resistances = [data[i][2] for i in chain]
            for i, other in zip(chain, exclusive_sums(resistances)):
                rho[i] = float(other) + float(bypass)
                denominator = data[i][2] + rho[i]
                if not math.isfinite(denominator):
                    raise ArithmeticError("Resistance overflow in chain reconstruction")
                delta[i] = data[i][2] / denominator
                metrics["outer_subnormal"] += sum(0 < x < tiny for x in (rho[i], denominator, delta[i]))
                metrics["outer_zero"] += int(delta[i] == 0 or rho[i] == 0)
                exponents[i] = edge_exponent

    range_events = ("hw_underflow", "subnormal", "positive_to_zero", "outer_subnormal",
                    "outer_zero", "nonfinite")
    if strict_range and any(metrics[key] for key in range_events):
        raise FloatingPointError(f"Outside the normal-range assumptions: {metrics}")
    return Profile(rho, delta, bridges, kernels, exponents), metrics


def vertex_paths(n, arrays):
    """Return bypasses and exact path diagnostics (sum q, pivot count, depth, max degree)."""
    _, temp = _load_kernel()
    fn = ctypes.CDLL(str(Path(temp.name) / "kernels.so")).kernel_paths
    ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ints, ints, doubles, doubles, ints]
    fn.restype = ctypes.c_int
    k = len(arrays[2])
    values = np.zeros(k)
    paths = np.zeros((k,4),dtype=np.int32)
    if fn(n,k,*arrays,values,paths):
        raise ArithmeticError("Path traversal failed")
    return values,paths

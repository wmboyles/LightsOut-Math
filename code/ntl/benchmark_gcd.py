import time
import random
import sys
from pathlib import Path

# Add 'code' directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from polynomials import GF2Polynomial
from ntl import ntl_gcd

def py_gcd(f: GF2Polynomial, g: GF2Polynomial) -> GF2Polynomial:
    left, right = f._value, g._value
    while right:
        left, right = right, GF2Polynomial._remainder_value(left, right)
    return GF2Polynomial.from_number(left)

def ntl_gcd_direct(f: GF2Polynomial, g: GF2Polynomial) -> GF2Polynomial:
    return GF2Polynomial.from_ntl(ntl_gcd(f.to_ntl(), g.to_ntl()))

def run_benchmark():
    degrees = [
        10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, 200000, 500000
    ]
    print(f"{'Degree':>8} | {'Py time (ms)':>12} | {'NTL time (ms)':>14} | {'Speedup':>10}")
    print("-" * 52)
    
    random.seed(42)
    for deg in degrees:
        v1 = random.getrandbits(deg) | (1 << deg) | 1
        v2 = random.getrandbits(deg) | (1 << deg) | 1
        f = GF2Polynomial.from_number(v1)
        g = GF2Polynomial.from_number(v2)
        
        # Verify correctness
        res_py = py_gcd(f, g)
        res_ntl = ntl_gcd_direct(f, g)
        assert res_py == res_ntl
        
        # Benchmark Python
        reps_py = max(1, min(30, int(15000 / (deg + 1))))
        t0 = time.perf_counter()
        for _ in range(reps_py):
            py_gcd(f, g)
        t_py = (time.perf_counter() - t0) / reps_py * 1000
        
        # Benchmark NTL
        reps_ntl = max(1, min(30, reps_py))
        t0 = time.perf_counter()
        for _ in range(reps_ntl):
            ntl_gcd_direct(f, g)
        t_ntl = (time.perf_counter() - t0) / reps_ntl * 1000
        
        speedup = t_py / t_ntl
        print(f"{deg:>8} | {t_py:>12.2f} | {t_ntl:>14.2f} | {speedup:>9.2f}x")

if __name__ == "__main__":
    run_benchmark()

"""Sharp verification of the prime-power target across many primes / prime powers.

For every prime power p^j (p odd prime, j>=2) up to BOUND, check
    gcd(C_{p^j}, C_{p^j}') == gcd(C_p, C_p').
Uses a memory-light windowed evaluation of C_n.
"""
import time
from cn_core import GF2Polynomial, S, ONE, ZERO, derivative, gcd

def c_window(n: int) -> GF2Polynomial:
    if n == 0: return ZERO
    if n <= 3: return [ZERO, ONE, ONE, S][n]
    a, b, c, d = ZERO, ONE, ONE, S
    for _ in range(n - 3):
        a, b, c, d = b, c, d, d + (S*c) + b + a
    return d

def mr(n):
    Cn = c_window(n)
    return gcd(Cn, derivative(Cn))

def primes_upto(m):
    sieve = [True]*(m+1); sieve[0]=sieve[1]=False
    for i in range(2,int(m**0.5)+1):
        if sieve[i]:
            for k in range(i*i,m+1,i): sieve[k]=False
    return [i for i in range(3,m+1) if sieve[i]]

BOUND = 60000
PRIME_MAX = 260
prime_base = {}
fails = []
tested = 0
t0 = time.time()
for p in primes_upto(PRIME_MAX):
    base = mr(p)
    prime_base[p] = base
    j = 2
    while p**j <= BOUND:
        g = mr(p**j)
        eq = (g == base)
        tested += 1
        if not eq:
            fails.append((p, j, g.degree, base.degree))
        j += 1

print(f"Tested {tested} prime powers p^j (j>=2, p<= {PRIME_MAX}, p^j <= {BOUND}) in {time.time()-t0:.1f}s")
print("Failures (target gcd not equal to level-1):", fails if fails else "NONE")

# Report the degrees of the multiple-root locus at each prime base (nonzero ones)
nz = [(p, prime_base[p].degree) for p in sorted(prime_base) if prime_base[p].degree>0]
print("\nPrimes with nonzero repeated-locus deg gcd(C_p,C_p')  (=d(p-1)/2):")
print(" ", nz)

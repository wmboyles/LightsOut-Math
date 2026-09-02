"""Verify every residue characteristic predicted by the p = 7, j = 2 resultant table.

For p^j = 49 the relative norms N^+(Lambda_{1,r}) over the fourteen classes of
nonzero ratios r factor into the primes 97, 197, 491, 881, 1373, 3527, 29009
and 139747 (the classes of r = 2, 3, 13, 14 give units).  A prime ell divides
one of those norms exactly when the five-term relation has a solution over
F_ell-bar with u or v of exact order 49, so this script checks the prediction
directly.  Run `python five_term_resultants.py 7 2` to regenerate the table.
"""

from five_term_relations import solutions
from plateau_scan import certificate, plateau_data, poly_str, verify_certificate

PRIMES = [97, 197, 491, 881, 1373, 3527, 29009, 139747]

if __name__ == "__main__":
    for ell in PRIMES:
        f1, c = plateau_data(ell, 7)
        sols = solutions(ell, 7, 2)
        prim = [
            (a, b)
            for a, b in sols
            if a % 7 and b % 7 and b % 49 not in (a % 49, (-a) % 49)
        ]
        print(
            f"ell={ell:7d}: ord_7(ell)={f1}, c=v_7(ell^ord-1)={c}, "
            f"|F|={ell**f1}, solutions={len(sols)}, nondeg. primitive={len(prim)}"
        )
        cert = certificate(ell, 7, 2)
        if cert:
            ok = verify_certificate(cert)
            mp = poly_str(cert["minpoly"])
            print(
                f"          certificate: F_{ell}[x]/({mp}), theta = x, "
                f"(a, b) = ({cert['a']}, {cert['b']}), verified = {ok}"
            )

# `C_n(S)` reciprocal-quartic power sums over GF(2)

Supporting code for `tex/five_term_plateau/five_term_plateau.tex`.
The suite lives under `code/ai_prototypes` because it is exploratory
verification code rather than part of the main package.

`C_n(S) \in F_2[S]` is defined by `C_n(X^2+X) = D_n(X) + D_n(X+1)` (Dickson
polynomials over F_2). Equivalently `C_n(S)` is the n-th Newton power sum of the
four roots of the reciprocal quartic `Q_S(T) = T^4 + T^3 + S T^2 + T + 1`, with
generating function `sum_n C_n(S) t^n = t(1+t)^2 / Q_S(t)`.

## Files
- `cn_core.py` — core module: builds `C_n` (recurrence / windowed), Dickson
  `D_n`, Fibonacci `F_n`, formal derivative, gcd, `x+1` translation, and the
  `x^2+x` composition. Reuses `code/polynomials.py::GF2Polynomial`.
- `verify_all.py` — proves/verifies every identity in the shifted-Dickson sections (definition,
  generating function, Frobenius `C_{2n}=C_n^2`, divisibility `m|n => C_m|C_n`,
  Dickson composition, the Hasse master GF, the Multiplicity-Two identity
  `C_n^{[2]}(Y)=Y^{-2}`, the bridge to Sutner nullity `d(n-1)=4 deg B_n`, and the
  primitive-quotient reformulation). Prints PASS/FAIL. Also computes
  `Sigma_a(Y)=1/Y^2` symbolically over `F_2(Y)`.
- `sharp_verify.py` — broad sweep of the prime-power target
  `gcd(C_{p^j},C_{p^j}') == gcd(C_p,C_p')` for all prime powers up to a bound.
- `integer_lift.py` — the monic integer lift `tilde C_n in Z[Y]` with
  `tilde C_n = C_n (mod 2)`, defined by
  `D_n(X)+D_n(X+1) = (2X+1) tilde C_n(X^2+X)` (odd n).
- `integer_verify.py` — verifies the lift, integer divisibility, the
  composition/telescoping identity
  `tilde C_{p^j} = prod_{i<j} Lambda_p[D_{p^i}(X), D_{p^i}(X+1)]`, and the
  discriminant/resultant oddness certificates for the target.
- `factorization_norms.py` — the real-cyclotomic factorization
  `tilde C_n(Y) = prod_k ((lambda_k+2)Y + lambda_k^2-3) = Res_lambda(R_n, lambda^2+Y*lambda+2Y-3)`
  (lambda_k = 2cos(2*pi*k/n)), the closed-form discriminant
  `disc(tilde C_n) = disc(R_n) * (collision norm)^2` with `disc(R_n)` ODD, and the
  collision <-> reciprocal-pair correspondence (2-adic localization).
- `relative_norm_identity.py` — derives and verifies the one-step relative
  norm descent
  `N_{K_j^+/K_{j-1}^+}(nu(lambda,mu)) =
   (mu+2)^p (lambda_p-D_p(M(mu)))`, with an independent packed scan of the
  resulting p-square criterion.
- `collision_parity.py` — parity of the collision factor
  `nu(l,m)=(l+2)(m+2)-1` via the involution `tau(x)=1/(x+2)-2` (which is inversion
  over F2). Checks the half-angle / circular-unit identities, special values
  `psi_n(-2)=±1`, resultants `Res(psi_{p^j}, R^tau)`, and the residue-degree
  obstruction: a mixed collision forces `ord_{p^j}(2) | 2 ord_{p^{j-1}}(2)`,
  impossible for `j > c_p`. Möbius conjugation of `M` to inversion identifies
  the palindromic lift of `psi^M` with a reciprocal; Apostol 2-powers
  `Res(Phi_{p^j}, Phi_{2^a p^j})` cannot divide it at mixed levels (degree),
  and cyclotomic irreducible degree `f_2 > phi(p)` already makes `M_2` odd
  for every non-Wieferich prime. On the plateau the remaining mechanism is
  inversion of the lower trace set inside `F_{2^{ord_p(2)}}`.

- `dynatomic_orbits.py` — arithmetic dynamics of `f=D_p`: the phi-linearization
  `D_n(u+1/u)=u^n+1/u^n` (phi(u)=u+1/u, deg phi=2), the critically-fixed
  identity `D_n'=F_n` (odd n), the covering-degree proof that `ord_{p^j}(2)`
  divides `2 ord_{p^{j-1}}(2)` for any mixed collision (an independent,
  shorter derivation of the residue-degree obstruction of
  `collision_parity.py`, via the degree-2 map phi rather than trace
  inversion), and the Frobenius-orbit <-> irreducible-factor correspondence
  for `B_n` (squaring a collision pair `{a,b}` doubles it to `{2a,2b}`, and
  the orbit size equals the degree of the associated irreducible factor of
  `B_p`; checked against independently-computed `deg B_p` for every prime
  `p<=257` with nontrivial repeated locus). Requires the `galois` package.

## Run

From this directory:

```
python verify_all.py
python sharp_verify.py
python integer_verify.py
python factorization_norms.py
python relative_norm_identity.py
python collision_parity.py
python dynatomic_orbits.py
```

On Windows, set `PYTHONUTF8=1` first for the Unicode output.

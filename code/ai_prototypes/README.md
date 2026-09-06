# AI prototypes

Exploratory and verification code for the mathematical writeups.
AI-generated, not verified by humans.
Code outside `ai_prototypes` must not depend on files inside it.
Prototypes may depend on shared production code.
Run scripts from the repository root, for example:

```powershell
$env:PYTHONUTF8 = '1'
python code\ai_prototypes\five_term_char2.py 1093 2
python code\ai_prototypes\cn_experiments\verify_all.py
```

## Five-term relation

- `five_term_char2.py`: optimized characteristic-two search and packed
  polynomial arithmetic.
- `five_term_field.py`: shared cyclotomic-field setup, power/trace tables,
  inverses, and Frobenius-orbit representatives.
- `five_term_relations.py`, `plateau_scan.py`: characteristic-independent
  searches and plateau certificates.
- `five_term_resultants.py`, `verify_p7_characteristics.py`: exact resultant
  tables and residue-characteristic certificates.
- `circular_module.py`, `sine_unit_obstruction.py`, `plateau_endpoint.py`:
  circular-unit, trace, endpoint, and phantom-system checks.
- `cn_experiments/`: shifted-Dickson, integer-lift, collision-norm, and
  arithmetic-dynamics verification suites.

## Other checks

The remaining scripts support the nullity bounds, multiprime stabilization,
explicit symplectic examples, and known Wieferich-prime computations.
The production NTL bridge, C++ helper, and CMake build live in `code\ntl`.
See `code\ntl\README.md` for build instructions before running NTL-backed checks.
The default executable is `code\ntl\build\bin\ntl_gf2x_gcd.exe`.
The build directory and Python bytecode are generated artifacts and are ignored.

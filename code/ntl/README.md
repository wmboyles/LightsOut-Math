# Native NTL helper

This directory contains the production NTL integration.
The executable is generated at `code\ntl\build\bin\ntl_gf2x_gcd.exe` on Windows.
The default build also copies its non-system runtime DLLs into the same directory.
The Python default path matches that location. Build products are ignored by Git.

## Prerequisites

Use a matching compiler and NTL installation; do not mix MSVC libraries with
MinGW-compiled code. You need:

- A C++11-capable compiler.
- CMake 3.20 or newer and a build tool such as Ninja.
- NTL development headers, libraries, and its installed `ntl.pc` metadata.
- `pkg-config` (or the compatible `pkgconf` implementation).
- Python 3.10 or newer for the bridge and smoke test.

### Windows: MSYS2 UCRT64

One way to obtain a compatible toolchain and prebuilt NTL is
[MSYS2](https://www.msys2.org/). Install MSYS2 and apply its prescribed system
updates first. Then run this in the **MSYS2 UCRT64 terminal**:

```text
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-ntl
```

The [NTL package](https://packages.msys2.org/package/mingw-w64-ucrt-x86_64-ntl)
includes `ntl.pc` and installs its library dependencies through the package
manager. This project builds only our helper, not NTL itself.

Return to **PowerShell**, with the repository root as the working directory.
For the default MSYS2 installation location:

```powershell
$msysRoot = "C:\msys64"
$env:PATH = "$msysRoot\ucrt64\bin;$env:PATH"
$env:PKG_CONFIG_SYSROOT_DIR = $msysRoot.Replace('\', '/')
cmake -S code\ntl -B code\ntl\build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build code\ntl\build --parallel 2
ctest --test-dir code\ntl\build --output-on-failure
```

Adjust `$msysRoot` if MSYS2 was installed elsewhere. Some MSYS2 package metadata
contains paths such as `/ucrt64/include`. `PKG_CONFIG_SYSROOT_DIR` translates
those paths for native Windows CMake; the forward-slash conversion is for
pkg-config's path syntax.

The UCRT64 `bin` directory and pkg-config setup are needed while building.
They are **not needed on Python's runtime PATH** after a successful build:
the `ntl_runtime` build target copies NTL, GCC, and their transitive non-system
DLL dependencies beside the helper. It runs on every default build, including
when the executable does not need relinking.

Keep the entire `build\bin` directory together rather than copying just the
executable. Windows system DLLs are not bundled. There is no requirement to use
an MSYS2 Python interpreter; the helper communicates with Python through a
subprocess. The Windows CTest smoke test deliberately removes the toolchain
from `PATH` to verify that the bundle is sufficient.

### Existing NTL installations

NTL's [installation instructions](https://libntl.org/doc/tour-unix.html)
describe building the library and installing its `ntl.pc` metadata.
Use the compiler that matches that installation and make the metadata discoverable
through `PKG_CONFIG_PATH`, for example in PowerShell:

```powershell
$env:PKG_CONFIG_PATH = "C:\Libraries\ntl\lib\pkgconfig;$env:PKG_CONFIG_PATH"
pkg-config --modversion ntl
pkg-config --cflags --libs ntl
```

If this installation is outside MSYS2, unset `PKG_CONFIG_SYSROOT_DIR` rather
than applying the MSYS2 root to its paths.

Then use the CMake commands above. The imported pkg-config target supplies the
include paths, compiler options, and link dependencies recorded by NTL.
This build expects a working shared-library installation; custom static builds
may require additional transitive link flags from NTL's `USER_MAKEFILE.txt`.
For DLLs installed outside the compiler and NTL prefixes, pass their directories
as a semicolon-separated `-DNTL_RUNTIME_DIRECTORIES=...` CMake argument.
If Python is not detected, pass `-DPython3_EXECUTABLE=C:\Path\To\python.exe` when
configuring. `-DBUILD_TESTING=OFF` disables the Python smoke-test requirement.

## Using the bridge

From the repository root, in an ordinary PowerShell session after building:

```powershell
$env:PYTHONPATH = (Resolve-Path code).Path
python -m ntl.smoke_test
python -c "from ntl import packed_ntl_gcd; print(packed_ntl_gcd(6, 10))"
```

The second command computes the GCD of `x^2+x` and `x^3+x`; its degree is `2`.
The result also contains native timing and configuration information.

For explicit polynomial conversion, use `GF2Polynomial.to_ntl()` and
`GF2Polynomial.from_ntl()`. The NTL representation on the Python side is packed
little-endian coefficient **bytes**, not a pointer to an object in the C++
subprocess:

```python
from polynomials import GF2Polynomial
from ntl import ntl_gcd

left = GF2Polynomial.from_number(0b10111)
right = GF2Polynomial.from_number(0b100011)
native_result = ntl_gcd(left.to_ntl(), right.to_ntl())
result = GF2Polynomial.from_ntl(native_result)
assert result == GF2Polynomial.from_number(0b1101)
```

`GF2Polynomial.gcd` performs this conversion automatically when the larger
operand's degree reaches `_NTL_GCD_DEGREE_THRESHOLD` (currently 100,000).
Smaller operands retain the Python Euclidean algorithm, and zero/equal operands
are handled without launching the helper. Missing executables and native failures
raise errors rather than silently switching backends.

To build somewhere other than `code\ntl\build`, pass that directory to CMake's
`-B` option. CTest uses the actual target path automatically. Python callers must
then supply the executable explicitly via `ntl_gcd(..., executable=Path(...))`,
`packed_ntl_gcd(..., executable=Path(...))`,
or the prototype scripts' `--ntl-executable` option.

## Protocol

Standard input contains two operands. Each is encoded as an unsigned 64-bit
little-endian byte count followed by that many little-endian coefficient bytes:
bit `i` is the coefficient of `x^i`. Standard output contains JSON; diagnostics
go to standard error. The optional first executable argument is the thread count.
Actual threading support depends on how NTL was built.

The response contains `gcd_bytes_hex`, the hexadecimal encoding of the same
little-endian coefficient bytes, alongside the existing degree and statistics.
For example, `"0201"` represents bytes `b"\x02\x01"`, or the polynomial `x^8+x`.
It is a byte sequence, not the usual most-significant-digit-first integer hex.
The zero polynomial has an empty byte sequence and native degree `-1`.

`ntl_gcd` returns the decoded coefficient bytes; the packed-integer compatibility
API `packed_ntl_gcd` continues returning only degree and statistics. The bridge
checks that the returned coefficients are canonical and match the reported degree.
An older helper without the coefficient field must be rebuilt.

## Troubleshooting

- **CMake, Ninja, or pkg-config not found:** check the toolchain's `bin` directory
  is on this shell's `PATH`.
- **Package `ntl` not found:** check `pkg-config --modversion ntl` and
  `PKG_CONFIG_PATH`; an NTL runtime DLL alone is not a development installation.
- **CMake reports a nonexistent `/ucrt64/include` path:** set
  `PKG_CONFIG_SYSROOT_DIR` as above and configure again with a fresh cache.
  CMake 3.24 or newer supports `cmake --fresh` with the same configuration
  arguments; otherwise use a new build directory.
- **Missing `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libntl-45.dll`, or exit
  status `3221225781` (`0xC0000135`):** rerun the default
  `cmake --build code\ntl\build` command to stage runtime dependencies. Keep
  those DLLs beside the executable. Do not download individual DLLs from
  third-party DLL sites.
- **Runtime dependency staging fails:** check the NTL installation and
  `NTL_RUNTIME_DIRECTORIES`. Missing dependencies fail the build explicitly.
- **Executable still exits before producing JSON:** check matching compiler
  architecture. Do not silently fall back to a different GCD.
- **Changed toolchains:** configure a fresh build directory rather than reusing
  CMake's cached compiler selection.

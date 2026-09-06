"""Exchange packed GF2X coefficient bytes with the native NTL helper."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys

DEFAULT_NTL_EXECUTABLE = Path(__file__).resolve().parent.joinpath(
    "build",
    "bin",
    "ntl_gf2x_gcd.exe" if sys.platform == "win32" else "ntl_gf2x_gcd",
)


def _run_ntl_gcd(
    left: bytes,
    right: bytes,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> tuple[bytes, dict[str, int | float | bool]]:
    """Return the native GCD's coefficient bytes and validated statistics."""

    if not isinstance(left, bytes) or not isinstance(right, bytes):
        raise TypeError("NTL polynomials must be packed coefficient bytes")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not executable.is_file():
        raise FileNotFoundError(f"NTL helper not found: {executable}")

    payload = b"".join((
        struct.pack("<Q", len(left)),
        left,
        struct.pack("<Q", len(right)),
        right,
    ))
    try:
        completed = subprocess.run(
            [executable, str(threads)],
            input=payload,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        if (error.returncode & 0xFFFFFFFF) == 0xC0000135:
            raise RuntimeError(
                "NTL helper could not load a required runtime DLL. "
                "Run 'cmake --build code\\ntl\\build' and keep the generated "
                "DLLs beside the executable."
            ) from error
        raise
    response = json.loads(completed.stdout)
    if not isinstance(response, dict):
        raise ValueError("NTL helper returned a non-object JSON response")

    encoded = response.pop("gcd_bytes_hex", None)
    if not isinstance(encoded, str):
        raise ValueError("NTL helper did not return coefficients; rebuild the executable")
    try:
        polynomial = bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError("NTL helper returned invalid coefficient bytes") from error
    if polynomial and polynomial[-1] == 0:
        raise ValueError("NTL helper returned noncanonical coefficient bytes")

    degree = (
        (len(polynomial) - 1) * 8 + polynomial[-1].bit_length() - 1
        if polynomial else -1
    )
    if type(response.get("degree")) is not int or response["degree"] != degree:
        raise ValueError("NTL helper returned inconsistent degree and coefficients")

    statistics: dict[str, int | float | bool] = {}
    for key, value in response.items():
        if not isinstance(value, (int, float, bool)):
            raise ValueError(f"NTL helper returned an invalid statistic: {key}")
        statistics[key] = value
    return polynomial, statistics


def ntl_gcd(
    left: bytes,
    right: bytes,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> bytes:
    """Return a GCD in NTL's little-endian GF2X coefficient-byte format.

    These bytes represent a serialized polynomial, not a pointer to a C++ object.
    """

    polynomial, _ = _run_ntl_gcd(left, right, executable, threads)
    return polynomial


def packed_ntl_gcd(
    left: int,
    right: int,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> dict[str, int | float | bool]:
    """Return GCD degree and statistics for the packed-integer prototype API."""

    if left < 0 or right < 0:
        raise ValueError("Packed polynomials must be non-negative")
    _, statistics = _run_ntl_gcd(
        left.to_bytes((left.bit_length() + 7) // 8, "little"),
        right.to_bytes((right.bit_length() + 7) // 8, "little"),
        executable,
        threads,
    )
    return statistics
"""Production bridge to the native NTL polynomial helper."""

from .backend import DEFAULT_NTL_EXECUTABLE, ntl_gcd, packed_ntl_gcd

__all__ = ["DEFAULT_NTL_EXECUTABLE", "ntl_gcd", "packed_ntl_gcd"]

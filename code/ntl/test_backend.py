import json
from pathlib import Path
import struct
import subprocess
import unittest
from unittest.mock import patch

from . import backend


def response_packet(response: object) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        ["ntl_gf2x_gcd"], 0,
        stdout=json.dumps(response).encode("ascii"),
        stderr=b"",
    )


class NTLBackendTests(unittest.TestCase):
    def test_byte_protocol_and_native_polynomial_result(self):
        response = {"degree": 3, "gcd_bytes_hex": "0d", "gcd_seconds": 0.25}
        executable = Path("custom-ntl.exe")
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(
                backend.subprocess, "run", return_value=response_packet(response)
            ) as run:
                result = backend.ntl_gcd(b"\x00\x0d", b"\x0d", executable, 3)
        self.assertEqual(result, b"\x0d")
        run.assert_called_once_with(
            [executable, "3"],
            input=struct.pack("<Q", 2) + b"\x00\x0d"
            + struct.pack("<Q", 1) + b"\x0d",
            capture_output=True,
            check=True,
        )

    def test_legacy_integer_api_preserves_statistics(self):
        response = {
            "degree": 3,
            "gcd_bytes_hex": "0d",
            "gcd_seconds": 0.25,
            "external_gf2x": True,
        }
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(
                backend.subprocess, "run", return_value=response_packet(response)
            ):
                result = backend.packed_ntl_gcd(0x0d00, 0x0d)
        self.assertEqual(result, {
            "degree": 3,
            "gcd_seconds": 0.25,
            "external_gf2x": True,
        })

    def test_zero_result_uses_empty_bytes_and_degree_minus_one(self):
        response = {"degree": -1, "gcd_bytes_hex": ""}
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(
                backend.subprocess, "run", return_value=response_packet(response)
            ):
                self.assertEqual(backend.ntl_gcd(b"", b""), b"")

    def test_malformed_native_results_are_rejected(self):
        responses = (
            [],
            {"degree": 3},
            {"degree": 3, "gcd_bytes_hex": "not hex"},
            {"degree": 3, "gcd_bytes_hex": "0d00"},
            {"degree": 2, "gcd_bytes_hex": "0d"},
            {"degree": True, "gcd_bytes_hex": "03"},
            {"degree": 3, "gcd_bytes_hex": "0d", "threads": "one"},
        )
        for response in responses:
            with self.subTest(response=response):
                with patch.object(Path, "is_file", return_value=True):
                    with patch.object(
                        backend.subprocess, "run",
                        return_value=response_packet(response),
                    ):
                        with self.assertRaises(ValueError):
                            backend.ntl_gcd(b"\x00\x0d", b"\x0d")

    def test_missing_helper_is_not_silently_ignored(self):
        with patch.object(Path, "is_file", return_value=False):
            with patch.object(backend.subprocess, "run") as run:
                with self.assertRaises(FileNotFoundError):
                    backend.ntl_gcd(b"\x01", b"\x03")
                run.assert_not_called()

    def test_native_process_failure_propagates(self):
        error = subprocess.CalledProcessError(
            1, ["ntl_gf2x_gcd"], stderr=b"native failure"
        )
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(backend.subprocess, "run", side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    backend.ntl_gcd(b"\x01", b"\x03")

    def test_missing_runtime_dll_has_actionable_error(self):
        for returncode in (0xC0000135, 0xC0000135 - (1 << 32)):
            with self.subTest(returncode=returncode):
                error = subprocess.CalledProcessError(returncode, ["ntl_gf2x_gcd"])
                with patch.object(Path, "is_file", return_value=True):
                    with patch.object(backend.subprocess, "run", side_effect=error):
                        with self.assertRaisesRegex(
                            RuntimeError, "required runtime DLL"
                        ) as raised:
                            backend.ntl_gcd(b"\x01", b"\x03")
                        self.assertIs(raised.exception.__cause__, error)

    def test_invalid_arguments_are_rejected_before_launch(self):
        with patch.object(backend.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                backend.ntl_gcd(b"", b"", threads=0)
            with self.assertRaises(ValueError):
                backend.packed_ntl_gcd(-1, 1)
            self.assertRaises(TypeError, backend.ntl_gcd, 1, b"")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

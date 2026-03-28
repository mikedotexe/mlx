# Copyright © 2026 Apple Inc.

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import _safetensors_repack_helper as repack_helper
import repack_safetensors_for_mmap as repack_cli


def _write_aligned_fixture(path: Path) -> None:
    payloads = OrderedDict(
        (
            ("dense.weight", struct.pack("<6f", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
            ("dense.bias", struct.pack("<8h", 3, -1, 7, 2, -5, 8, 13, -21)),
        )
    )
    header = OrderedDict(
        (
            ("__metadata__", OrderedDict((("fixture", "aligned"), ("kind", "test")))),
        )
    )
    offset = 0
    for name, payload in payloads.items():
        header[name] = OrderedDict(
            (
                ("dtype", "F32" if name.endswith("weight") else "I16"),
                ("shape", [2, 3] if name.endswith("weight") else [8]),
                ("data_offsets", [offset, offset + len(payload)]),
            )
        )
        offset += len(payload)

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    with path.open("wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        for payload in payloads.values():
            handle.write(payload)


def _write_misaligned_fixture(path: Path) -> None:
    header = b'{"x":{"dtype":"I16","shape":[2],"data_offsets":[1,5]}}'
    payload = b"\x00" + np.array([123, -456], dtype=np.int16).tobytes()
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(payload)


def _write_small_u8_fixture(path: Path) -> None:
    header = b'{"tiny":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(b"\x2A")


def _write_nonmonotonic_header_fixture(path: Path) -> None:
    payload_a = struct.pack("<2h", 11, -13)
    payload_b = struct.pack("<2h", 21, -34)
    header = OrderedDict(
        (
            (
                "later",
                OrderedDict(
                    (
                        ("dtype", "I16"),
                        ("shape", [2]),
                        ("data_offsets", [len(payload_a), len(payload_a) + len(payload_b)]),
                    )
                ),
            ),
            (
                "earlier",
                OrderedDict(
                    (
                        ("dtype", "I16"),
                        ("shape", [2]),
                        ("data_offsets", [0, len(payload_a)]),
                    )
                ),
            ),
        )
    )
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    with path.open("wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(payload_a)
        handle.write(payload_b)


def _tensor_payload_bytes(path: Path) -> dict[str, bytes]:
    info = repack_helper.inspect_safetensors_file(path)
    raw = path.read_bytes()
    payloads = {}
    for tensor in info["tensors"]:
        begin = info["payload_start"] + tensor.relative_begin
        end = info["payload_start"] + tensor.relative_end
        payloads[tensor.name] = raw[begin:end]
    return payloads


class TestRepackSafetensorsForMmap(unittest.TestCase):
    def test_repack_preserves_tensor_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "aligned.safetensors"
            dst = Path(tmp_dir) / "aligned.repacked.safetensors"
            _write_aligned_fixture(src)

            summary = repack_helper.repack_safetensors_file(src, output_path=dst)
            original_info = repack_helper.inspect_safetensors_file(src)
            repacked_info = repack_helper.inspect_safetensors_file(dst)

            self.assertEqual(original_info["metadata"], repacked_info["metadata"])
            self.assertEqual(original_info["tensor_names"], repacked_info["tensor_names"])
            self.assertEqual(_tensor_payload_bytes(src), _tensor_payload_bytes(dst))
            self.assertEqual(summary["tensor_count"], 2)

            original_loaded, original_metadata = mx.load(
                str(src), return_metadata=True, memory_map=False
            )
            repacked_loaded, repacked_metadata = mx.load(
                str(dst), return_metadata=True, memory_map=False
            )
            self.assertEqual(original_metadata, repacked_metadata)
            self.assertEqual(original_loaded.keys(), repacked_loaded.keys())
            for key in original_loaded.keys():
                self.assertTrue(mx.array_equal(original_loaded[key], repacked_loaded[key]))

    def test_repack_default_output_path_and_force(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "fixture.safetensors"
            _write_aligned_fixture(src)

            summary = repack_helper.repack_safetensors_file(src)
            default_output = src.with_name("fixture.aligned.safetensors")

            self.assertTrue(default_output.exists())
            self.assertEqual(summary["output_file"], str(default_output))

            with self.assertRaises(FileExistsError):
                repack_helper.repack_safetensors_file(src)

            forced_summary = repack_helper.repack_safetensors_file(src, force=True)
            self.assertEqual(forced_summary["output_file"], str(default_output))

    def test_repack_cli_json_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "fixture.safetensors"
            dst = Path(tmp_dir) / "fixture.aligned.safetensors"
            _write_aligned_fixture(src)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = repack_cli.main(
                    [str(src), "--output", str(dst), "--format", "json"]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["input_file"], str(src))
            self.assertEqual(payload["output_file"], str(dst))
            self.assertIn("alignment_before", payload)
            self.assertIn("alignment_after", payload)
            self.assertIn("expected_to_eliminate_misaligned_offset", payload)

    def test_fixed_point_layout_converges_when_offset_digits_grow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "tiny.safetensors"
            dst = Path(tmp_dir) / "tiny.aligned.safetensors"
            _write_small_u8_fixture(src)

            summary = repack_helper.repack_safetensors_file(
                src,
                output_path=dst,
                base_alignment=100_000,
            )
            info = repack_helper.inspect_safetensors_file(dst)
            first_tensor = info["tensors"][0]
            absolute_begin = info["payload_start"] + first_tensor.relative_begin

            self.assertGreater(summary["layout_iterations"], 1)
            self.assertEqual(absolute_begin % 100_000, 0)

    def test_repack_accepts_nonmonotonic_header_order_when_ranges_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "unordered.safetensors"
            dst = Path(tmp_dir) / "unordered.aligned.safetensors"
            _write_nonmonotonic_header_fixture(src)

            source_info = repack_helper.inspect_safetensors_file(src)
            summary = repack_helper.repack_safetensors_file(src, output_path=dst)
            repacked_info = repack_helper.inspect_safetensors_file(dst)

            self.assertFalse(source_info["header_offsets_monotonic"])
            self.assertTrue(repacked_info["header_offsets_monotonic"])
            self.assertEqual(summary["tensor_count"], 2)
            loaded = mx.load(str(dst), memory_map=False)
            self.assertTrue(mx.array_equal(loaded["earlier"], mx.array([11, -13], dtype=mx.int16)))
            self.assertTrue(mx.array_equal(loaded["later"], mx.array([21, -34], dtype=mx.int16)))

    @unittest.skipUnless(
        hasattr(mx, "last_mmap_load_stats"), "requires rebuilt mlx.core"
    )
    def test_repacked_misaligned_fixture_maps_without_misaligned_offset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "hostile.safetensors"
            dst = Path(tmp_dir) / "hostile.aligned.safetensors"
            _write_misaligned_fixture(src)

            mx.last_mmap_load_stats(clear=True)
            original_loaded = mx.load(str(src), memory_map=True)
            original_stats = mx.last_mmap_load_stats(clear=True)

            self.assertTrue(
                mx.array_equal(original_loaded["x"], mx.array([123, -456], dtype=mx.int16))
            )
            self.assertEqual(
                (original_stats.get("fallback_reasons") or {}).get("misaligned_offset"),
                1,
            )

            summary = repack_helper.repack_safetensors_file(src, output_path=dst)

            mx.last_mmap_load_stats(clear=True)
            repacked_loaded = mx.load(str(dst), memory_map=True)
            repacked_stats = mx.last_mmap_load_stats(clear=True)

            if (
                repacked_stats.get("mapped_bytes", 0) == 0
                and (repacked_stats.get("fallback_reasons") or {}).get(
                    "make_buffer_failed", 0
                )
                > 0
            ):
                self.skipTest("backend cannot materialize direct mapped safetensors views")

            self.assertTrue(summary["expected_to_eliminate_misaligned_offset"])
            self.assertTrue(
                mx.array_equal(repacked_loaded["x"], mx.array([123, -456], dtype=mx.int16))
            )
            self.assertGreater(repacked_stats.get("mapped_bytes", 0), 0)
            self.assertEqual(
                (repacked_stats.get("fallback_reasons") or {}).get(
                    "misaligned_offset", 0
                ),
                0,
            )
            self.assertEqual(repacked_stats.get("fallback_tensors"), 0)


if __name__ == "__main__":
    unittest.main()

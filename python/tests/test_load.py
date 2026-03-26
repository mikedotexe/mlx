# Copyright © 2023 Apple Inc.

import os
import platform
import struct
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx_tests
import numpy as np


class TestLoad(mlx_tests.MLXTestCase):
    dtypes = [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float16",
        "complex64",
    ]

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir)
        cls._direct_mmap_view_supported = False
        cls._direct_mmap_view_probe_reason = None
        if hasattr(mx, "last_mmap_load_stats"):
            probe_file = os.path.join(cls.test_dir, "test_last_mmap_probe.safetensors")
            probe_arr = mx.arange(8, dtype=mx.float32).reshape(2, 4)
            mx.save_safetensors(probe_file, {"weights": probe_arr})
            mx.last_mmap_load_stats(clear=True)
            mx.load(probe_file, memory_map=True)
            stats = mx.last_mmap_load_stats(clear=True) or {}
            cls._direct_mmap_view_supported = bool(
                int(stats.get("mapped_bytes") or 0) > 0
            )
            fallback_reasons = stats.get("fallback_reasons") or {}
            if fallback_reasons:
                cls._direct_mmap_view_probe_reason = next(iter(fallback_reasons))

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def _write_quantized_q4_0_gguf_fixture(self, path):
        tensor_name = b"quant.weight"
        header = bytearray()
        header.extend(b"GGUF")
        header.extend(struct.pack("<I", 3))
        header.extend(struct.pack("<Q", 1))  # tensor_count
        header.extend(struct.pack("<Q", 0))  # metadata_kv_count
        header.extend(struct.pack("<Q", len(tensor_name)))
        header.extend(tensor_name)
        header.extend(struct.pack("<I", 1))  # ndim
        header.extend(struct.pack("<Q", 32))  # dim0
        header.extend(struct.pack("<I", 2))  # GGUF_TYPE_Q4_0
        header.extend(struct.pack("<Q", 0))  # tensor data offset
        aligned_header = ((len(header) + 31) // 32) * 32
        header.extend(b"\x00" * (aligned_header - len(header)))
        block = struct.pack("<e", 0.5) + bytes(range(16))
        with open(path, "wb") as f:
            f.write(header)
            f.write(block)

    def test_save_and_load(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(self.test_dir, f"mlx_{dt}_{i}.npy")
                        save_file_npy = os.path.join(self.test_dir, f"npy_{dt}_{i}.npy")

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_mlx = mx.array(save_arr_npy)

                        mx.save(save_file_mlx, save_arr_mlx)
                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by mlx as mlx array
                        load_arr_mlx_mlx = mx.load(save_file_mlx)
                        self.assertTrue(mx.array_equal(load_arr_mlx_mlx, save_arr_mlx))

                        # Load array saved by numpy as mlx array
                        load_arr_npy_mlx = mx.load(save_file_npy)
                        self.assertTrue(mx.array_equal(load_arr_npy_mlx, save_arr_mlx))

                        # Load array saved by mlx as numpy array
                        load_arr_mlx_npy = np.load(save_file_mlx)
                        self.assertTrue(np.array_equal(load_arr_mlx_npy, save_arr_npy))

        save_file = os.path.join(self.test_dir, f"mlx_path.npy")
        save_arr = mx.ones((32,))
        mx.save(Path(save_file), save_arr)

        # Load array saved by mlx as mlx array
        load_arr = mx.load(Path(save_file))
        self.assertTrue(mx.array_equal(load_arr, save_arr))

    def test_load_npy_dtype(self):
        save_file = os.path.join(self.test_dir, "mlx_path.npy")
        a = np.random.randn(8).astype(np.float64)
        np.save(save_file, a)
        out = mx.load(save_file, stream=mx.cpu)
        self.assertEqual(out.dtype, mx.float64)
        self.assertTrue(np.array_equal(np.array(out), a))

        a = np.random.randn(8).astype(np.float64)
        b = np.random.randn(8).astype(np.float64)
        c = a + 0j * b
        np.save(save_file, c)
        with self.assertRaises(Exception):
            out = mx.load(save_file, stream=mx.cpu)

    def test_save_and_load_safetensors(self):
        test_file = os.path.join(self.test_dir, "test.safetensors")
        with self.assertRaises(Exception):
            mx.save_safetensors(test_file, {"a": mx.ones((4, 4))}, {"testing": 0})

        for obj in [str, Path]:
            mx.save_safetensors(
                obj(test_file),
                {"test": mx.ones((2, 2))},
                {"testing": "test", "format": "mlx"},
            )
            res = mx.load(obj(test_file), return_metadata=True)
            self.assertEqual(len(res), 2)
            self.assertEqual(res[1], {"testing": "test", "format": "mlx"})

        for dt in self.dtypes + ["bfloat16"]:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.safetensors"
                        )
                        save_dict = {
                            "test": (
                                mx.random.normal(shape=shape, dtype=getattr(mx, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else mx.ones(shape, dtype=getattr(mx, dt))
                            )
                        }

                        with open(save_file_mlx, "wb") as f:
                            mx.save_safetensors(f, save_dict)
                        with open(save_file_mlx, "rb") as f:
                            load_dict = mx.load(f)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            mx.array_equal(load_dict["test"], save_dict["test"])
                        )

    def test_load_safetensors_memory_map(self):
        test_file = os.path.join(self.test_dir, "test_memory_map.safetensors")
        save_dict = {
            "f32": mx.arange(12, dtype=mx.float32).reshape(3, 4),
            "i16": mx.arange(8, dtype=mx.int16),
        }
        metadata = {"source": "mapped-test"}
        mx.save_safetensors(test_file, save_dict, metadata)

        mapped, mapped_metadata = mx.load(
            test_file, return_metadata=True, memory_map=True
        )
        default, default_metadata = mx.load(test_file, return_metadata=True)

        self.assertEqual(mapped_metadata, metadata)
        self.assertEqual(default_metadata, metadata)
        self.assertEqual(mapped.keys(), default.keys())
        for key in default.keys():
            self.assertTrue(mx.array_equal(mapped[key], default[key]))

    def test_load_safetensors_memory_map_file_object_fallback(self):
        test_file = os.path.join(self.test_dir, "test_memory_map_fileobj.safetensors")
        expected = mx.arange(6, dtype=mx.int16).reshape(2, 3)
        mx.save_safetensors(test_file, {"x": expected})

        with open(test_file, "rb") as f:
            loaded = mx.load(f, format="safetensors", memory_map=True)
        self.assertTrue(mx.array_equal(loaded["x"], expected))

    def test_load_safetensors_memory_map_misaligned_offsets_fallback(self):
        test_file = os.path.join(
            self.test_dir, "test_memory_map_bad_offsets.safetensors"
        )
        header = b'{"x":{"dtype":"I16","shape":[2],"data_offsets":[1,5]}}'
        payload = b"\x00" + np.array([123, -456], dtype=np.int16).tobytes()
        with open(test_file, "wb") as f:
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            f.write(payload)

        out = mx.load(test_file, memory_map=True)["x"]
        self.assertTrue(mx.array_equal(out, mx.array([123, -456], dtype=mx.int16)))

    @unittest.skipUnless(
        hasattr(mx, "last_mmap_load_stats"), "requires rebuilt mlx.core"
    )
    def test_last_mmap_load_stats_safetensors_misaligned_offset_reason(self):
        if not self._direct_mmap_view_supported:
            self.skipTest(
                "current backend cannot materialize mapped safetensors views "
                f"({self._direct_mmap_view_probe_reason or 'unknown reason'})"
            )
        test_file = os.path.join(
            self.test_dir, "test_last_mmap_stats_bad_offsets.safetensors"
        )
        header = b'{"x":{"dtype":"I16","shape":[2],"data_offsets":[1,5]}}'
        payload = b"\x00" + np.array([123, -456], dtype=np.int16).tobytes()
        with open(test_file, "wb") as f:
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            f.write(payload)

        mx.last_mmap_load_stats(clear=True)
        loaded = mx.load(test_file, memory_map=True)
        stats = mx.last_mmap_load_stats(clear=True)

        self.assertTrue(mx.array_equal(loaded["x"], mx.array([123, -456], dtype=mx.int16)))
        self.assertIsNotNone(stats)
        self.assertEqual(stats["fallback_reasons"].get("misaligned_offset"), 1)

    @unittest.skipUnless(
        hasattr(mx, "last_mmap_load_stats"), "requires rebuilt mlx.core"
    )
    def test_last_mmap_load_stats_safetensors_memory_map(self):
        test_file = os.path.join(self.test_dir, "test_last_mmap_stats.safetensors")
        expected = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        mx.save_safetensors(test_file, {"weights": expected})

        self.assertIsNone(mx.last_mmap_load_stats(clear=True))

        loaded = mx.load(test_file, memory_map=True)
        stats = mx.last_mmap_load_stats(clear=True)

        self.assertTrue(mx.array_equal(loaded["weights"], expected))
        self.assertIsNotNone(stats)
        if self._direct_mmap_view_supported:
            self.assertGreater(stats["mapped_bytes"], 0)
        else:
            self.assertEqual(stats["mapped_bytes"], 0)
            self.assertEqual(stats["fallback_reasons"].get("make_buffer_failed"), 1)
        self.assertGreaterEqual(stats["copied_bytes"], 0)
        self.assertIn("fallback_tensors", stats)
        self.assertIn("fallback_reasons", stats)
        self.assertIn("fallback_reason_bytes", stats)
        self.assertIn("fallback_reason_source_bytes", stats)
        self.assertIn("hotset_promoted_tensors", stats)
        self.assertIn("hotset_promotion_strategy", stats)
        self.assertIsNone(mx.last_mmap_load_stats(clear=True))

    @unittest.skipUnless(
        hasattr(mx, "last_mmap_load_stats"), "requires rebuilt mlx.core"
    )
    def test_last_mmap_load_stats_cleared_by_non_mmap_load(self):
        safetensors_file = os.path.join(
            self.test_dir, "test_last_mmap_stats_clear.safetensors"
        )
        npy_file = os.path.join(self.test_dir, "test_last_mmap_stats_clear.npy")
        mx.save_safetensors(
            safetensors_file,
            {"weights": mx.arange(8, dtype=mx.float16).reshape(2, 4)},
        )
        mx.save(npy_file, mx.arange(6, dtype=mx.float32))

        mx.last_mmap_load_stats(clear=True)
        mx.load(safetensors_file, memory_map=True)
        self.assertIsNotNone(mx.last_mmap_load_stats(clear=False))

        mx.load(npy_file)
        self.assertIsNone(mx.last_mmap_load_stats(clear=True))

    @unittest.skipUnless(
        hasattr(mx, "last_load_phase_stats"), "requires rebuilt mlx.core"
    )
    def test_last_load_phase_stats_safetensors_memory_map(self):
        test_file = os.path.join(self.test_dir, "test_last_phase_stats.safetensors")
        expected = mx.arange(16, dtype=mx.float32).reshape(4, 4)
        mx.save_safetensors(test_file, {"weights": expected})

        self.assertIsNone(mx.last_load_phase_stats(clear=True))

        loaded = mx.load(test_file, memory_map=True)
        stats = mx.last_load_phase_stats(clear=True)

        self.assertTrue(mx.array_equal(loaded["weights"], expected))
        self.assertIsNotNone(stats)
        self.assertEqual(stats["tag"], "safetensors")
        self.assertTrue(stats["memory_map"])
        self.assertIn("open_map_seconds", stats)
        self.assertIn("parse_seconds", stats)
        self.assertIn("tensor_setup_seconds", stats)
        self.assertGreaterEqual(stats["open_map_seconds"], 0.0)
        self.assertGreaterEqual(stats["parse_seconds"], 0.0)
        self.assertGreaterEqual(stats["tensor_setup_seconds"], 0.0)
        self.assertIn("parse_minor_faults", stats)
        self.assertIn("tensor_setup_major_faults", stats)
        self.assertIsNone(mx.last_load_phase_stats(clear=True))

    @unittest.skipUnless(
        hasattr(mx, "last_load_phase_stats"), "requires rebuilt mlx.core"
    )
    def test_last_load_phase_stats_cleared_by_non_mmap_load(self):
        safetensors_file = os.path.join(
            self.test_dir, "test_last_phase_stats_clear.safetensors"
        )
        npy_file = os.path.join(self.test_dir, "test_last_phase_stats_clear.npy")
        mx.save_safetensors(
            safetensors_file,
            {"weights": mx.arange(4, dtype=mx.float16).reshape(2, 2)},
        )
        mx.save(npy_file, mx.arange(6, dtype=mx.float32))

        mx.last_load_phase_stats(clear=True)
        mx.load(safetensors_file, memory_map=True)
        self.assertIsNotNone(mx.last_load_phase_stats(clear=False))

        mx.load(npy_file)
        stats = mx.last_load_phase_stats(clear=True)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["tag"], "npy")
        self.assertFalse(stats["memory_map"])
        self.assertIsNone(mx.last_load_phase_stats(clear=True))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        # TODO: Add support for other dtypes (self.dtypes + ["bfloat16"])
        supported_dtypes = ["float16", "float32", "int8", "int16", "int32"]
        for dt in supported_dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.gguf"
                        )
                        save_dict = {
                            "test": (
                                mx.random.normal(shape=shape, dtype=getattr(mx, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else mx.ones(shape, dtype=getattr(mx, dt))
                            )
                        }

                        mx.save_gguf(save_file_mlx, save_dict)
                        load_dict = mx.load(save_file_mlx)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            mx.array_equal(load_dict["test"], save_dict["test"])
                        )

        save_file_mlx = os.path.join(self.test_dir, f"mlx_path_test_fs.gguf")
        save_dict = {"test": mx.ones(shape)}
        mx.save_gguf(Path(save_file_mlx), save_dict)
        load_dict = mx.load(Path(save_file_mlx))
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_load_gguf_memory_map(self):
        test_file = os.path.join(self.test_dir, "test_memory_map.gguf")
        save_dict = {
            "f32": mx.arange(24, dtype=mx.float32).reshape(4, 6),
            "f16": mx.arange(10, dtype=mx.float16),
            "i8": mx.arange(12, dtype=mx.int8),
        }
        metadata = {"meta": "mapped-test"}
        mx.save_gguf(test_file, save_dict, metadata)

        mapped, mapped_metadata = mx.load(
            test_file, return_metadata=True, memory_map=True
        )
        default, default_metadata = mx.load(test_file, return_metadata=True)

        self.assertEqual(mapped_metadata["meta"], "mapped-test")
        self.assertEqual(default_metadata["meta"], "mapped-test")
        self.assertEqual(mapped.keys(), default.keys())
        for key in default.keys():
            self.assertTrue(mx.array_equal(mapped[key], default[key]))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_load_quantized_gguf_memory_map_parity(self):
        test_file = os.path.join(self.test_dir, "test_memory_map_quantized_q4_0.gguf")
        self._write_quantized_q4_0_gguf_fixture(test_file)

        mapped = mx.load(test_file, format="gguf", memory_map=True)
        default = mx.load(test_file, format="gguf")

        self.assertEqual(set(mapped.keys()), set(default.keys()))
        self.assertEqual(
            set(mapped.keys()),
            {"quant.weight", "quant.scales", "quant.biases"},
        )
        for key in default.keys():
            self.assertTrue(mx.array_equal(mapped[key], default[key]))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    @unittest.skipUnless(
        hasattr(mx, "last_mmap_load_stats"), "requires rebuilt mlx.core"
    )
    def test_last_mmap_load_stats_quantized_gguf_reason(self):
        test_file = os.path.join(self.test_dir, "test_last_mmap_stats_quantized.gguf")
        self._write_quantized_q4_0_gguf_fixture(test_file)

        mx.last_mmap_load_stats(clear=True)
        loaded = mx.load(test_file, format="gguf", memory_map=True)
        stats = mx.last_mmap_load_stats(clear=True)

        self.assertTrue("quant.weight" in loaded)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["fallback_reasons"].get("quantized_conversion"), 1)

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_load_gguf_nvfp4_compat_flag(self):
        test_file = os.path.join(self.test_dir, "test_nvfp4_compat.gguf")

        data = bytearray()
        data.extend(b"GGUF")
        data.extend(struct.pack("<I", 3))  # version
        data.extend(struct.pack("<Q", 2))  # metadata count
        data.extend(struct.pack("<Q", 0))  # secondary header field
        data.extend(b"format")
        data.extend(struct.pack("<Q", 5))
        data.extend(b"nvfp4")
        data.extend(struct.pack("<Q", 7))
        data.extend(b"version")
        data.extend(struct.pack("<Q", 3))
        data.extend(b"1.0")
        data.extend(struct.pack("<Q", 2))  # tensor count

        header_end = 139
        data.extend(struct.pack("<Q", 3))
        data.extend(b"foo")
        data.extend(struct.pack("<I", 1))
        data.extend(struct.pack("<Q", 4))
        data.extend(struct.pack("<Q", header_end))

        data.extend(struct.pack("<Q", 3))
        data.extend(b"bar")
        data.extend(struct.pack("<I", 1))
        data.extend(struct.pack("<Q", 4))
        data.extend(struct.pack("<Q", header_end + 4))

        data.extend(bytes([1, 2, 3, 4, 10, 11, 12, 13]))
        with open(test_file, "wb") as f:
            f.write(data)

        with self.assertRaises(RuntimeError):
            mx.load(test_file, format="gguf")

        loaded, metadata = mx.load(
            test_file,
            format="gguf",
            return_metadata=True,
            gguf_nvfp4_compat=True,
        )
        self.assertEqual(metadata["format"], "nvfp4")
        self.assertEqual(metadata["version"], "1.0")
        self.assertTrue(
            mx.array_equal(loaded["foo"], mx.array([1, 2, 3, 4], dtype=mx.uint8))
        )
        self.assertTrue(
            mx.array_equal(loaded["bar"], mx.array([10, 11, 12, 13], dtype=mx.uint8))
        )

        mapped = mx.load(
            test_file, format="gguf", memory_map=True, gguf_nvfp4_compat=True
        )
        self.assertTrue(mx.array_equal(mapped["foo"], loaded["foo"]))
        self.assertTrue(mx.array_equal(mapped["bar"], loaded["bar"]))

    def test_load_f8_e4m3(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        expected = [
            0,
            448,
            -448,
            -0.875,
            0.4375,
            -0.005859,
            -1.25,
            -1.25,
            -1.5,
            -0.0039,
        ]
        expected = mx.array(expected, dtype=mx.bfloat16)
        contents = b'H\x00\x00\x00\x00\x00\x00\x00{"tensor":{"dtype":"F8_E4M3","shape":[10],"data_offsets":[0,10]}}       \x00~\xfe\xb6.\x83\xba\xba\xbc\x82'
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(contents)
            f.seek(0)
            out = mx.load(f)["tensor"]
        self.assertTrue(mx.allclose(mx.from_fp8(out), expected))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_basic(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}
        metadata = {}

        # Empty works
        mx.save_gguf(save_file_mlx, save_dict, metadata)

        # Loads without the metadata
        load_dict = mx.load(save_file_mlx)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))

        # Loads empty metadata
        load_dict, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 0)

        # Loads string metadata
        metadata = {"meta": "data"}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        load_dict, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 1)
        self.assertTrue("meta" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta"], "data")

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_arrays(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}

        # Test scalars and one dimensional arrays
        for t in [
            mx.uint8,
            mx.int8,
            mx.uint16,
            mx.int16,
            mx.uint32,
            mx.int32,
            mx.uint64,
            mx.int64,
            mx.float32,
        ]:
            for shape in [(), (2,)]:
                arr = mx.random.uniform(shape=shape).astype(t)
                metadata = {"meta": arr}
                mx.save_gguf(save_file_mlx, save_dict, metadata)
                _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
                self.assertEqual(len(meta_load_dict), 1)
                self.assertTrue("meta" in meta_load_dict)
                self.assertTrue(mx.array_equal(meta_load_dict["meta"], arr))
                self.assertEqual(meta_load_dict["meta"].dtype, arr.dtype)

        for t in [mx.float16, mx.bfloat16, mx.complex64]:
            with self.assertRaises(ValueError):
                arr = mx.array(1, t)
                metadata = {"meta": arr}
                mx.save_gguf(save_file_mlx, save_dict, metadata)

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_mixed(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}

        # Test string and array
        arr = mx.array(1.5)
        metadata = {"meta1": arr, "meta2": "data"}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 2)
        self.assertTrue("meta1" in meta_load_dict)
        self.assertTrue(mx.array_equal(meta_load_dict["meta1"], arr))
        self.assertEqual(meta_load_dict["meta1"].dtype, arr.dtype)
        self.assertTrue("meta2" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta2"], "data")

        # Test list of strings
        metadata = {"meta": ["data1", "data2", "data345"]}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 1)
        self.assertEqual(meta_load_dict["meta"], metadata["meta"])

        # Test a combination of stuff
        metadata = {
            "meta1": ["data1", "data2", "data345"],
            "meta2": mx.array([1, 2, 3, 4]),
            "meta3": "data",
            "meta4": mx.array(1.5),
        }
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 4)
        for k, v in metadata.items():
            if isinstance(v, mx.array):
                self.assertTrue(mx.array_equal(meta_load_dict[k], v))
            else:
                self.assertEqual(meta_load_dict[k], v)

    def test_save_and_load_fs(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.npy"
                        )
                        save_file_npy = os.path.join(
                            self.test_dir, f"npy_{dt}_{i}_fs.npy"
                        )

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_mlx = mx.array(save_arr_npy)

                        with open(save_file_mlx, "wb") as f:
                            mx.save(f, save_arr_mlx)

                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by mlx as mlx array
                        with open(save_file_mlx, "rb") as f:
                            load_arr_mlx_mlx = mx.load(f)
                        self.assertTrue(mx.array_equal(load_arr_mlx_mlx, save_arr_mlx))

                        # Load array saved by numpy as mlx array
                        with open(save_file_npy, "rb") as f:
                            load_arr_npy_mlx = mx.load(f)
                        self.assertTrue(mx.array_equal(load_arr_npy_mlx, save_arr_mlx))

                        # Load array saved by mlx as numpy array
                        load_arr_mlx_npy = np.load(save_file_mlx)
                        self.assertTrue(np.array_equal(load_arr_mlx_npy, save_arr_npy))

    def test_savez_and_loadz(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                shapes = [(6,), (6, 6), (4, 1, 3, 1, 2)]
                save_file_mlx_uncomp = os.path.join(
                    self.test_dir, f"mlx_{dt}_uncomp.npz"
                )
                save_file_npy_uncomp = os.path.join(
                    self.test_dir, f"npy_{dt}_uncomp.npz"
                )
                save_file_mlx_comp = os.path.join(self.test_dir, f"mlx_{dt}_comp.npz")
                save_file_npy_comp = os.path.join(self.test_dir, f"npy_{dt}_comp.npz")

                # Make dictionary of multiple
                save_arrs_npy = {
                    f"save_arr_{i}": np.random.uniform(
                        0.0, 32.0, size=shapes[i]
                    ).astype(getattr(np, dt))
                    for i in range(len(shapes))
                }
                save_arrs_mlx = {k: mx.array(v) for k, v in save_arrs_npy.items()}

                # Save as npz files
                np.savez(save_file_npy_uncomp, **save_arrs_npy)
                mx.savez(save_file_mlx_uncomp, **save_arrs_mlx)
                np.savez_compressed(save_file_npy_comp, **save_arrs_npy)
                mx.savez_compressed(save_file_mlx_comp, **save_arrs_mlx)

                for save_file_npy, save_file_mlx in (
                    (save_file_npy_uncomp, save_file_mlx_uncomp),
                    (save_file_npy_comp, save_file_mlx_comp),
                ):
                    # Load array saved by mlx as mlx array
                    load_arr_mlx_mlx = mx.load(save_file_mlx)
                    for k, v in load_arr_mlx_mlx.items():
                        self.assertTrue(mx.array_equal(save_arrs_mlx[k], v))

                    # Load arrays saved by numpy as mlx arrays
                    load_arr_npy_mlx = mx.load(save_file_npy)
                    for k, v in load_arr_npy_mlx.items():
                        self.assertTrue(mx.array_equal(save_arrs_mlx[k], v))

                    # Load array saved by mlx as numpy array
                    load_arr_mlx_npy = np.load(save_file_mlx)
                    for k, v in load_arr_mlx_npy.items():
                        self.assertTrue(np.array_equal(save_arrs_npy[k], v))

    def test_non_contiguous(self):
        a = mx.broadcast_to(mx.array([1, 2]), [4, 2])

        save_file = os.path.join(self.test_dir, "a.npy")
        mx.save(save_file, a)
        aload = mx.load(save_file)
        self.assertTrue(mx.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.safetensors")
        mx.save_safetensors(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        if platform.system() == "Windows":
            return

        save_file = os.path.join(self.test_dir, "a.gguf")
        mx.save_gguf(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        # safetensors and gguf only work with row contiguous
        # make sure col contiguous is handled properly
        save_file = os.path.join(self.test_dir, "a.safetensors")
        a = mx.arange(4).reshape(2, 2).T
        mx.save_safetensors(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.gguf")
        mx.save_gguf(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

    def test_load_donation(self):
        x = mx.random.normal((1024,))
        mx.eval(x)
        save_file = os.path.join(self.test_dir, "donation.npy")
        mx.save(save_file, x)
        mx.synchronize()

        mx.reset_peak_memory()
        scale = mx.array(2.0)
        y = mx.load(save_file)
        mx.eval(y)
        mx.synchronize()
        load_only = mx.get_peak_memory()
        y = mx.load(save_file) * scale
        mx.eval(y)
        mx.synchronize()
        load_with_binary = mx.get_peak_memory()

        self.assertEqual(load_only, load_with_binary)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()

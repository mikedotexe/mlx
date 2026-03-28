#!/usr/bin/env python3

# Copyright © 2026 Apple Inc.

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FIXED_POINT_ITERATIONS = 32
COPY_CHUNK_BYTES = 1024 * 1024
DIAGNOSTIC_SAMPLE_LIMIT = 6
SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "C64": 8,
}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    itemsize: int
    relative_begin: int
    relative_end: int
    nbytes: int


def default_base_alignment() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 4096


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _ordered_json_loads(raw: bytes) -> OrderedDict:
    return json.loads(raw, object_pairs_hook=OrderedDict)


def _compact_json_dumps(value: OrderedDict) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_shape(name: str, shape_value) -> tuple[int, ...]:
    if not isinstance(shape_value, list):
        raise ValueError(f"{name}: shape must be a list")
    shape: list[int] = []
    for dim in shape_value:
        if not isinstance(dim, int):
            raise ValueError(f"{name}: shape entries must be integers")
        if dim < 0:
            raise ValueError(f"{name}: negative shape dimension {dim} is not allowed")
        shape.append(int(dim))
    return tuple(shape)


def _checked_nbytes(name: str, shape: tuple[int, ...], itemsize: int) -> int:
    nelem = 1
    for dim in shape:
        nelem *= dim
    nbytes = nelem * itemsize
    if nbytes <= 0:
        raise ValueError(f"{name}: empty tensors are not supported in safetensors")
    return nbytes


def _validate_offsets(name: str, offsets_value) -> tuple[int, int]:
    if not isinstance(offsets_value, list) or len(offsets_value) != 2:
        raise ValueError(f"{name}: data_offsets must be a [begin, end] list")
    begin, end = offsets_value
    if not isinstance(begin, int) or not isinstance(end, int):
        raise ValueError(f"{name}: data_offsets must be integers")
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(f"{name}: invalid data_offsets [{begin}, {end}]")
    return int(begin), int(end)


def inspect_safetensors_file(path: str | Path) -> dict:
    file_path = Path(path)
    raw = file_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{file_path}: file is too small to be safetensors")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if header_len <= 0 or header_len > len(raw) - 8:
        raise ValueError(f"{file_path}: invalid safetensors header length {header_len}")
    header = _ordered_json_loads(raw[8 : 8 + header_len])
    if not isinstance(header, OrderedDict):
        raise ValueError(f"{file_path}: expected object header")

    payload_start = 8 + header_len
    tensors: list[TensorEntry] = []
    tensor_names: list[str] = []
    header_offsets_monotonic = True
    previous_begin = -1
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not key:
            raise ValueError(f"{file_path}: tensor key must be a non-empty string")
        if not isinstance(value, dict):
            raise ValueError(f"{file_path}: tensor entry {key} must be an object")
        dtype = value.get("dtype")
        if dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{file_path}: tensor {key} has unsupported dtype {dtype!r}")
        shape = _validate_shape(key, value.get("shape"))
        begin, end = _validate_offsets(key, value.get("data_offsets"))
        if begin < previous_begin:
            header_offsets_monotonic = False
        itemsize = SAFETENSORS_DTYPES[dtype]
        nbytes = _checked_nbytes(key, shape, itemsize)
        if end - begin != nbytes:
            raise ValueError(
                f"{file_path}: tensor {key} byte span {end - begin} does not match expected {nbytes}"
            )
        absolute_begin = payload_start + begin
        absolute_end = payload_start + end
        if absolute_end > len(raw):
            raise ValueError(f"{file_path}: tensor {key} payload range is out of bounds")
        tensors.append(
            TensorEntry(
                name=key,
                dtype=dtype,
                shape=shape,
                itemsize=itemsize,
                relative_begin=begin,
                relative_end=end,
                nbytes=nbytes,
            )
        )
        tensor_names.append(key)
        previous_begin = begin

    sorted_tensors = sorted(tensors, key=lambda tensor: tensor.relative_begin)
    previous_end = 0
    for tensor in sorted_tensors:
        if tensor.relative_begin < previous_end:
            raise ValueError(
                f"{file_path}: tensor {tensor.name} payload overlaps a previous tensor range"
            )
        previous_end = tensor.relative_end

    return {
        "path": str(file_path),
        "file_size": len(raw),
        "header_len": int(header_len),
        "payload_start": payload_start,
        "header": header,
        "tensors": tensors,
        "tensor_names": tensor_names,
        "metadata": header.get("__metadata__", OrderedDict()),
        "header_offsets_monotonic": header_offsets_monotonic,
    }


def _build_header_with_offsets(
    info: dict, relative_offsets: list[tuple[int, int]]
) -> OrderedDict:
    header = OrderedDict()
    tensor_index_by_name = {
        name: index for index, name in enumerate(info["tensor_names"])
    }
    for key, value in info["header"].items():
        if key == "__metadata__":
            header[key] = value
            continue
        tensor_index = tensor_index_by_name[key]
        begin, end = relative_offsets[tensor_index]
        header[key] = OrderedDict(
            (
                ("dtype", value["dtype"]),
                ("shape", list(value["shape"])),
                ("data_offsets", [begin, end]),
            )
        )
    return header


def compute_repacked_layout(
    info: dict,
    *,
    base_alignment: int | None = None,
    max_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
) -> dict:
    alignment = int(base_alignment or default_base_alignment())
    if alignment <= 0:
        raise ValueError("base_alignment must be positive")

    current_offsets = [
        (tensor.relative_begin, tensor.relative_end) for tensor in info["tensors"]
    ]
    header_bytes = b""
    payload_start = 0
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        header = _build_header_with_offsets(info, current_offsets)
        header_bytes = _compact_json_dumps(header)
        payload_start = 8 + len(header_bytes)

        next_offsets: list[tuple[int, int]] = []
        cursor = align_up(payload_start, alignment) - payload_start
        for tensor in info["tensors"]:
            absolute_cursor = payload_start + cursor
            absolute_cursor = align_up(absolute_cursor, tensor.itemsize)
            cursor = absolute_cursor - payload_start
            begin = cursor
            end = begin + tensor.nbytes
            next_offsets.append((begin, end))
            cursor = end

        iterations = iteration
        if next_offsets == current_offsets:
            return {
                "header": header,
                "header_bytes": header_bytes,
                "header_len": len(header_bytes),
                "payload_start": payload_start,
                "relative_offsets": next_offsets,
                "base_alignment": alignment,
                "iterations": iterations,
                "output_size": payload_start + next_offsets[-1][1],
            }
        current_offsets = next_offsets

    raise RuntimeError(
        f"{info['path']}: failed to converge repacked safetensors layout "
        f"after {max_iterations} iterations"
    )


def _alignment_diagnostics(
    *,
    payload_start: int,
    tensors: list[TensorEntry],
    relative_offsets: list[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    diagnostics: OrderedDict[str, dict] = OrderedDict()
    offsets = relative_offsets or [
        (tensor.relative_begin, tensor.relative_end) for tensor in tensors
    ]
    for tensor, (begin, _) in zip(tensors, offsets):
        absolute_begin = payload_start + begin
        mod = absolute_begin % tensor.itemsize
        bucket = diagnostics.setdefault(
            tensor.dtype,
            {
                "itemsize": tensor.itemsize,
                "tensor_count": 0,
                "misaligned_tensors": 0,
                "sample_absolute_mods": [],
            },
        )
        bucket["tensor_count"] += 1
        if mod != 0:
            bucket["misaligned_tensors"] += 1
        if len(bucket["sample_absolute_mods"]) < DIAGNOSTIC_SAMPLE_LIMIT and mod not in bucket["sample_absolute_mods"]:
            bucket["sample_absolute_mods"].append(mod)
    for bucket in diagnostics.values():
        bucket["sample_absolute_mods"].sort()
        bucket["aligned"] = bucket["misaligned_tensors"] == 0
    return diagnostics


def _expected_to_eliminate_misaligned_offsets(alignment_after: dict[str, dict]) -> bool:
    return all(bucket["aligned"] for bucket in alignment_after.values())


def summarize_repacked_pair(
    input_path: str | Path,
    output_path: str | Path,
    *,
    base_alignment: int | None = None,
    layout: dict | None = None,
) -> dict:
    input_info = inspect_safetensors_file(input_path)
    output_info = inspect_safetensors_file(output_path)
    computed_layout = layout or compute_repacked_layout(
        input_info, base_alignment=base_alignment
    )
    if computed_layout["relative_offsets"] != [
        (tensor.relative_begin, tensor.relative_end) for tensor in output_info["tensors"]
    ]:
        raise ValueError(
            f"{output_path}: repacked file does not match the expected alignment-friendly layout"
        )

    padding_added = output_info["file_size"] - input_info["file_size"]
    alignment_before = _alignment_diagnostics(
        payload_start=input_info["payload_start"], tensors=input_info["tensors"]
    )
    alignment_after = _alignment_diagnostics(
        payload_start=output_info["payload_start"], tensors=output_info["tensors"]
    )
    return {
        "input_file": str(Path(input_path)),
        "output_file": str(Path(output_path)),
        "input_size": input_info["file_size"],
        "output_size": output_info["file_size"],
        "tensor_count": len(input_info["tensors"]),
        "total_padding_bytes_added": padding_added,
        "base_alignment": computed_layout["base_alignment"],
        "layout_iterations": computed_layout["iterations"],
        "payload_start_before": input_info["payload_start"],
        "payload_start_after": output_info["payload_start"],
        "header_offsets_monotonic_before": input_info["header_offsets_monotonic"],
        "header_offsets_monotonic_after": output_info["header_offsets_monotonic"],
        "alignment_before": alignment_before,
        "alignment_after": alignment_after,
        "expected_to_eliminate_misaligned_offset": _expected_to_eliminate_misaligned_offsets(
            alignment_after
        ),
    }


def repack_safetensors_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    base_alignment: int | None = None,
    force: bool = False,
) -> dict:
    source = Path(input_path)
    if output_path is None:
        output = source.with_name(f"{source.stem}.aligned.safetensors")
    else:
        output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    if source.resolve() == output.resolve():
        raise ValueError("output path must be different from the input path")

    info = inspect_safetensors_file(source)
    layout = compute_repacked_layout(info, base_alignment=base_alignment)

    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, output.open("wb") as dst:
        dst.write(struct.pack("<Q", layout["header_len"]))
        dst.write(layout["header_bytes"])
        cursor = 0
        payload_start = info["payload_start"]
        for tensor, (begin, end) in zip(info["tensors"], layout["relative_offsets"]):
            if begin < cursor:
                raise RuntimeError("repacked offsets must be monotonic")
            pad = begin - cursor
            if pad:
                dst.write(b"\x00" * pad)
            src.seek(payload_start + tensor.relative_begin)
            remaining = tensor.nbytes
            while remaining:
                chunk = src.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise RuntimeError(
                        f"{source}: unexpected EOF while copying tensor {tensor.name}"
                    )
                dst.write(chunk)
                remaining -= len(chunk)
            cursor = end

    return summarize_repacked_pair(
        source, output, base_alignment=layout["base_alignment"], layout=layout
    )


def format_repack_summary_text(summary: dict) -> str:
    before = summary["alignment_before"]
    after = summary["alignment_after"]
    lines = [
        "Safetensors Repack for mmap",
        f"input={summary['input_file']}",
        f"output={summary['output_file']}",
        (
            f"size input={summary['input_size']} output={summary['output_size']} "
            f"padding_added={summary['total_padding_bytes_added']}"
        ),
        (
            f"base_alignment={summary['base_alignment']} "
            f"layout_iterations={summary['layout_iterations']}"
        ),
        (
            f"payload_start before={summary['payload_start_before']} "
            f"after={summary['payload_start_after']}"
        ),
        (
            f"header_offsets_monotonic before={summary['header_offsets_monotonic_before']} "
            f"after={summary['header_offsets_monotonic_after']}"
        ),
        f"expected_to_eliminate_misaligned_offset={summary['expected_to_eliminate_misaligned_offset']}",
        "alignment before:",
    ]
    for dtype, bucket in before.items():
        lines.append(
            f"  {dtype}: misaligned={bucket['misaligned_tensors']}/{bucket['tensor_count']} "
            f"mods={bucket['sample_absolute_mods']}"
        )
    lines.append("alignment after:")
    for dtype, bucket in after.items():
        lines.append(
            f"  {dtype}: misaligned={bucket['misaligned_tensors']}/{bucket['tensor_count']} "
            f"mods={bucket['sample_absolute_mods']}"
        )
    return "\n".join(lines)

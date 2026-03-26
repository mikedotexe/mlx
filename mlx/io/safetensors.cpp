// Copyright © 2023 Apple Inc.
//
#include <json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stack>
#include <unordered_set>
#include <vector>

#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/io/mmap.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

namespace {

constexpr uint64_t kMaxJsonHeaderLength = 100000000;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void assign_fault_delta(
    const io::LoadFaultCounts& before,
    const io::LoadFaultCounts& after,
    size_t* minor_out,
    size_t* major_out) {
  *minor_out = after.minor_faults - before.minor_faults;
  *major_out = after.major_faults - before.major_faults;
}

std::optional<size_t> checked_tensor_nbytes(const Shape& shape, Dtype dtype) {
  size_t nelem = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      return std::nullopt;
    }
    if (dim != 0 &&
        nelem > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return std::nullopt;
    }
    nelem *= static_cast<size_t>(dim);
  }

  if (dtype.size() != 0 &&
      nelem > std::numeric_limits<size_t>::max() / dtype.size()) {
    return std::nullopt;
  }
  return nelem * dtype.size();
}

array copy_tensor_from_bytes(
    const char* src,
    Shape shape,
    Dtype dtype,
    size_t nbytes) {
  auto buffer = allocator::malloc(nbytes);
  if (nbytes > 0) {
    std::memcpy(buffer.raw_ptr(), src, nbytes);
  }
  return array(buffer, std::move(shape), dtype);
}

bool should_copy_small_tensor(const LoadOptions& options, size_t nbytes) {
  return options.memory_map &&
      options.mmap_small_tensor_copy_max_bytes.has_value() &&
      nbytes <= options.mmap_small_tensor_copy_max_bytes.value();
}

bool hotset_promotion_enabled(const LoadOptions& options) {
  return options.memory_map &&
      options.mmap_hotset_promotion_top_k.has_value() &&
      options.mmap_hotset_promotion_top_k.value() > 0;
}

} // namespace

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else {
    throw std::runtime_error(
        "[safetensor] unsupported dtype " + std::string(str));
  }
}

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  return load_safetensors(in_stream, s, LoadOptions{});
}

/** Load array from reader in safetensor format with options */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s,
    const LoadOptions& options) {
  io::clear_last_mmap_load_stats();
  io::clear_last_load_phase_stats();
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensors] Failed to open " + in_stream->label());
  }

  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);
  io::LoadPhaseStats phase_stats;
  phase_stats.tag = "safetensors";
  phase_stats.memory_map = false;
  auto parse_faults_before = io::current_load_fault_counts();
  auto parse_start = Clock::now();

  uint64_t jsonHeaderLength = 0;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream->read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json metadata " + in_stream->label());
  }
  phase_stats.parse_seconds = elapsed_seconds(parse_start);
  assign_fault_delta(
      parse_faults_before,
      io::current_load_fault_counts(),
      &phase_stats.parse_minor_faults,
      &phase_stats.parse_major_faults);
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  auto setup_faults_before = io::current_load_fault_counts();
  auto setup_start = Clock::now();
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    res.insert(
        {item.key(),
         array(
             shape,
             type,
             std::make_shared<Load>(
                 stream, in_stream, offset + data_offsets.at(0), false),
                 std::vector<array>{})});
  }
  phase_stats.tensor_setup_seconds = elapsed_seconds(setup_start);
  assign_fault_delta(
      setup_faults_before,
      io::current_load_fault_counts(),
      &phase_stats.tensor_setup_minor_faults,
      &phase_stats.tensor_setup_major_faults);
  io::set_last_load_phase_stats(std::move(phase_stats));
  return {res, metadata_map};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(file, s, LoadOptions{});
}

SafetensorsLoad load_safetensors(
    const std::string& file,
    StreamOrDevice s,
    const LoadOptions& options) {
  io::clear_last_mmap_load_stats();
  io::clear_last_load_phase_stats();
  if (!options.memory_map) {
    return load_safetensors(
        std::make_shared<io::ParallelFileReader>(file), s, options);
  }

  io::LoadPhaseStats phase_stats;
  phase_stats.tag = "safetensors";
  phase_stats.memory_map = true;
  auto open_map_faults_before = io::current_load_fault_counts();
  auto open_map_start = Clock::now();
  auto mapped = io::map_file_readonly(file, options.mmap_prefetch_strategy);
  auto mapped_size = mapped->size();
  auto base = io::make_mapped_base_array(
      mapped->data(), mapped_size, [mapped](allocator::Buffer) mutable {
        mapped.reset();
      });
  phase_stats.open_map_seconds = elapsed_seconds(open_map_start);
  assign_fault_delta(
      open_map_faults_before,
      io::current_load_fault_counts(),
      &phase_stats.open_map_minor_faults,
      &phase_stats.open_map_major_faults);

  const auto* data = static_cast<const char*>(mapped->data());
  auto parse_faults_before = io::current_load_fault_counts();
  auto parse_start = Clock::now();
  if (mapped_size < 8) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length file " + file);
  }

  uint64_t jsonHeaderLength = 0;
  std::memcpy(&jsonHeaderLength, data, sizeof(jsonHeaderLength));
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength ||
      jsonHeaderLength > mapped_size - 8) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json header length file " + file);
  }

  auto metadata = json::parse(data + 8, data + 8 + jsonHeaderLength);
  if (!metadata.is_object()) {
    throw std::runtime_error(
        "[load_safetensors] Invalid json metadata file " + file);
  }
  phase_stats.parse_seconds = elapsed_seconds(parse_start);
  assign_fault_delta(
      parse_faults_before,
      io::current_load_fault_counts(),
      &phase_stats.parse_minor_faults,
      &phase_stats.parse_major_faults);

  const size_t payload_offset = static_cast<size_t>(jsonHeaderLength) + 8;
  io::MmapLoadStats stats;
  const bool mapped_views_enabled = base.has_value();

  auto setup_faults_before = io::current_load_fault_counts();
  auto setup_start = Clock::now();
  std::unordered_map<std::string, array> res;
  std::unordered_map<std::string, std::string> metadata_map;
  std::vector<io::HotsetPromotionCandidate> promotion_candidates;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        metadata_map.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    if (!mapped_views_enabled || !hotset_promotion_enabled(options)) {
      continue;
    }

    const std::string& dtype = item.value().at("dtype");
    const Shape shape = item.value().at("shape");
    const std::vector<size_t> data_offsets = item.value().at("data_offsets");
    const Dtype type = dtype_from_safetensor_str(dtype);
    if (data_offsets.size() != 2 || data_offsets[1] < data_offsets[0]) {
      continue;
    }

    const auto expected_nbytes = checked_tensor_nbytes(shape, type);
    if (!expected_nbytes.has_value()) {
      continue;
    }
    if (data_offsets[1] - data_offsets[0] != expected_nbytes.value()) {
      continue;
    }
    if (data_offsets[0] > mapped_size - payload_offset) {
      continue;
    }

    size_t tensor_offset = payload_offset;
    if (data_offsets[0] > std::numeric_limits<size_t>::max() - payload_offset) {
      continue;
    }
    tensor_offset += data_offsets[0];
    if (tensor_offset > mapped_size ||
        expected_nbytes.value() > mapped_size - tensor_offset) {
      continue;
    }
    if (should_copy_small_tensor(options, expected_nbytes.value())) {
      continue;
    }
    promotion_candidates.push_back({item.key(), expected_nbytes.value()});
  }
  const auto promoted_tensors = io::select_hotset_promotion_candidates(
      std::move(promotion_candidates),
      options.mmap_hotset_promotion_top_k.value_or(0),
      options.mmap_hotset_promotion_min_bytes.value_or(0));
  for (const auto& [name, reason] : promoted_tensors.reasons) {
    stats.record_hotset_selection(name, reason, promoted_tensors.strategy);
  }

  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      continue;
    }

    const std::string& dtype = item.value().at("dtype");
    const Shape shape = item.value().at("shape");
    const std::vector<size_t> data_offsets = item.value().at("data_offsets");
    const Dtype type = dtype_from_safetensor_str(dtype);

    if (data_offsets.size() != 2 || data_offsets[1] < data_offsets[0]) {
      throw std::runtime_error(
          "[load_safetensors] Invalid tensor offsets in file " + file);
    }

    std::string fallback_reason;
    const auto expected_nbytes = checked_tensor_nbytes(shape, type);
    if (!expected_nbytes.has_value()) {
      fallback_reason = "size_overflow";
    } else if (data_offsets[1] - data_offsets[0] != expected_nbytes.value()) {
      fallback_reason = "size_mismatch";
    } else if (data_offsets[0] > mapped_size - payload_offset) {
      fallback_reason = "out_of_bounds";
    }

    size_t tensor_offset = payload_offset;
    if (data_offsets[0] > std::numeric_limits<size_t>::max() - payload_offset) {
      fallback_reason = "offset_overflow";
    } else {
      tensor_offset += data_offsets[0];
    }
    if (mapped_views_enabled && fallback_reason.empty() &&
        should_copy_small_tensor(options, expected_nbytes.value())) {
      fallback_reason = "small_tensor_threshold";
    } else if (
        mapped_views_enabled && fallback_reason.empty() &&
        promoted_tensors.names.find(item.key()) != promoted_tensors.names.end()) {
      fallback_reason = "hotset_promotion";
    }
    if (mapped_views_enabled && fallback_reason.empty()) {
      std::string view_failure;
      auto view = io::make_mapped_view(
          *base, tensor_offset, shape, type, &view_failure);
      if (view.has_value()) {
        stats.record_mapped(expected_nbytes.value());
        res.emplace(item.key(), std::move(*view));
        continue;
      }
      fallback_reason =
          view_failure.empty() ? "view_creation_failed" : view_failure;
    }

    if (fallback_reason.empty()) {
      fallback_reason = "make_buffer_failed";
    }

    size_t copied_bytes =
        expected_nbytes.has_value() ? expected_nbytes.value() : 0;
    stats.record_fallback(fallback_reason, copied_bytes, copied_bytes);
    if (tensor_offset > mapped_size ||
        copied_bytes > mapped_size - tensor_offset) {
      throw std::runtime_error(
          "[load_safetensors] Tensor payload out of bounds in file " + file);
    }

    res.emplace(
        item.key(),
        copy_tensor_from_bytes(
            data + tensor_offset, shape, type, copied_bytes));
  }

  phase_stats.tensor_setup_seconds = elapsed_seconds(setup_start);
  assign_fault_delta(
      setup_faults_before,
      io::current_load_fault_counts(),
      &phase_stats.tensor_setup_minor_faults,
      &phase_stats.tensor_setup_major_faults);
  io::set_last_load_phase_stats(std::move(phase_stats));
  io::set_last_mmap_load_stats(stats);
  stats.maybe_log("safetensors", file);
  return {res, metadata_map};
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensors] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensors] cannot serialize an empty array key: " + key);
    }

    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)), a, metadata);
}

} // namespace mlx::core

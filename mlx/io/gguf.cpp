// Copyright © 2023-2024 Apple Inc.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>

#include "mlx/io/gguf.h"
#include "mlx/io/mmap.h"
#include "mlx/ops.h"

namespace mlx::core {

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

template <typename T>
T read_unaligned(const uint8_t* ptr) {
  T out;
  std::memcpy(&out, ptr, sizeof(T));
  return out;
}

void validate_next_key_layout(const gguf_ctx* ctx) {
  constexpr uint64_t min_key_bytes = sizeof(uint64_t) + sizeof(uint32_t);

  if (ctx->off > ctx->size || (ctx->size - ctx->off) < min_key_bytes) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF metadata section (truncated key header).");
  }

  uint64_t key_len = read_unaligned<uint64_t>(ctx->data + ctx->off);
  if (key_len > (ctx->size - ctx->off - min_key_bytes)) {
    throw std::runtime_error(
        "[load_gguf] Malformed or unsupported GGUF key encoding.");
  }
}

uint64_t validate_and_advance_tensor_header(
    const gguf_ctx* ctx,
    uint64_t offset) {
  constexpr uint64_t name_len_bytes = sizeof(uint64_t);
  constexpr uint64_t num_dim_bytes = sizeof(uint32_t);
  constexpr uint64_t type_bytes = sizeof(uint32_t);
  constexpr uint64_t tensor_offset_bytes = sizeof(uint64_t);

  if (offset > ctx->size ||
      (ctx->size - offset) <
          name_len_bytes + num_dim_bytes + type_bytes + tensor_offset_bytes) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF tensor header (truncated tensor prefix).");
  }

  uint64_t name_len = read_unaligned<uint64_t>(ctx->data + offset);
  offset += name_len_bytes;
  if (name_len > (ctx->size - offset)) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF tensor header (name exceeds file bounds).");
  }
  offset += name_len;

  if ((ctx->size - offset) < num_dim_bytes + type_bytes + tensor_offset_bytes) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF tensor header (missing tensor fields).");
  }

  uint32_t ndim = read_unaligned<uint32_t>(ctx->data + offset);
  offset += num_dim_bytes;
  if (ndim == 0 || ndim > GGUF_TENSOR_MAX_DIM) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF tensor header (invalid ndim).");
  }

  uint64_t dims_bytes = static_cast<uint64_t>(ndim) * sizeof(uint64_t);
  if ((ctx->size - offset) < dims_bytes + type_bytes + tensor_offset_bytes) {
    throw std::runtime_error(
        "[load_gguf] Malformed GGUF tensor header (invalid dims span).");
  }

  offset += dims_bytes + type_bytes + tensor_offset_bytes;
  return offset;
}

void validate_tensor_section_layout(const gguf_ctx* ctx) {
  uint64_t offset = ctx->off;
  for (uint64_t i = 0; i < ctx->left_tensors; ++i) {
    offset = validate_and_advance_tensor_header(ctx, offset);
  }
}

struct Nvfp4TensorRecord {
  std::string name;
  uint32_t rank;
  uint64_t nbytes;
  uint64_t offset;
};

struct Nvfp4CompatLayout {
  std::unordered_map<std::string, std::string> metadata;
  std::vector<Nvfp4TensorRecord> tensors;
  uint64_t offset_bias{0};
};

bool checked_u64_add(uint64_t a, uint64_t b, uint64_t* out) {
  if (a > (std::numeric_limits<uint64_t>::max() - b)) {
    return false;
  }
  *out = a + b;
  return true;
}

class Nvfp4Cursor {
 public:
  Nvfp4Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  template <typename T>
  T read_scalar(const char* label) {
    require(sizeof(T), label);
    T value = read_unaligned<T>(data_ + offset_);
    offset_ += sizeof(T);
    return value;
  }

  std::string read_string(size_t len, const char* label) {
    require(len, label);
    std::string out(reinterpret_cast<const char*>(data_ + offset_), len);
    offset_ += len;
    return out;
  }

  std::string read_ascii_token(const char* label) {
    size_t start = offset_;
    while (offset_ < size_ && data_[offset_] >= 32 && data_[offset_] <= 126) {
      offset_++;
    }
    if (offset_ == start) {
      throw std::runtime_error(
          std::string("[load_gguf] NVFP4 parse error: ") + label);
    }
    return std::string(
        reinterpret_cast<const char*>(data_ + start), offset_ - start);
  }

  size_t offset() const {
    return offset_;
  }

 private:
  void require(size_t len, const char* label) const {
    if (offset_ > size_ || len > (size_ - offset_)) {
      throw std::runtime_error(
          std::string("[load_gguf] NVFP4 parse error: truncated ") + label);
    }
  }

  const uint8_t* data_{nullptr};
  size_t size_{0};
  size_t offset_{0};
};

bool looks_like_nvfp4_dialect(const uint8_t* data, size_t size) {
  if (data == nullptr || size < 30) {
    return false;
  }
  if (std::memcmp(data, "GGUF", 4) != 0) {
    return false;
  }
  if (read_unaligned<uint32_t>(data + 4) != 3) {
    return false;
  }
  return std::memcmp(data + 24, "format", 6) == 0;
}

Nvfp4CompatLayout parse_nvfp4_layout(const uint8_t* data, size_t size) {
  Nvfp4Cursor cursor(data, size);
  auto magic = cursor.read_string(4, "magic");
  if (magic != "GGUF") {
    throw std::runtime_error("[load_gguf] NVFP4 parse error: invalid magic.");
  }
  uint32_t version = cursor.read_scalar<uint32_t>("version");
  if (version != 3) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: unsupported version.");
  }

  uint64_t metadata_entries = cursor.read_scalar<uint64_t>("metadata count");
  (void)cursor.read_scalar<uint64_t>("secondary header value");
  if (metadata_entries == 0 || metadata_entries > 4096) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: invalid metadata count.");
  }

  Nvfp4CompatLayout layout;
  layout.metadata.reserve(metadata_entries);

  auto first_key = cursor.read_ascii_token("first metadata key");
  auto first_value_len =
      cursor.read_scalar<uint64_t>("first metadata value length");
  if (first_value_len > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: metadata value is too large.");
  }
  layout.metadata.emplace(
      first_key,
      cursor.read_string(
          static_cast<size_t>(first_value_len), "first metadata value"));

  for (uint64_t i = 1; i < metadata_entries; ++i) {
    auto key_len = cursor.read_scalar<uint64_t>("metadata key length");
    if (key_len > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: metadata key/value is too large.");
    }
    auto key = cursor.read_string(static_cast<size_t>(key_len), "metadata key");
    auto value_len = cursor.read_scalar<uint64_t>("metadata value length");
    if (value_len > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: metadata key/value is too large.");
    }
    auto value =
        cursor.read_string(static_cast<size_t>(value_len), "metadata value");
    layout.metadata.insert_or_assign(std::move(key), std::move(value));
  }

  auto format_it = layout.metadata.find("format");
  if (format_it == layout.metadata.end() || format_it->second != "nvfp4") {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: expected metadata format=nvfp4.");
  }

  uint64_t tensor_count = cursor.read_scalar<uint64_t>("tensor count");
  if (tensor_count == 0 || tensor_count > 1000000) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: invalid tensor count.");
  }
  layout.tensors.reserve(static_cast<size_t>(tensor_count));

  for (uint64_t i = 0; i < tensor_count; ++i) {
    auto name_len = cursor.read_scalar<uint64_t>("tensor name length");
    if (name_len == 0 || name_len > 4096 ||
        name_len > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: invalid tensor name length.");
    }
    Nvfp4TensorRecord record;
    record.name =
        cursor.read_string(static_cast<size_t>(name_len), "tensor name");
    record.rank = cursor.read_scalar<uint32_t>("tensor rank");
    record.nbytes = cursor.read_scalar<uint64_t>("tensor nbytes");
    record.offset = cursor.read_scalar<uint64_t>("tensor offset");
    layout.tensors.push_back(std::move(record));
  }

  for (size_t i = 1; i < layout.tensors.size(); ++i) {
    if (layout.tensors[i].offset < layout.tensors[i - 1].offset) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: tensor offsets are not monotonic.");
    }
  }

  uint64_t raw_end = 0;
  const auto& last = layout.tensors.back();
  if (!checked_u64_add(last.offset, last.nbytes, &raw_end)) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: tensor offset overflow.");
  }
  if (raw_end > size) {
    layout.offset_bias = raw_end - static_cast<uint64_t>(size);
  }

  const auto header_end = static_cast<uint64_t>(cursor.offset());
  for (const auto& t : layout.tensors) {
    if (t.offset < layout.offset_bias) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: invalid tensor offset bias.");
    }
    uint64_t adjusted_offset = t.offset - layout.offset_bias;
    uint64_t adjusted_end = 0;
    if (!checked_u64_add(adjusted_offset, t.nbytes, &adjusted_end) ||
        adjusted_end > size) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 parse error: tensor range out of file bounds.");
    }
  }
  if ((layout.tensors.front().offset - layout.offset_bias) < header_end) {
    throw std::runtime_error(
        "[load_gguf] NVFP4 parse error: tensor data overlaps headers.");
  }

  return layout;
}

GGUFLoad load_nvfp4_compat(
    const std::shared_ptr<io::MappedFile>& mapped_file,
    const std::string& file,
    const LoadOptions& options) {
  auto* bytes = reinterpret_cast<const uint8_t*>(mapped_file->data());
  auto parsed = parse_nvfp4_layout(bytes, mapped_file->size());

  std::unordered_map<std::string, GGUFMetaData> metadata;
  metadata.reserve(parsed.metadata.size());
  for (const auto& [key, value] : parsed.metadata) {
    metadata.insert({key, value});
  }

  io::MmapLoadStats stats;
  std::optional<array> mapped_base;
  if (options.memory_map) {
    auto mapped_file_owner = mapped_file;
    mapped_base = io::make_mapped_base_array(
        mapped_file->data(),
        mapped_file->size(),
        [mapped_file_owner](allocator::Buffer) mutable {
          mapped_file_owner.reset();
        });
    if (!mapped_base.has_value()) {
      stats.record_fallback("make_buffer_failed");
    }
  }

  std::unordered_map<std::string, array> arrays;
  arrays.reserve(parsed.tensors.size());
  for (const auto& tensor : parsed.tensors) {
    if (tensor.nbytes >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 tensor exceeds host size_t capacity.");
    }
    if (tensor.nbytes >
        static_cast<uint64_t>(std::numeric_limits<ShapeElem>::max())) {
      throw std::runtime_error(
          "[load_gguf] NVFP4 tensor exceeds MLX shape element limits.");
    }

    size_t byte_offset =
        static_cast<size_t>(tensor.offset - parsed.offset_bias);
    size_t byte_count = static_cast<size_t>(tensor.nbytes);
    Shape shape{static_cast<ShapeElem>(byte_count)};

    if (mapped_base.has_value()) {
      std::string fallback_reason;
      auto view = io::make_mapped_view(
          mapped_base.value(), byte_offset, shape, uint8, &fallback_reason);
      if (view.has_value()) {
        stats.record_mapped(view->nbytes());
        arrays.insert({tensor.name, std::move(*view)});
        continue;
      }
      stats.record_fallback(
          fallback_reason.empty() ? "view_creation_failed" : fallback_reason);
    }

    auto buffer = allocator::malloc(byte_count);
    std::memcpy(buffer.raw_ptr(), bytes + byte_offset, byte_count);
    auto copied = array(buffer, shape, uint8);
    stats.record_copied(copied.nbytes());
    arrays.insert({tensor.name, std::move(copied)});
  }

  stats.maybe_log("gguf_nvfp4", file);
  return {arrays, metadata};
}

std::optional<uint32_t> dtype_to_gguf_tensor_type(const Dtype& dtype) {
  switch (dtype) {
    case float32:
      return GGUF_TYPE_F32;
    case float16:
      return GGUF_TYPE_F16;
    case int8:
      return GGUF_TYPE_I8;
    case int16:
      return GGUF_TYPE_I16;
    case int32:
      return GGUF_TYPE_I32;
    default:
      return {};
  }
}

std::optional<Dtype> gguf_type_to_dtype(const uint32_t& gguf_type) {
  switch (gguf_type) {
    case GGUF_TYPE_F32:
      return float32;
    case GGUF_TYPE_F16:
      return float16;
    case GGUF_TYPE_I8:
      return int8;
    case GGUF_TYPE_I16:
      return int16;
    case GGUF_TYPE_I32:
      return int32;
    default:
      return {};
  }
}

Shape get_shape(const gguf_tensor& tensor) {
  Shape shape;
  // The dimension order in GGML is the reverse of the order used in MLX.
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    shape.push_back(tensor.dim[i]);
  }
  return shape;
}

std::tuple<allocator::Buffer, Dtype> extract_tensor_data(gguf_tensor* tensor) {
  if (tensor == nullptr) {
    throw std::invalid_argument(
        "[extract_tensor_data] Input tensor pointer is null.");
  }
  std::optional<Dtype> equivalent_dtype = gguf_type_to_dtype(tensor->type);
  // If there's an equivalent type, we can simply copy.
  if (equivalent_dtype.has_value()) {
    if (tensor->weights_data == nullptr) {
      throw std::runtime_error("[load_gguf] NULL tensor data pointer");
    }
    allocator::Buffer buffer = allocator::malloc(tensor->bsize);
    memcpy(
        buffer.raw_ptr(),
        tensor->weights_data,
        tensor->num_weights * equivalent_dtype.value().size());
    return {buffer, equivalent_dtype.value()};
  }
  // Otherwise, we convert to float16.
  // TODO: Add other dequantization options.
  int16_t* data = gguf_tensor_to_f16(tensor);
  if (data == NULL) {
    throw std::runtime_error("[load_gguf] gguf_tensor_to_f16 failed");
  }
  const size_t new_size = tensor->num_weights * sizeof(int16_t);
  allocator::Buffer buffer = allocator::malloc(new_size);
  memcpy(buffer.raw_ptr(), data, new_size);
  free(data);
  return {buffer, float16};
}

void set_mx_value_from_gguf(
    gguf_ctx* ctx,
    uint32_t type,
    gguf_value* val,
    GGUFMetaData& value) {
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
      value = array(val->uint8, uint8);
      break;
    case GGUF_VALUE_TYPE_INT8:
      value = array(val->int8, int8);
      break;
    case GGUF_VALUE_TYPE_UINT16:
      value = array(val->uint16, uint16);
      break;
    case GGUF_VALUE_TYPE_INT16:
      value = array(val->int16, int16);
      break;
    case GGUF_VALUE_TYPE_UINT32:
      value = array(val->uint32, uint32);
      break;
    case GGUF_VALUE_TYPE_INT32:
      value = array(val->int32, int32);
      break;
    case GGUF_VALUE_TYPE_UINT64:
      value = array(val->uint64, uint64);
      break;
    case GGUF_VALUE_TYPE_INT64:
      value = array(val->int64, int64);
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
      value = array(val->float32, float32);
      break;
    case GGUF_VALUE_TYPE_BOOL:
      value = array(val->boolval, bool_);
      break;
    case GGUF_VALUE_TYPE_STRING:
      value =
          std::string(val->string.string, static_cast<int>(val->string.len));
      break;
    case GGUF_VALUE_TYPE_FLOAT64:
      value = array(val->float64, float32);
      break;
    case GGUF_VALUE_TYPE_ARRAY: {
      char* data = reinterpret_cast<char*>(val) + gguf_array_header_size;
      auto size = static_cast<int>(val->array.len);
      if (val->array.type == GGUF_VALUE_TYPE_ARRAY) {
        throw std::invalid_argument(
            "[load_gguf] Only supports loading 1-layer of nested arrays.");
      }
      switch (val->array.type) {
        case GGUF_VALUE_TYPE_UINT8:
          value = array(reinterpret_cast<uint8_t*>(data), {size}, uint8);
          break;
        case GGUF_VALUE_TYPE_INT8:
          value = array(reinterpret_cast<int8_t*>(data), {size}, int8);
          break;
        case GGUF_VALUE_TYPE_UINT16:
          value = array(reinterpret_cast<uint16_t*>(data), {size}, uint16);
          break;
        case GGUF_VALUE_TYPE_INT16:
          value = array(reinterpret_cast<int16_t*>(data), {size}, int16);
          break;
        case GGUF_VALUE_TYPE_UINT32:
          value = array(reinterpret_cast<uint32_t*>(data), {size}, uint32);
          break;
        case GGUF_VALUE_TYPE_INT32:
          value = array(reinterpret_cast<int32_t*>(data), {size}, int32);
          break;
        case GGUF_VALUE_TYPE_UINT64:
          value = array(reinterpret_cast<uint64_t*>(data), {size}, uint64);
          break;
        case GGUF_VALUE_TYPE_INT64:
          value = array(reinterpret_cast<int64_t*>(data), {size}, int64);
          break;
        case GGUF_VALUE_TYPE_FLOAT32:
          value = array(reinterpret_cast<float*>(data), {size}, float32);
          break;
        case GGUF_VALUE_TYPE_BOOL:
          value = array(reinterpret_cast<bool*>(data), {size}, bool_);
          break;
        case GGUF_VALUE_TYPE_STRING: {
          std::vector<std::string> strs(size);
          for (auto& str : strs) {
            auto str_val = reinterpret_cast<gguf_string*>(data);
            data += (str_val->len + sizeof(gguf_string));
            str = std::string(str_val->string, static_cast<int>(str_val->len));
          }
          value = std::move(strs);
          break;
        }
        case GGUF_VALUE_TYPE_FLOAT64:
          value = array(reinterpret_cast<double*>(data), {size}, float32);
          break;
        default:
          throw std::runtime_error(
              "[load_gguf] Multiple levels of nested arrays are not supported.");
      }
      break;
    }
    default:
      throw std::runtime_error("[load_gguf] Received unexpected type.");
      break;
  }
  gguf_do_with_value(ctx, type, val, nullptr, 0, 0, nullptr);
}

std::unordered_map<std::string, GGUFMetaData> load_metadata(gguf_ctx* ctx) {
  std::unordered_map<std::string, GGUFMetaData> metadata;
  while (ctx->left_kv > 0) {
    validate_next_key_layout(ctx);
    gguf_key key;
    if (!gguf_get_key(ctx, &key)) {
      break;
    }
    std::string key_name = std::string(key.name, key.namelen);
    auto& val = metadata.insert({key_name, GGUFMetaData{}}).first->second;
    set_mx_value_from_gguf(ctx, key.type, key.val, val);
  }
  return metadata;
}

std::unordered_map<std::string, array> load_arrays(
    gguf_ctx* ctx,
    const std::optional<array>& mapped_base,
    io::MmapLoadStats* stats) {
  std::unordered_map<std::string, array> array_map;

  validate_tensor_section_layout(ctx);

  auto check_insert = [](const auto& inserted) {
    if (!inserted.second) {
      std::ostringstream msg;
      msg << "[load_gguf] Duplicate parameter name " << inserted.first->second
          << " this can happend when loading quantized tensors.";
      throw std::runtime_error(msg.str());
    }
  };

  while (ctx->left_tensors > 0) {
    gguf_tensor tensor;
    if (!gguf_get_tensor(ctx, &tensor)) {
      break;
    }
    if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1 ||
        tensor.type == GGUF_TYPE_Q8_0) {
      gguf_load_quantized(array_map, tensor);
      if (stats) {
        stats->record_fallback("quantized_conversion", tensor.bsize);
      }
      continue;
    }

    std::string name(tensor.name, tensor.namelen);
    auto equivalent_dtype = gguf_type_to_dtype(tensor.type);
    auto shape = get_shape(tensor);

    if (mapped_base.has_value() && equivalent_dtype.has_value()) {
      std::string fallback_reason;
      auto view = io::make_mapped_view(
          mapped_base.value(),
          tensor.offset,
          shape,
          equivalent_dtype.value(),
          &fallback_reason);
      if (view.has_value()) {
        if (stats) {
          stats->record_mapped(view->nbytes());
        }
        check_insert(array_map.insert({name, std::move(*view)}));
        continue;
      }

      if (stats) {
        stats->record_fallback(
            fallback_reason.empty() ? "view_creation_failed" : fallback_reason);
      }
    } else if (stats && !equivalent_dtype.has_value()) {
      stats->record_fallback("dtype_conversion");
    }

    const auto& [data, dtype] = extract_tensor_data(&tensor);
    array loaded_array = array(data, std::move(shape), dtype);
    if (stats) {
      stats->record_copied(loaded_array.nbytes());
    }
    check_insert(array_map.insert({name, loaded_array}));
  }
  return array_map;
}

GGUFLoad load_gguf(const std::string& file, StreamOrDevice s) {
  return load_gguf(file, s, LoadOptions{});
}

GGUFLoad load_gguf(
    const std::string& file,
    StreamOrDevice s,
    const LoadOptions& options) {
  (void)s;
  bool exists;
  {
    std::ifstream f(file.c_str());
    exists = f.good();
  }
  if (!exists) {
    throw std::invalid_argument("[load_gguf] Failed to open " + file);
  }

  if (options.gguf_nvfp4_compat) {
    auto mapped_file = io::map_file_readonly(file);
    auto* data = reinterpret_cast<const uint8_t*>(mapped_file->data());
    if (looks_like_nvfp4_dialect(data, mapped_file->size())) {
      return load_nvfp4_compat(mapped_file, file, options);
    }
  }

  auto ctx = std::shared_ptr<gguf_ctx>(gguf_open(file.data()), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_init failed");
  }

  auto metadata = load_metadata(ctx.get());
  if (!options.memory_map) {
    auto arrays = load_arrays(ctx.get(), std::nullopt, nullptr);
    return {arrays, metadata};
  }

  io::MmapLoadStats stats;
  auto mapped_base = io::make_mapped_base_array(
      ctx->data, ctx->size, [ctx](allocator::Buffer) mutable { ctx.reset(); });
  if (!mapped_base.has_value()) {
    stats.record_fallback("make_buffer_failed");
    auto arrays = load_arrays(ctx.get(), std::nullopt, &stats);
    stats.maybe_log("gguf", file);
    return {arrays, metadata};
  }

  auto arrays = load_arrays(ctx.get(), mapped_base, &stats);
  stats.maybe_log("gguf", file);
  return {arrays, metadata};
}

void append_kv_array(
    gguf_ctx* ctx,
    const std::string& key,
    array& val,
    uint32_t gguf_type) {
  if (val.ndim() == 1) {
    size_t gguf_size = val.nbytes() + gguf_array_header_size;
    std::vector<char> val_vec(gguf_size);
    gguf_value* gguf_val = reinterpret_cast<gguf_value*>(val_vec.data());
    gguf_val->array.type = gguf_type;
    gguf_val->array.len = val.size();
    memcpy(
        val_vec.data() + gguf_array_header_size,
        val.data<char>(),
        val.nbytes());
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        GGUF_VALUE_TYPE_ARRAY,
        reinterpret_cast<void*>(val_vec.data()),
        gguf_size);
  } else {
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        gguf_type,
        reinterpret_cast<void*>(val.data<char>()),
        val.nbytes());
  }
}

void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> metadata /* = {} */) {
  // Add .gguf to file name if it is not there
  if (file.length() < 5 || file.substr(file.length() - 5, 5) != ".gguf") {
    file += ".gguf";
  }

  std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(
      gguf_create(file.c_str(), GGUF_OVERWRITE), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[save_gguf] gguf_create failed");
  }

  auto string_to_gguf = [](char* dst, const std::string& src) {
    gguf_string* val = reinterpret_cast<gguf_string*>(dst);
    val->len = src.length();
    memcpy(val->string, src.c_str(), src.length());
  };

  // Save any meta data
  for (auto& [key, value] : metadata) {
    if (auto pv = std::get_if<std::string>(&value); pv) {
      const std::string& str = *pv;
      size_t size = sizeof(gguf_string) + str.length();
      std::vector<char> val_vec(size);
      string_to_gguf(val_vec.data(), str);
      gguf_append_kv(
          ctx.get(),
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_STRING,
          static_cast<void*>(val_vec.data()),
          size);
    } else if (auto pv = std::get_if<std::vector<std::string>>(&value); pv) {
      const auto& str_vec = *pv;
      auto mem_size = std::accumulate(
          str_vec.begin(), str_vec.end(), 0, [](size_t accum, const auto& s) {
            return accum + s.size();
          });
      mem_size += str_vec.size() * sizeof(gguf_string) + gguf_array_header_size;
      std::vector<char> val_vec(mem_size);
      gguf_value* val = reinterpret_cast<gguf_value*>(val_vec.data());
      val->array.type = GGUF_VALUE_TYPE_STRING;
      val->array.len = str_vec.size();
      auto str_ptr = val_vec.data() + gguf_array_header_size;
      for (auto& str : str_vec) {
        string_to_gguf(str_ptr, str);
        str_ptr += str.length() + sizeof(gguf_string);
      }
      gguf_append_kv(
          ctx.get(),
          key.c_str(),
          key.length(),
          GGUF_VALUE_TYPE_ARRAY,
          static_cast<void*>(val),
          mem_size);
    } else if (auto pv = std::get_if<array>(&value); pv) {
      array v = *pv;
      if (v.ndim() > 1) {
        throw std::runtime_error(
            "[save_gguf] Cannot save arrays with more than one dimension.");
      }
      if (v.size() == 0) {
        throw std::runtime_error("[save_gguf] Cannot save empty arrays.");
      }

      eval(v);
      if (!v.flags().row_contiguous) {
        v = reshape(flatten(v), v.shape());
      }
      if (!v.flags().row_contiguous) {
        throw std::runtime_error(
            "[save_gguf] Cannot save non contiguous arrays.");
      }
      switch (v.dtype()) {
        case float32:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_FLOAT32);
          break;
        case int64:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT64);
          break;
        case int32:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT32);
          break;
        case int16:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT16);
          break;
        case int8:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT8);
          break;
        case uint64:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT64);
          break;
        case uint32:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT32);
          break;
        case uint16:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT16);
          break;
        case uint8:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT8);
          break;
        case bool_:
          append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_BOOL);
          break;
        default:
          std::ostringstream msg;
          msg << "[save_gguf] array type " << v.dtype()
              << " not support for metadata.";
          throw std::invalid_argument(msg.str());
      }
    } else {
      throw std::runtime_error(
          "[save_gguf] Received unexpected type in metadata");
    }
  }

  // Tensor offsets are relative to data section, so we start at offset 0.
  uint64_t tensor_offset = 0;

  // First, append the tensor info
  for (auto& [key, arr] : array_map) {
    arr.eval();

    // Try to make it row contiguous
    if (!arr.flags().row_contiguous) {
      arr = reshape(flatten(arr), arr.shape());
      arr.eval();
    }

    // Has to be row-major now but, check one more time in case
    // any of the above change in the future
    if (!arr.flags().row_contiguous) {
      throw std::invalid_argument(
          "[save_gguf] can only serialize row-major arrays");
    }

    tensor_offset += gguf_get_alignment_padding(ctx->alignment, tensor_offset);
    const std::optional<uint32_t> gguf_type =
        dtype_to_gguf_tensor_type(arr.dtype());
    if (!gguf_type.has_value()) {
      std::ostringstream msg;
      msg << "[save_gguf] dtype " << arr.dtype() << " is not supported";
      throw std::runtime_error(msg.str());
    }
    const char* tensorname = key.c_str();
    const uint64_t namelen = key.length();
    const uint32_t num_dim = arr.ndim();
    std::vector<uint64_t> dim(num_dim);
    for (int i = 0; i < num_dim; i++) {
      dim[i] = arr.shape()[num_dim - 1 - i];
    }
    if (!gguf_append_tensor_info(
            ctx.get(),
            tensorname,
            namelen,
            num_dim,
            dim.data(),
            gguf_type.value(),
            tensor_offset)) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
    }
    tensor_offset += arr.nbytes();
  }

  // Then, append the tensor weights
  for (const auto& [key, arr] : array_map) {
    if (!gguf_append_tensor_data(
            ctx.get(), (void*)arr.data<void>(), arr.nbytes())) {
      throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
    }
  }
}

} // namespace mlx::core

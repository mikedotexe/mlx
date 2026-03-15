// Copyright © 2026 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "mlx/array.h"

namespace mlx::core::io {

struct MmapLoadStats {
  size_t mapped_bytes{0};
  size_t copied_bytes{0};
  size_t fallback_tensors{0};
  std::unordered_map<std::string, size_t> fallback_reasons;

  void record_mapped(size_t bytes);
  void record_copied(size_t bytes);
  void record_fallback(const std::string& reason, size_t copied_bytes = 0);
  void maybe_log(const std::string& tag, const std::string& file) const;
};

class MappedFile {
 public:
  explicit MappedFile(std::string path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  void* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  const std::string& path() const {
    return path_;
  }

 private:
  std::string path_;
  int fd_{-1};
  void* data_{nullptr};
  size_t size_{0};
};

std::shared_ptr<MappedFile> map_file_readonly(const std::string& path);

std::optional<array>
make_mapped_base_array(void* data, size_t size, Deleter deleter);

std::optional<array> make_mapped_view(
    const array& base,
    size_t byte_offset,
    Shape shape,
    Dtype dtype,
    std::string* failure_reason = nullptr);

} // namespace mlx::core::io

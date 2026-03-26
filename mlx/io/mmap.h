// Copyright © 2026 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/array.h"

namespace mlx::core::io {

struct MmapLoadStats {
  size_t mapped_bytes{0};
  size_t copied_bytes{0};
  size_t fallback_tensors{0};
  std::unordered_map<std::string, size_t> fallback_reasons;
  // Materialized destination bytes resident after the fallback copy.
  std::unordered_map<std::string, size_t> fallback_reason_bytes;
  // Original source bytes consumed from file-backed storage for the fallback.
  std::unordered_map<std::string, size_t> fallback_reason_source_bytes;
  std::unordered_map<std::string, std::string> hotset_promoted_tensors;
  std::string hotset_promotion_strategy;

  void record_mapped(size_t bytes);
  void record_copied(size_t bytes);
  void record_fallback(
      const std::string& reason,
      size_t copied_bytes = 0,
      std::optional<size_t> source_bytes = std::nullopt);
  void record_hotset_selection(
      const std::string& name,
      const std::string& reason,
      std::string strategy = "name_aware");
  void maybe_log(const std::string& tag, const std::string& file) const;
};

struct HotsetPromotionCandidate {
  std::string name;
  size_t bytes{0};
};

struct HotsetPromotionSelection {
  std::unordered_set<std::string> names;
  std::unordered_map<std::string, std::string> reasons;
  std::string strategy{"name_aware"};
};

struct LoadPhaseStats {
  std::string tag;
  bool memory_map{false};
  double open_map_seconds{0.0};
  double parse_seconds{0.0};
  double tensor_setup_seconds{0.0};
  size_t open_map_minor_faults{0};
  size_t open_map_major_faults{0};
  size_t parse_minor_faults{0};
  size_t parse_major_faults{0};
  size_t tensor_setup_minor_faults{0};
  size_t tensor_setup_major_faults{0};
};

struct LoadFaultCounts {
  size_t minor_faults{0};
  size_t major_faults{0};
};

MLX_API std::optional<MmapLoadStats> last_mmap_load_stats(bool clear = false);
MLX_API void set_last_mmap_load_stats(MmapLoadStats stats);
MLX_API void clear_last_mmap_load_stats();
MLX_API std::optional<LoadPhaseStats> last_load_phase_stats(bool clear = false);
MLX_API void set_last_load_phase_stats(LoadPhaseStats stats);
MLX_API void clear_last_load_phase_stats();
MLX_API LoadFaultCounts current_load_fault_counts();
MLX_API HotsetPromotionSelection select_hotset_promotion_candidates(
    std::vector<HotsetPromotionCandidate> candidates,
    size_t top_k,
    size_t min_bytes = 0);

class MappedFile {
 public:
  explicit MappedFile(
      std::string path,
      std::string_view prefetch_strategy = "sequential");
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

std::shared_ptr<MappedFile> map_file_readonly(
    const std::string& path,
    std::string_view prefetch_strategy = "sequential");

std::optional<array>
make_mapped_base_array(void* data, size_t size, Deleter deleter);

std::optional<array> make_mapped_view(
    const array& base,
    size_t byte_offset,
    Shape shape,
    Dtype dtype,
    std::string* failure_reason = nullptr);

} // namespace mlx::core::io

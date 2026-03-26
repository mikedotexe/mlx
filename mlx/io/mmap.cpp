// Copyright © 2026 Apple Inc.

#include "mlx/io/mmap.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mlx::core::io {

namespace {

thread_local std::optional<MmapLoadStats> g_last_mmap_load_stats;
thread_local std::optional<LoadPhaseStats> g_last_load_phase_stats;

bool debug_enabled() {
  if (const char* value = std::getenv("MLX_DEBUG_IO_MEMORY_MAP")) {
    return std::atoi(value) != 0;
  }
  return false;
}

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool contains_token(
    const std::string& lowered_name,
    std::initializer_list<std::string_view> tokens) {
  for (auto token : tokens) {
    if (lowered_name.find(token) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::pair<int, std::string> hotset_name_priority(const std::string& name) {
  const std::string lowered_name = lowercase_copy(name);
  if (contains_token(
          lowered_name,
          {
              "lm_head",
              "output.weight",
              "output_projection",
              "embed_out",
              "logits",
          })) {
    return {500, "output_head"};
  }
  if (contains_token(
          lowered_name,
          {
              "model.norm",
              "final_norm",
              "norm_out",
              "output_norm",
              "ln_f",
          })) {
    return {450, "final_norm"};
  }
  if (contains_token(
          lowered_name,
          {
              "tok_embeddings",
              "token_embd",
              "embed_tokens",
              "word_embeddings",
              "embedding",
              "wte",
          })) {
    return {400, "token_embedding"};
  }
  if (contains_token(
          lowered_name,
          {
              ".attn.",
              "attention",
              "self_attn",
              "q_proj",
              "k_proj",
              "v_proj",
              "o_proj",
          })) {
    return {300, "attention_path"};
  }
  if (contains_token(
          lowered_name,
          {
              ".mlp.",
              ".ffn.",
              "feed_forward",
              "gate_proj",
              "up_proj",
              "down_proj",
          })) {
    return {250, "mlp_path"};
  }
  return {100, "size_rank"};
}

#ifndef _WIN32
void apply_prefetch_strategy(
    void* data,
    size_t size,
    std::string_view prefetch_strategy) {
  if (prefetch_strategy == "none") {
    return;
  }
  if (prefetch_strategy == "sequential") {
    // Model loading iterates tensors sequentially; hint the kernel to prefetch
    // ahead and reclaim behind the read head. Non-fatal if it fails.
    madvise(data, size, MADV_SEQUENTIAL);
    return;
  }
  if (prefetch_strategy == "willneed") {
#ifdef MADV_WILLNEED
    // Ask the kernel to stage the mapped pages more aggressively.
    madvise(data, size, MADV_WILLNEED);
    return;
#else
    throw std::runtime_error(
        "[mmap] MADV_WILLNEED is not supported in this build.");
#endif
  }
  throw std::invalid_argument(
      "[mmap] Unsupported prefetch strategy: " +
      std::string(prefetch_strategy));
}
#endif

} // namespace

std::optional<MmapLoadStats> last_mmap_load_stats(bool clear) {
  auto stats = g_last_mmap_load_stats;
  if (clear) {
    g_last_mmap_load_stats.reset();
  }
  return stats;
}

void set_last_mmap_load_stats(MmapLoadStats stats) {
  g_last_mmap_load_stats = std::move(stats);
}

void clear_last_mmap_load_stats() {
  g_last_mmap_load_stats.reset();
}

std::optional<LoadPhaseStats> last_load_phase_stats(bool clear) {
  auto stats = g_last_load_phase_stats;
  if (clear) {
    g_last_load_phase_stats.reset();
  }
  return stats;
}

void set_last_load_phase_stats(LoadPhaseStats stats) {
  g_last_load_phase_stats = std::move(stats);
}

void clear_last_load_phase_stats() {
  g_last_load_phase_stats.reset();
}

LoadFaultCounts current_load_fault_counts() {
#ifdef _WIN32
  return {};
#else
  rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return {};
  }
  return {
      static_cast<size_t>(usage.ru_minflt),
      static_cast<size_t>(usage.ru_majflt),
  };
#endif
}

HotsetPromotionSelection select_hotset_promotion_candidates(
    std::vector<HotsetPromotionCandidate> candidates,
    size_t top_k,
    size_t min_bytes) {
  HotsetPromotionSelection selection;
  if (top_k == 0 || candidates.empty()) {
    return selection;
  }

  candidates.erase(
      std::remove_if(
          candidates.begin(),
          candidates.end(),
          [min_bytes](const auto& candidate) {
            return candidate.bytes < min_bytes;
          }),
      candidates.end());
  if (candidates.empty()) {
    return selection;
  }

  struct RankedCandidate {
    HotsetPromotionCandidate candidate;
    int priority{0};
    std::string reason;
  };

  std::vector<RankedCandidate> ranked;
  ranked.reserve(candidates.size());
  for (auto& candidate : candidates) {
    auto [priority, reason] = hotset_name_priority(candidate.name);
    ranked.push_back({std::move(candidate), priority, std::move(reason)});
  }

  std::sort(
      ranked.begin(),
      ranked.end(),
      [](const auto& lhs, const auto& rhs) {
        if (lhs.priority != rhs.priority) {
          return lhs.priority > rhs.priority;
        }
        if (lhs.candidate.bytes != rhs.candidate.bytes) {
          return lhs.candidate.bytes > rhs.candidate.bytes;
        }
        return lhs.candidate.name < rhs.candidate.name;
      });

  const size_t limit = std::min(top_k, ranked.size());
  selection.names.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    selection.names.insert(ranked[i].candidate.name);
    std::ostringstream reason;
    reason << ranked[i].reason << ",bytes=" << ranked[i].candidate.bytes;
    selection.reasons.emplace(ranked[i].candidate.name, reason.str());
  }
  return selection;
}

void MmapLoadStats::record_mapped(size_t bytes) {
  mapped_bytes += bytes;
}

void MmapLoadStats::record_copied(size_t bytes) {
  copied_bytes += bytes;
}

void MmapLoadStats::record_fallback(
    const std::string& reason,
    size_t copied_bytes_,
    std::optional<size_t> source_bytes) {
  fallback_tensors++;
  fallback_reasons[reason]++;
  fallback_reason_bytes[reason] += copied_bytes_;
  fallback_reason_source_bytes[reason] += source_bytes.value_or(copied_bytes_);
  copied_bytes += copied_bytes_;
}

void MmapLoadStats::record_hotset_selection(
    const std::string& name,
    const std::string& reason,
    std::string strategy) {
  hotset_promoted_tensors[name] = reason;
  if (!strategy.empty()) {
    hotset_promotion_strategy = std::move(strategy);
  }
}

void MmapLoadStats::maybe_log(const std::string& tag, const std::string& file)
    const {
  if (!debug_enabled()) {
    return;
  }
  std::ostringstream msg;
  msg << "[io mmap] " << tag << " file=" << file
      << " mapped_bytes=" << mapped_bytes << " copied_bytes=" << copied_bytes
      << " fallback_tensors=" << fallback_tensors;
  if (!fallback_reasons.empty()) {
    msg << " fallback_reasons={";
    bool first = true;
    for (const auto& [reason, count] : fallback_reasons) {
      if (!first) {
        msg << ",";
      }
      first = false;
      msg << reason << ":" << count;
    }
    msg << "}";
  }
  if (!fallback_reason_bytes.empty()) {
    msg << " fallback_reason_bytes={";
    bool first = true;
    for (const auto& [reason, bytes] : fallback_reason_bytes) {
      if (!first) {
        msg << ",";
      }
      first = false;
      msg << reason << ":" << bytes;
    }
    msg << "}";
  }
  if (!fallback_reason_source_bytes.empty()) {
    msg << " fallback_reason_source_bytes={";
    bool first = true;
    for (const auto& [reason, bytes] : fallback_reason_source_bytes) {
      if (!first) {
        msg << ",";
      }
      first = false;
      msg << reason << ":" << bytes;
    }
    msg << "}";
  }
  std::cerr << msg.str() << std::endl;
}

MappedFile::MappedFile(
    std::string path,
    std::string_view prefetch_strategy)
    : path_(std::move(path)) {
#ifdef _WIN32
  throw std::runtime_error(
      "[mmap] Memory mapping is not supported on Windows in this build.");
#else
  fd_ = open(path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("[mmap] Failed to open file: " + path_);
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("[mmap] Failed to stat file: " + path_);
  }
  if (st.st_size <= 0) {
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("[mmap] File is empty: " + path_);
  }
  size_ = static_cast<size_t>(st.st_size);

  data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data_ == MAP_FAILED) {
    close(fd_);
    fd_ = -1;
    data_ = nullptr;
    throw std::runtime_error("[mmap] Failed to map file: " + path_);
  }
  apply_prefetch_strategy(data_, size_, prefetch_strategy);
#endif
}

MappedFile::~MappedFile() {
#ifndef _WIN32
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
#endif
}

std::shared_ptr<MappedFile> map_file_readonly(
    const std::string& path,
    std::string_view prefetch_strategy) {
  return std::make_shared<MappedFile>(path, prefetch_strategy);
}

std::optional<array>
make_mapped_base_array(void* data, size_t size, Deleter deleter) {
  if (data == nullptr || size == 0) {
    return std::nullopt;
  }
  auto buffer = allocator::make_buffer(data, size);
  if (buffer.ptr() == nullptr) {
    return std::nullopt;
  }
  return array(buffer, Shape{1}, uint8, std::move(deleter));
}

std::optional<array> make_mapped_view(
    const array& base,
    size_t byte_offset,
    Shape shape,
    Dtype dtype,
    std::string* failure_reason) {
  auto no_op = [](allocator::Buffer) {};
  auto view = array(base.buffer(), shape, dtype, no_op);

  size_t base_size = allocator::allocator().size(base.buffer());
  if (byte_offset > base_size || view.nbytes() > (base_size - byte_offset)) {
    if (failure_reason) {
      *failure_reason = "out_of_bounds";
    }
    return std::nullopt;
  }

  if (view.itemsize() == 0 || (byte_offset % view.itemsize() != 0)) {
    if (failure_reason) {
      *failure_reason = "misaligned_offset";
    }
    return std::nullopt;
  }

  size_t elem_offset = byte_offset / view.itemsize();
  if (elem_offset > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    if (failure_reason) {
      *failure_reason = "offset_overflow";
    }
    return std::nullopt;
  }

  view.copy_shared_buffer(
      base,
      view.strides(),
      view.flags(),
      view.size(),
      static_cast<int64_t>(elem_offset));
  return view;
}

} // namespace mlx::core::io

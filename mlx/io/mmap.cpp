// Copyright © 2026 Apple Inc.

#include "mlx/io/mmap.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mlx::core::io {

namespace {

bool debug_enabled() {
  if (const char* value = std::getenv("MLX_DEBUG_IO_MEMORY_MAP")) {
    return std::atoi(value) != 0;
  }
  return false;
}

} // namespace

void MmapLoadStats::record_mapped(size_t bytes) {
  mapped_bytes += bytes;
}

void MmapLoadStats::record_copied(size_t bytes) {
  copied_bytes += bytes;
}

void MmapLoadStats::record_fallback(
    const std::string& reason,
    size_t copied_bytes_) {
  fallback_tensors++;
  fallback_reasons[reason]++;
  copied_bytes += copied_bytes_;
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
  std::cerr << msg.str() << std::endl;
}

MappedFile::MappedFile(std::string path) : path_(std::move(path)) {
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

std::shared_ptr<MappedFile> map_file_readonly(const std::string& path) {
  return std::make_shared<MappedFile>(path);
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

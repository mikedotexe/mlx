// Copyright © 2023-2024 Apple Inc.

#include "mlx/io.h"

namespace mlx::core {

SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  return load_safetensors(in_stream, s, LoadOptions{});
}

SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader>,
    StreamOrDevice,
    const LoadOptions&) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(file, s, LoadOptions{});
}

SafetensorsLoad
load_safetensors(const std::string&, StreamOrDevice, const LoadOptions&) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::shared_ptr<io::Writer>,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

} // namespace mlx::core

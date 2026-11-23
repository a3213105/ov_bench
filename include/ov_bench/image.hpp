#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace ovbench {

struct Image {
    size_t width;
    size_t height;
    std::vector<uint8_t> rgb;
};

Image decode_image(const std::filesystem::path& path);
ov::Tensor preprocess_image(const std::filesystem::path& path, const std::string& family,
                            size_t size, double scale = 1.0);

}  // namespace ovbench

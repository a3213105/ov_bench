#include "ov_bench/image.hpp"
#include "ov_bench/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <jpeglib.h>
#include <png.h>
#include <setjmp.h>

namespace ovbench {
namespace {

constexpr int kCoefficientBits = 22;
constexpr int64_t kCoefficientScale = int64_t{1} << kCoefficientBits;

size_t checked_rgb_bytes(size_t width, size_t height) {
    if (width == 0 || height == 0 ||
        width > std::numeric_limits<size_t>::max() / height ||
        width * height > std::numeric_limits<size_t>::max() / 3) {
        throw std::runtime_error("Invalid or oversized image dimensions");
    }
    return width * height * 3;
}

bool has_png_signature(const std::array<unsigned char, 8>& bytes) {
    static constexpr std::array<unsigned char, 8> signature = {
        0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a
    };
    return bytes == signature;
}

struct JpegErrorManager {
    jpeg_error_mgr base;
    jmp_buf jump_buffer;
    char message[JMSG_LENGTH_MAX];
};

void jpeg_error_exit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    (*common->err->format_message)(common, error->message);
    longjmp(error->jump_buffer, 1);
}

Image decode_jpeg(const std::filesystem::path& path) {
    const std::string filename = path.string();
    FILE* file = std::fopen(filename.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Unable to open image: " + filename);
    }

    auto* decoder = static_cast<jpeg_decompress_struct*>(
        std::calloc(1, sizeof(jpeg_decompress_struct)));
    auto* error = static_cast<JpegErrorManager*>(std::calloc(1, sizeof(JpegErrorManager)));
    if (!decoder || !error) {
        std::free(decoder);
        std::free(error);
        std::fclose(file);
        throw std::bad_alloc();
    }
    decoder->err = jpeg_std_error(&error->base);
    error->base.error_exit = jpeg_error_exit;

    volatile bool decoder_created = false;
    unsigned char* volatile pixels = nullptr;
    if (setjmp(error->jump_buffer)) {
        char message[JMSG_LENGTH_MAX];
        std::memcpy(message, error->message, sizeof(message));
        if (pixels) {
            std::free(const_cast<unsigned char*>(pixels));
        }
        if (decoder_created) {
            jpeg_destroy_decompress(decoder);
        }
        std::free(decoder);
        std::free(error);
        std::fclose(file);
        throw std::runtime_error("Unable to decode JPEG " + filename + ": " + message);
    }

    decoder_created = true;
    jpeg_create_decompress(decoder);
    jpeg_stdio_src(decoder, file);
    jpeg_read_header(decoder, TRUE);
    decoder->out_color_space = JCS_RGB;
    jpeg_start_decompress(decoder);

    const size_t width = decoder->output_width;
    const size_t height = decoder->output_height;
    if (width == 0 || height == 0 ||
        width > std::numeric_limits<size_t>::max() / height ||
        width * height > std::numeric_limits<size_t>::max() / 3) {
        jpeg_destroy_decompress(decoder);
        std::free(decoder);
        std::free(error);
        std::fclose(file);
        throw std::runtime_error("Invalid or oversized image dimensions");
    }
    const size_t byte_count = width * height * 3;
    if (decoder->output_components != 3) {
        jpeg_destroy_decompress(decoder);
        std::free(decoder);
        std::free(error);
        std::fclose(file);
        throw std::runtime_error("JPEG decoder did not produce RGB pixels: " + filename);
    }
    pixels = static_cast<unsigned char*>(std::malloc(byte_count));
    if (!pixels) {
        jpeg_destroy_decompress(decoder);
        std::free(decoder);
        std::free(error);
        std::fclose(file);
        throw std::bad_alloc();
    }

    while (decoder->output_scanline < decoder->output_height) {
        JSAMPROW row = const_cast<unsigned char*>(pixels) +
                       static_cast<size_t>(decoder->output_scanline) * width * 3;
        jpeg_read_scanlines(decoder, &row, 1);
    }
    jpeg_finish_decompress(decoder);
    jpeg_destroy_decompress(decoder);
    decoder_created = false;
    std::free(decoder);
    std::free(error);
    std::fclose(file);

    std::unique_ptr<unsigned char, decltype(&std::free)> decoded_pixels(
        const_cast<unsigned char*>(pixels), std::free);
    Image image{width, height, std::vector<uint8_t>(byte_count)};
    std::memcpy(image.rgb.data(), decoded_pixels.get(), byte_count);
    return image;
}

Image decode_png(const std::filesystem::path& path) {
    struct PngImageGuard {
        png_image image{};
        ~PngImageGuard() {
            png_image_free(&image);
        }
    } guard;
    png_image& png = guard.image;
    png.version = PNG_IMAGE_VERSION;
    const std::string filename = path.string();
    if (!png_image_begin_read_from_file(&png, filename.c_str())) {
        throw std::runtime_error("Unable to decode PNG " + filename + ": " + png.message);
    }
    png.format = PNG_FORMAT_RGBA;
    if (png.width == 0 || png.height == 0 ||
        static_cast<size_t>(png.width) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(png.height) ||
        static_cast<size_t>(png.width) * static_cast<size_t>(png.height) >
            std::numeric_limits<size_t>::max() / 4) {
        throw std::runtime_error("Invalid or oversized image dimensions");
    }
    const size_t width = png.width;
    const size_t height = png.height;
    const size_t pixel_count = width * height;
    std::vector<uint8_t> rgba(pixel_count * 4);
    const size_t byte_count = checked_rgb_bytes(width, height);
    if (!png_image_finish_read(&png, nullptr, rgba.data(), 0, nullptr)) {
        throw std::runtime_error("Unable to decode PNG " + filename + ": " + png.message);
    }
    Image image{width, height, std::vector<uint8_t>(byte_count)};
    for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
        std::copy_n(rgba.data() + pixel * 4, 3, image.rgb.data() + pixel * 3);
    }
    return image;
}

int64_t round_ties_to_even(double value) {
    if (!std::isfinite(value) || value < 0.0 ||
        value >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("Resize dimension is not representable");
    }
    const double lower_value = std::floor(value);
    const double fraction = value - lower_value;
    int64_t lower = static_cast<int64_t>(lower_value);
    if (fraction > 0.5 || (fraction == 0.5 && (lower & 1) != 0)) {
        ++lower;
    }
    return lower;
}

double cubic_filter(double value) {
    constexpr double a = -0.5;
    value = std::abs(value);
    if (value < 1.0) {
        return ((a + 2.0) * value - (a + 3.0)) * value * value + 1.0;
    }
    if (value < 2.0) {
        return (((value - 5.0) * value + 8.0) * value - 4.0) * a;
    }
    return 0.0;
}

double bilinear_filter(double value) {
    value = std::abs(value);
    return value < 1.0 ? 1.0 - value : 0.0;
}

struct FilterRow {
    size_t first;
    std::vector<int32_t> coefficients;
};

template <typename Filter>
std::vector<FilterRow> make_filter_rows(size_t input_size, size_t output_size, Filter filter,
                                        double filter_support) {
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double inverse_filter_scale = 1.0 / filter_scale;
    const double support = filter_support * filter_scale;
    std::vector<FilterRow> rows;
    rows.reserve(output_size);

    for (size_t output = 0; output < output_size; ++output) {
        const double center = (static_cast<double>(output) + 0.5) * scale;
        const int64_t first_unclamped = static_cast<int64_t>(center - support + 0.5);
        const int64_t last_unclamped = static_cast<int64_t>(center + support + 0.5);
        const size_t first = static_cast<size_t>(std::max<int64_t>(0, first_unclamped));
        const size_t last = static_cast<size_t>(
            std::min<int64_t>(static_cast<int64_t>(input_size), last_unclamped));
        FilterRow row{first, {}};
        row.coefficients.reserve(last - first);
        std::vector<double> weights;
        weights.reserve(last - first);
        double total = 0.0;
        for (size_t input = first; input < last; ++input) {
            const double weight =
                filter((static_cast<double>(input) + 0.5 - center) * inverse_filter_scale);
            weights.push_back(weight);
            total += weight;
        }
        if (total == 0.0) {
            throw std::runtime_error("Image resize generated an empty filter");
        }
        const double normalization = 1.0 / total;
        for (const double weight : weights) {
            const double scaled = weight * normalization * static_cast<double>(kCoefficientScale);
            row.coefficients.push_back(
                static_cast<int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
        rows.emplace_back(std::move(row));
    }
    return rows;
}

uint8_t apply_coefficients(const uint8_t* samples, size_t stride, const FilterRow& row) {
    int64_t sum = 0;
    for (size_t i = 0; i < row.coefficients.size(); ++i) {
        sum += static_cast<int64_t>(samples[i * stride]) * row.coefficients[i];
    }
    sum = (sum + (int64_t{1} << (kCoefficientBits - 1))) >> kCoefficientBits;
    return static_cast<uint8_t>(std::clamp<int64_t>(sum, 0, 255));
}

template <typename Filter>
Image resize_image(const Image& input, size_t output_width, size_t output_height,
                   Filter filter, double support) {
    if (input.width == output_width && input.height == output_height) {
        return input;
    }

    Image horizontal{output_width, input.height, {}};
    horizontal.rgb.resize(checked_rgb_bytes(output_width, input.height));
    if (input.width == output_width) {
        horizontal.rgb = input.rgb;
    } else {
        const auto rows = make_filter_rows(input.width, output_width, filter, support);
        for (size_t y = 0; y < input.height; ++y) {
            for (size_t x = 0; x < output_width; ++x) {
                for (size_t channel = 0; channel < 3; ++channel) {
                    const auto& row = rows[x];
                    const uint8_t* samples = input.rgb.data() +
                        (y * input.width + row.first) * 3 + channel;
                    horizontal.rgb[(y * output_width + x) * 3 + channel] =
                        apply_coefficients(samples, 3, row);
                }
            }
        }
    }

    if (input.height == output_height) {
        return horizontal;
    }
    Image output{output_width, output_height, {}};
    output.rgb.resize(checked_rgb_bytes(output_width, output_height));
    const auto rows = make_filter_rows(input.height, output_height, filter, support);
    for (size_t y = 0; y < output_height; ++y) {
        for (size_t x = 0; x < output_width; ++x) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const auto& row = rows[y];
                const uint8_t* samples = horizontal.rgb.data() +
                    (row.first * output_width + x) * 3 + channel;
                output.rgb[(y * output_width + x) * 3 + channel] =
                    apply_coefficients(samples, output_width * 3, row);
            }
        }
    }
    return output;
}

Image center_crop(const Image& input, size_t size) {
    const size_t left = (input.width - size) / 2;
    const size_t top = (input.height - size) / 2;
    Image output{size, size, {}};
    output.rgb.resize(checked_rgb_bytes(size, size));
    for (size_t y = 0; y < size; ++y) {
        const uint8_t* source = input.rgb.data() + ((top + y) * input.width + left) * 3;
        std::memcpy(output.rgb.data() + y * size * 3, source, size * 3);
    }
    return output;
}

void copy_into_letterbox(const Image& source, Image& destination, size_t left, size_t top) {
    for (size_t y = 0; y < source.height; ++y) {
        const uint8_t* src = source.rgb.data() + y * source.width * 3;
        uint8_t* dst = destination.rgb.data() + ((top + y) * destination.width + left) * 3;
        std::memcpy(dst, src, source.width * 3);
    }
}

}  // namespace

Image decode_image(const std::filesystem::path& path) {
    const std::string filename = path.string();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Unable to open image: " + filename);
    }
    std::array<unsigned char, 8> signature{};
    input.read(reinterpret_cast<char*>(signature.data()), static_cast<std::streamsize>(signature.size()));
    const std::streamsize bytes_read = input.gcount();
    input.close();

    if (bytes_read == static_cast<std::streamsize>(signature.size()) &&
        has_png_signature(signature)) {
        return decode_png(path);
    }
    if (bytes_read >= 2 && signature[0] == 0xff && signature[1] == 0xd8) {
        return decode_jpeg(path);
    }
    throw std::runtime_error("Unsupported or invalid image format: " + filename);
}

ov::Tensor preprocess_image(const std::filesystem::path& path, const std::string& family,
                            size_t size, double scale) {
    if (family != "dinov3" && family != "yolo-world" && family != "unet") {
        throw ovbench::OVBError("Unsupported image preprocessing family: " + family);
    }
    if (family != "unet" &&
        (size == 0 || size > static_cast<size_t>(std::numeric_limits<int64_t>::max()))) {
        throw ovbench::OVBError("Image preprocessing size must be positive and representable");
    }
    if (family == "unet") {
        if (!std::isfinite(scale) || scale <= 0.0) {
            throw ovbench::OVBError("UNet preprocessing scale must be finite and positive");
        }
    } else {
        checked_rgb_bytes(size, size);
    }

    Image image = decode_image(path);
    Image processed{};
    size_t output_width = size;
    size_t output_height = size;

    if (family == "dinov3") {
        const double scale = static_cast<double>(size) /
                             static_cast<double>(std::min(image.width, image.height));
        const int64_t resized_width = round_ties_to_even(image.width * scale);
        const int64_t resized_height = round_ties_to_even(image.height * scale);
        if (resized_width < static_cast<int64_t>(size) ||
            resized_height < static_cast<int64_t>(size)) {
            throw std::runtime_error("DINOv3 resize produced dimensions smaller than the crop");
        }
        const Image resized = resize_image(image, static_cast<size_t>(resized_width),
                                           static_cast<size_t>(resized_height), cubic_filter, 2.0);
        processed = center_crop(resized, size);
    } else if (family == "yolo-world") {
        const double scale = std::min(static_cast<double>(size) / image.width,
                                      static_cast<double>(size) / image.height);
        const int64_t resized_width =
            std::max<int64_t>(1, round_ties_to_even(image.width * scale));
        const int64_t resized_height =
            std::max<int64_t>(1, round_ties_to_even(image.height * scale));
        const Image resized = resize_image(image, static_cast<size_t>(resized_width),
                                           static_cast<size_t>(resized_height), bilinear_filter, 1.0);
        processed = Image{size, size, std::vector<uint8_t>(checked_rgb_bytes(size, size), 114)};
        copy_into_letterbox(resized, processed,
                            (size - resized.width) / 2, (size - resized.height) / 2);
    } else {
        output_width = std::max<size_t>(1, static_cast<size_t>(image.width * scale));
        output_height = std::max<size_t>(1, static_cast<size_t>(image.height * scale));
        processed = resize_image(image, output_width, output_height, cubic_filter, 2.0);
    }

    ov::Tensor tensor(ov::element::f32, ov::Shape{1, 3, output_height, output_width});
    float* output = tensor.data<float>();
    constexpr float means[3] = {0.485f, 0.456f, 0.406f};
    constexpr float standard_deviations[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane_size = output_width * output_height;
    for (size_t y = 0; y < output_height; ++y) {
        for (size_t x = 0; x < output_width; ++x) {
            const size_t pixel = (y * output_width + x) * 3;
            const size_t planar = y * output_width + x;
            for (size_t channel = 0; channel < 3; ++channel) {
                const float value = static_cast<float>(processed.rgb[pixel + channel]) / 255.0f;
                output[channel * plane_size + planar] = family == "dinov3"
                    ? (value - means[channel]) / standard_deviations[channel]
                    : value;
            }
        }
    }
    return tensor;
}

}  // namespace ovbench

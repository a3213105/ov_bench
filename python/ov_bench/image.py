"""Image decoding + preprocessing, ported from ``src/image.cpp``.

The resize math (cubic / bilinear fixed-point filter, center crop, letterbox)
is reproduced exactly in numpy so pixel values match the native engine. Image
*decoding* uses Pillow (JPEG and PNG) instead of libjpeg / libpng; both produce
the same sRGB RGB pixels for the supported inputs.
"""

import math
import os

import numpy as np
import openvino as ov

from .utils import OVRuntimeError, fail
from .tensor_io import from_numpy

_K_COEFFICIENT_BITS = 22
_K_COEFFICIENT_SCALE = 1 << _K_COEFFICIENT_BITS

_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)


def checked_rgb_bytes(width, height):
    if width == 0 or height == 0 or width * height > sys_max_size() // 3:
        raise OVRuntimeError("Invalid or oversized image dimensions")
    return width * height * 3


def sys_max_size():
    return 2 ** 63 - 1


def _decode_with_pillow(path):
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise OVRuntimeError("Pillow (PIL) is required for the Python image backend")
    try:
        with PILImage.open(path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            rgb = np.asarray(image, dtype=np.uint8).reshape(height, width, 3)
    except Exception as error:
        raise OVRuntimeError("Unable to decode image %s: %s" % (path, error))
    return rgb


def decode_image(path):
    """Return an HxWx3 uint8 RGB numpy array."""
    if not os.path.isfile(path):
        raise OVRuntimeError("Unable to open image: " + str(path))
    return _decode_with_pillow(path)


def round_ties_to_even(value):
    if not math.isfinite(value) or value < 0.0 or value >= 2 ** 63 - 1:
        raise OVRuntimeError("Resize dimension is not representable")
    lower_value = math.floor(value)
    fraction = value - lower_value
    lower = int(lower_value)
    if fraction > 0.5 or (fraction == 0.5 and (lower & 1) != 0):
        lower += 1
    return lower


def _cubic(distance):
    t = np.abs(distance)
    a = -0.5
    w = np.where(t < 1.0, ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0,
                 np.where(t < 2.0, (((t - 5.0) * t + 8.0) * t - 4.0) * a, 0.0))
    return w


def _bilinear(distance):
    t = np.abs(distance)
    return np.where(t < 1.0, 1.0 - t, 0.0)


def make_filter_rows(input_size, output_size, filter_fn, filter_support):
    scale = float(input_size) / float(output_size)
    filter_scale = max(scale, 1.0)
    inverse_filter_scale = 1.0 / filter_scale
    support = filter_support * filter_scale
    rows = np.empty((output_size, input_size), dtype=np.float64)
    rows[...] = 0.0
    for output in range(output_size):
        center = (output + 0.5) * scale
        first_unclamped = int(center - support + 0.5)
        last_unclamped = int(center + support + 0.5)
        first = max(0, first_unclamped)
        last = min(input_size, last_unclamped)
        if last <= first:
            continue
        inputs = np.arange(first, last, dtype=np.float64)
        weights = filter_fn((inputs + 0.5 - center) * inverse_filter_scale)
        total = weights.sum()
        if total == 0.0:
            raise OVRuntimeError("Image resize generated an empty filter")
        weights = weights / total
        scaled = weights * _K_COEFFICIENT_SCALE
        rounded = np.where(scaled < 0.0, scaled - 0.5, scaled + 0.5).round(0)
        rows[output, first:last] = rounded.astype(np.float64)
    return rows.astype(np.int64)


def _apply_rows(image, rows, axis):
    """2D separable fixed-point filter step.

    ``image`` is HxWx3 uint8. ``rows`` maps each output position to weighted
    integer coefficients along the given axis: (out_w, in_w) for horizontal and
    (out_h, in_h) for vertical.
    """
    samples = image.astype(np.int64)
    coeffs = rows.astype(np.int64)
    if axis == "horizontal":
        out = np.einsum("hic,ji->hjc", samples, coeffs)
    else:
        out = np.einsum("hwc,kh->kwc", samples, coeffs)
    out = (out + (1 << (_K_COEFFICIENT_BITS - 1))) >> _K_COEFFICIENT_BITS
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_image(image, output_width, output_height, filter_fn, support):
    if image.shape[1] == output_width and image.shape[0] == output_height:
        return image
    if image.shape[1] != output_width:
        rows = make_filter_rows(image.shape[1], output_width, filter_fn, support)
        image = _apply_rows(image, rows, "horizontal")
    if image.shape[0] != output_height:
        rows = make_filter_rows(image.shape[0], output_height, filter_fn, support)
        image = _apply_rows(image, rows, "vertical")
    return image


def center_crop(image, size):
    left = (image.shape[1] - size) // 2
    top = (image.shape[0] - size) // 2
    return image[top:top + size, left:left + size, :].copy()


def copy_into_letterbox(source, destination, left, top):
    destination[top:top + source.shape[0], left:left + source.shape[1], :] = source


def preprocess_image(path, family, size, scale=1.0):
    if family not in ("dinov3", "yolo-world", "unet"):
        fail("Unsupported image preprocessing family: " + family)
    if family != "unet" and (size == 0):
        fail("Image preprocessing size must be positive and representable")
    if family == "unet":
        if not math.isfinite(scale) or scale <= 0.0:
            fail("UNet preprocessing scale must be finite and positive")
    else:
        checked_rgb_bytes(size, size)

    image = decode_image(path)
    width = int(image.shape[1])
    height = int(image.shape[0])

    if family == "dinov3":
        scale_factor = float(size) / float(min(width, height))
        resized_width = round_ties_to_even(width * scale_factor)
        resized_height = round_ties_to_even(height * scale_factor)
        if resized_width < size or resized_height < size:
            raise OVRuntimeError("DINOv3 resize produced dimensions smaller than the crop")
        resized = resize_image(image, resized_width, resized_height, _cubic, 2.0)
        processed = center_crop(resized, size)
        output_width = output_height = size
    elif family == "yolo-world":
        scale_factor = min(float(size) / width, float(size) / height)
        resized_width = max(1, round_ties_to_even(width * scale_factor))
        resized_height = max(1, round_ties_to_even(height * scale_factor))
        resized = resize_image(image, resized_width, resized_height, _bilinear, 1.0)
        processed = np.full((size, size, 3), 114, dtype=np.uint8)
        copy_into_letterbox(resized, processed,
                            (size - resized_width) // 2, (size - resized_height) // 2)
        output_width = output_height = size
    else:
        output_width = max(1, int(width * scale))
        output_height = max(1, int(height * scale))
        processed = resize_image(image, output_width, output_height, _cubic, 2.0)

    output = np.zeros((1, 3, output_height, output_width), dtype=np.float32)
    floats = processed.astype(np.float32) / 255.0
    for channel in range(3):
        if family == "dinov3":
            output[0, channel] = ((floats[:, :, channel] - _MEANS[channel])
                                  / _STDS[channel]).astype(np.float32)
        else:
            output[0, channel] = floats[:, :, channel]
    return ov.Tensor(output)

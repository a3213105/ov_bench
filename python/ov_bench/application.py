"""Model registry, loading/validation and random-input generation.

Python port of ``include/ov_bench/application.hpp`` + ``src/application.cpp``.

Differences (documented, mirroring the package docstring):
* Random streams are generated with :mod:`numpy` (PCG64) seeded from
  ``(seed, model id)``. Distribution parameters match the C++ engine
  (float ``normal(0, 0.1)``, integer ``uniform{0, 1}``, boolean
  ``Bernoulli(0.75)``) but the exact sequence is not bit-identical to
  ``std::mt19937_64``.
* The native engine's SDPA->BF16 graph rewrite (``force_sdpa_bf16``) is
  ported one-for-one: each ScaledDotProductAttention node is replaced with a
  fresh BF16 node plus a Convert back to the original element type, using
  ``ov.opset13.convert`` / ``ov.opset13.scaled_dot_product_attention`` and the
  in-place ``node.output(0).replace(...)`` primitive (same operation sequence
  as ``src/application.cpp`` and the reference ``dinov3_benchmark.py``).
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from .tensor_io import from_numpy, tensor_info
from .utils import OVRuntimeError, fail

# element type name -> fixed-point bf16 conversion helpers live below.

F32_NAME = "f32"
BOOLEAN_NAME = "boolean"
I32_NAME = "i32"


@dataclass
class ModelSpec:
    id: str
    family: str
    artifact: str
    input_size: int
    output_count: int
    input_names: list = field(default_factory=list)
    output_names: list = field(default_factory=list)
    artifact_dir: str = ""
    artifact_file: str = ""
    preprocess_scale: float = 1.0


def _specs():
    return [
        ModelSpec("dinov3-convnext-tiny-pretrain-lvd1689m", "dinov3",
                  "dinov3-convnext-tiny-pretrain-lvd1689m", 224, 1),
        ModelSpec("dinov3-vits16-pretrain-lvd1689m", "dinov3",
                  "dinov3-vits16-pretrain-lvd1689m", 224, 1),
        ModelSpec("dinov3-vits16plus-pretrain-lvd1689m", "dinov3",
                  "dinov3-vits16plus-pretrain-lvd1689m", 224, 1),
        ModelSpec("v2.1:640", "yolo-world", "yolo_world_v2_1_m_640", 640, 3),
        ModelSpec("v2.1:1280", "yolo-world", "yolo_world_v2_1_m_1280", 1280, 3),
        ModelSpec("v2:640", "yolo-world", "yolo_world_v2_m_640", 640, 3),
        ModelSpec("v2:1280", "yolo-world", "yolo_world_v2_m_1280", 1280, 3),
        ModelSpec("unet_carvana_scale0.5_epoch2", "unet", "unet_carvana_scale0.5_epoch2",
                  0, 1, [], [], "unet_carvana_scale0_5_epoch2",
                  "unet_carvana_scale0.5_epoch2", 0.5),
        ModelSpec("unet_carvana_scale1.0_epoch2", "unet", "unet_carvana_scale1.0_epoch2",
                  0, 1, [], [], "unet_carvana_scale1_0_epoch2",
                  "unet_carvana_scale1.0_epoch2", 1.0),
        ModelSpec("26Q2_round13_run2_e40_no_map_attn", "prediction",
                  "26Q2_round13_run2_e40_no_map_attn", 0, 11,
                  ["agents_feature", "agents_mask", "map_objects_feature",
                   "map_objects_mask", "route_feature", "route_mask",
                   "lidar_patch_feature", "lidar_patch_mask",
                   "patch_cuboid_feature", "patch_cuboid_mask",
                   "lidar_context_feature", "lidar_context_mask"],
                  ["trajs", "traj_probs", "yield_prob", "covariance",
                   "traj_box_headings", "traj_speeds", "traj_accels",
                   "traj_steerings", "modals_index", "ignore_ego_intention_prob",
                   "fallen_behavior_prob"]),
        ModelSpec("ped_26Q1_history_data_gen4_800v2_issue2_e23_refine", "prediction",
                  "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine", 0, 5,
                  ["agents_feature", "agents_mask", "map_objects_feature",
                   "map_objects_mask", "skeleton_feature", "skeleton_mask",
                   "route_feature", "route_mask", "lidar_patch_feature",
                   "lidar_patch_mask", "patch_cuboid_feature", "patch_cuboid_mask"],
                  ["trajs", "traj_probs", "yield_prob", "covariance",
                   "ignore_ego_intention_prob"]),
        ModelSpec("vehicle_q2_release_v0_w_leading_25", "prediction",
                  "vehicle_q2_release_v0_w_leading_25", 0, 15,
                  ["agents_feature", "agents_mask", "map_objects_feature",
                   "map_objects_mask", "lidar_patch_feature", "lidar_patch_mask",
                   "patch_cuboid_feature", "patch_cuboid_mask"],
                  ["trajs", "traj_probs", "yield_prob", "covariance",
                   "start_to_move_probs", "start_to_move_means",
                   "start_to_move_ln_sd", "traj_box_headings", "modals_index",
                   "traj_speeds", "traj_accels", "traj_steerings", "vae_scores",
                   "anomaly_scores", "ignore_ego_intention_prob"]),
    ]


_SPEC_CACHE = None


def model_specs():
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        _SPEC_CACHE = _specs()
    return _SPEC_CACHE


def find_model(model_id):
    for spec in model_specs():
        if spec.id == model_id:
            return spec
    raise OVRuntimeError("Unsupported model: " + model_id)


def model_path(spec, root):
    root = Path(root)
    if spec.family == "generic":
        return spec.artifact
    artifact_dir = spec.artifact_dir or spec.artifact
    artifact_file = spec.artifact_file or spec.artifact
    if spec.family == "prediction":
        filename = artifact_file + ".onnx"
        primary = root / artifact_dir / filename
        if not primary.exists():
            if (root / filename).exists():
                return root / filename
            if (root / "pred_model" / filename).exists():
                return root / "pred_model" / filename
        return primary
    return root / artifact_dir / (artifact_file + ".xml")


def read_model(path):
    return ov.Core().read_model(str(path))


def _element_name(port):
    return port.get_element_type().get_type_name()


def _partial_shape(port):
    return port.get_partial_shape()


def _rank_length(partial):
    rank = partial.rank
    if rank.is_dynamic:
        return None
    return rank.get_length()


def enable_stable_topk_order(model):
    """C++ ``enable_stable_topk_order``: force stable ordering on TopK nodes."""
    count = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "TopK":
            continue
        try:
            node.set_stable(True)
            count += 1
        except Exception:
            warnings.warn("TopK stable ordering is not supported by the Python "
                          "OpenVINO API; leaving TopK defaults unchanged")
            return count
    return count


_REAL_GENERIC = {"f16", "bf16", "f32", "f64"}
_INT_GENERIC = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}
_INT_DTYPES = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
               "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64}


def _fill_generic(name, generator, size):
    if name in _REAL_GENERIC:
        raw = generator.normal(0.0, 0.1, size)
        if name == "f64":
            return raw.astype(np.float64)
        if name in ("f16", "bf16"):
            return raw.astype(np.float64).astype(np.float16)
        return raw.astype(np.float32)
    if name in _INT_GENERIC:
        return generator.integers(0, 2, size).astype(_INT_DTYPES[name])
    fail("Unsupported generic random input dtype: " + name)


def random_inputs(spec, model, generator, shapes=None):
    """Generate one synthetic input set, mirroring C++ ``random_inputs``.

    ``generator`` is a :class:`numpy.random.Generator`; ``shapes`` maps an input
    port name to an explicit shape override.
    """
    shapes = shapes or {}
    inputs = []
    for port in model.inputs:
        names = list(port.get_names())
        if not names:
            raise OVRuntimeError("Random input requires named model inputs")
        name = min(names)
        override_keys = [key for key, _ in shapes.items() if key in set(names)]
        if len(override_keys) > 1:
            fail("Multiple random shape overrides for aliases of input: " + name)
        partial = port.get_partial_shape()
        if override_keys:
            shape = list(shapes.pop(0) if False else shapes[override_keys[0]])
            if not partial.compatible(ov.PartialShape(list(shape))):
                fail("Random shape incompatible with input: " + name)
        else:
            rank_length = _rank_length(partial)
            if rank_length is None:
                fail("Random input requires explicit shape for dynamic rank: " + name)
            shape = []
            for axis in range(rank_length):
                dim = partial[axis]
                if dim.is_static:
                    shape.append(dim.get_length())
                elif axis == 0 and dim.compatible(ov.Dimension(1)):
                    shape.append(1)
                else:
                    fail("Unresolved random input dimension: " + name +
                         "; specify --input-shape NAME=DIM[,DIM...] for --model-file")
        if 0 in list(shape):
            fail("Random input has empty dimension: " + name)

        etype = _element_name(port)
        size = int(np.prod([d for d in shape])) if shape else 1
        if etype == F32_NAME:
            array = generator.normal(0.0, 0.1, size).astype(np.float32)
            tensor = from_numpy(array, "f32", shape=list(shape))
        elif etype == BOOLEAN_NAME:
            array = (generator.random(size) < 0.75)
            if spec.family == "prediction":
                group = int(shape[-1]) if shape else 1
                flat = array.view(np.uint8)
                for i in range(0, int(flat.size), group if group else 1):
                    flat[i] = 1
                    if group > 1:
                        flat[i + group - 1] = 0
            tensor = from_numpy(array, "boolean", shape=list(shape))
        elif spec.family == "generic":
            array = _fill_generic(etype, generator, size)
            tensor = from_numpy(array, etype, shape=list(shape))
        else:
            fail("Unsupported random input dtype: " + name)
        inputs.append((name, tensor))
    validate_inputs(model, inputs)
    return inputs


def load_model(spec, root):
    model = read_model(model_path(spec, root))
    if spec.family == "generic":
        if len(model.inputs) == 0 or len(model.outputs) == 0:
            fail("Generic testing requires model inputs and outputs")
        input_names = set()
        for input in model.inputs:
            for entry in input.get_names():
                input_names.add(entry)
        for index in range(len(model.inputs)):
            port = model.input(index)
            if port.get_names():
                continue
            candidate = "input_%d" % index
            while candidate in input_names:
                candidate += "_"
            try:
                port.get_tensor().set_names({candidate})
            except Exception:
                pass
            input_names.add(candidate)
        return model

    enable_stable_topk_order(model)

    if spec.family == "prediction":
        if len(spec.input_names) != len(model.inputs) or \
                len(model.outputs) != spec.output_count or \
                len(spec.output_names) != spec.output_count:
            raise OVRuntimeError("Unexpected input/output count for " + spec.id)
        for entry in spec.input_names:
            if _element_name(model.input(entry)) not in (F32_NAME, BOOLEAN_NAME):
                raise OVRuntimeError("Unsupported prediction input type for " +
                                     spec.id + ": " + entry)
        for entry in spec.output_names:
            if _element_name(model.output(entry)) not in (F32_NAME, I32_NAME):
                raise OVRuntimeError("Unsupported prediction output type for " +
                                     spec.id + ": " + entry)
        return model

    if len(model.inputs) != 1 or len(model.outputs) != spec.output_count:
        raise OVRuntimeError("Unexpected input/output count for " + spec.id)
    if _element_name(model.input(0)) != F32_NAME:
        raise OVRuntimeError("Expected f32 image input for " + spec.id)

    if spec.family == "unet":
        input_shape = _partial_shape(model.input(0))
        rank = _rank_length(input_shape)
        if rank is None or rank != 4:
            raise OVRuntimeError("UNet expects 4D NCHW input for " + spec.id)
        if input_shape[1].is_static and input_shape[1].get_length() != 3:
            raise OVRuntimeError("UNet expects 3-channel input for " + spec.id)
        if input_shape[0].is_static and input_shape[0].get_length() != 1:
            raise OVRuntimeError("UNet expects batch=1 input for " + spec.id)
        output_shape = _partial_shape(model.output(0))
        if _rank_length(output_shape) is None or _rank_length(output_shape) != 4:
            raise OVRuntimeError("UNet expects 4D output for " + spec.id)
        if _element_name(model.output(0)) != F32_NAME:
            raise OVRuntimeError("UNet expects f32 output for " + spec.id)
        if output_shape[1].is_static and output_shape[1].get_length() != 2:
            raise OVRuntimeError("UNet expects 2-class logits output for " + spec.id)
        return model

    expected = [1, 3, int(spec.input_size), int(spec.input_size)]
    if not _partial_shape(model.input(0)).compatible(ov.PartialShape(expected)):
        raise OVRuntimeError("Incompatible NCHW image shape for " + spec.id)
    model.reshape(expected)
    if spec.family == "yolo-world":
        for port in range(spec.output_count):
            spatial = int(spec.input_size // (8 << port))
            head = [1, 84, spatial, spatial]
            if not _partial_shape(model.output(port)).compatible(ov.PartialShape(head)):
                raise OVRuntimeError("YOLO-World raw head shape/order mismatch for " + spec.id)
    else:
        hidden = 768 if "convnext" in spec.id else 384
        if not _partial_shape(model.output(0)).compatible(
                ov.PartialShape([1, -1, hidden])):
            raise OVRuntimeError("DINOv3 hidden-state shape mismatch for " + spec.id)
    return model


def validate_input(model, tensor):
    port = model.input(0)
    if _element_name(port) != tensor.element_type.get_type_name() or \
            not port.get_partial_shape().compatible(
                ov.PartialShape(list(tensor.shape))):
        raise OVRuntimeError("Input tensor type/shape does not match model input")


def validate_inputs(model, tensors):
    if len(tensors) != len(model.inputs):
        raise OVRuntimeError("Input count does not match model input count")
    seen = set()
    for name, tensor in tensors:
        if name in seen:
            raise OVRuntimeError("Duplicate input name: " + name)
        seen.add(name)
        try:
            port = model.input(name)
        except Exception:
            raise OVRuntimeError("Input tensor type/shape does not match model input: " + name)
        if not port.get_partial_shape().compatible(ov.PartialShape(list(tensor.shape))) or \
                port.get_element_type().get_type_name() != tensor.element_type.get_type_name():
            raise OVRuntimeError("Input tensor type/shape does not match model input: " + name)
    for port in model.inputs:
        if not (set(port.get_names()) & seen):
            raise OVRuntimeError("Missing named model input: " + port.get_any_name())


def force_sdpa_bf16(model):
    """Rewrite every SDPA node to a BF16 subgraph, mirroring the native pass.

    For each ``ScaledDotProductAttention`` node, Q/K/V/scale are converted to
    BF16 (a boolean mask keeps its element type, exactly like the C++ pass),
    a fresh BF16 SDPA node is built, and the original node's output is
    replaced by a Convert back to the original element type.
    """
    patched = 0
    for node in list(model.get_ordered_ops()):
        if node.get_type_name() != "ScaledDotProductAttention":
            continue
        new_inputs = []
        for i in range(node.get_input_size()):
            value = node.input_value(i)
            if i == 3 and value.get_element_type() == ov.Type.boolean:
                new_inputs.append(value)
            else:
                new_inputs.append(ov.opset13.convert(value, ov.Type.bf16))
        replacement = ov.opset13.scaled_dot_product_attention(*new_inputs, False)
        restored = ov.opset13.convert(replacement, node.get_output_element_type(0))
        restored.set_friendly_name(node.get_friendly_name() + "/sdpa_bf16")
        names = [str(x) for x in node.output(0).get_names()]
        if names:
            restored.output(0).set_names(set(names))
        node.output(0).replace(restored.output(0))
        patched += 1
    if patched:
        model.validate_nodes_and_infer_types()
    return patched


def apply_default_sdpa_bf16(spec, precision, model, enabled=True):
    if not enabled or precision != "FP16" or spec.family != "dinov3" or \
            not spec.id.startswith("dinov3-vits16"):
        return 0
    count = force_sdpa_bf16(model)
    if count != 12:
        raise OVRuntimeError("Expected 12 SDPA nodes, found %d" % count)
    return count

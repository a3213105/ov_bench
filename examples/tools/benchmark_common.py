"""Shared model selection and report helpers for benchmark orchestration."""

import json


MODELS = [
    "dinov3-convnext-tiny-pretrain-lvd1689m",
    "dinov3-vits16-pretrain-lvd1689m", "dinov3-vits16plus-pretrain-lvd1689m",
    "v2.1:640", "v2.1:1280", "v2:640", "v2:1280",
]

PRED_MODELS = [
    "26Q2_round13_run2_e40_no_map_attn",
    "ped_26Q1_history_data_gen4_800v2_issue2_e23_refine",
    "vehicle_q2_release_v0_w_leading_25",
]


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def require_runtime(version):
    if version.split("-", 1)[0] != "2025.4.1":
        raise RuntimeError(f"Expected OpenVINO 2025.4.1, got {version}")

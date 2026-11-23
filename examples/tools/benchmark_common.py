"""Shared model selection and report helpers for benchmark orchestration."""

import json


MODELS = [
    "dinov3-convnext-tiny-pretrain-lvd1689m",
    "dinov3-vits16-pretrain-lvd1689m", "dinov3-vits16plus-pretrain-lvd1689m",
    "v2.1:640", "v2.1:1280", "v2:640", "v2:1280",
]

def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def require_runtime(version):
    if version.split("-", 1)[0] != "2025.4.1":
        raise RuntimeError(f"Expected OpenVINO 2025.4.1, got {version}")

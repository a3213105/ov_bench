#!/usr/bin/env bash
# Register (and optionally run) the DINOv3 image-model suite: three ViT/
# convnext IRs from the IR root, CBSH 224 image inputs. Overrides and pipeline
# options (e.g. --binary, --mode, --ir-root, --images-dir, --run-dir) are
# forwarded to tools/register_suite.sh / tools/ov_bench_all.sh.
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/register_suite.sh"
exec bash "$TOOLS" --suite dinov3 "$@"

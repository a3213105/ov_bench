#!/usr/bin/env bash
# Register (and optionally run) the UNet image-model suite: the two
# unet_carvana scale 0.5 / 1.0 IRs. Overrides and pipeline options are
# forwarded to tools/register_suite.sh / tools/ov_bench_all.sh.
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/register_suite.sh"
exec bash "$TOOLS" --suite unet "$@"

#!/usr/bin/env bash
# ov_bench SUITE-REGISTRATION entrypoint: manually register one model suite
# (family, model ID list, model root, input source) for the pipeline and either
# print the resolved registration or run every test item through
# tools/ov_bench_all.sh.
#
# The "suite" selection layer lives in:
#   scripts/parameters.sh   SUITES + per-suite model CSVs / roots (env + CLI)
#   scripts/ovbench_lib.sh  suite_models / set_model_args / set_input_args
#   src/application.cpp     static model_specs() table (model ID -> family)
# The C++ static registration is the final authority on which model IDs exist;
# this script prepares the suite selection, roots and input source only.
#
# Registration only (sourceable key="value" lines; missing paths warn only):
#   bash tools/register_suite.sh --print --suite dinov3 \
#        --models dinov3-vits16-pretrain-lvd1689m \
#        --ir-root /path/to/models_ov --images-dir /path/to/test_images
#
# Register and run (all pipeline options after the registration flags are
# forwarded verbatim to tools/ov_bench_all.sh):
#   bash examples/scripts/register_unet.sh \
#        --ir-root /path/to/models_ov --images-dir /path/to/test_images \
#        --binary build/ov_bench --mode all --run-dir reports/unet
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(dirname "$TOOLS_DIR")"
PROJECT_ROOT="$(dirname "$CPP_DIR")"

usage() {
	cat <<'EOF'
Usage: bash tools/register_suite.sh [registration] [pipeline options]

Register one model suite and either print it (--print) or run the full pipeline
(tools/ov_bench_all.sh) with it. Family model defaults and roots:
  dinov3      IR, default models ov_bench documents (3 ViT/convnext IDs)
  yolo-world  IR, default models v2:640,v2:1280,v2.1:640,v2.1:1280
  unet        IR, default models unet_carvana_scale0.5/1.0_epoch2
  pred        ONNX, default models are the three trajectory ONNX IDs

Registration options:
  --suite NAME           dinov3|yolo-world|unet|pred (required)
  --models CSV           Model IDs to register (default per family)
  --ir-root DIR          IR root for dinov3/yolo-world/unet (default: ../models_ov)
  --model-root DIR       ONNX root for pred (default: ../models)
  --images-dir PATH      JPEG/PNG input images (default: ../test_images)
  --image-manifest FILE  Prepared image reference (dinov3/yolo-world only)
  --pred-manifest FILE   Prepared prediction input/reference (pred only)
  --print                Print resolved registration as key="value" lines and exit
  -h, --help             Show this help

Every other option is forwarded to tools/ov_bench_all.sh (e.g. --binary,
--mode, --precisions, --threads, --run-dir, --dry-run).
EOF
}

SUITE=""
MODELS=""
REG_IR_ROOT="${IR_ROOT:-}"
REG_MODEL_ROOT="${MODEL_ROOT_PRED:-}"
REG_IMAGES_DIR="${IMAGES_DIR:-}"
REG_IMAGE_MANIFEST="${IMAGE_MANIFEST:-}"
REG_PRED_MANIFEST="${PRED_MANIFEST:-}"
MODELS_EXPLICIT=0
PRINT=0
FORWARD=()

while (($#)); do
	case "$1" in
		-h|--help) usage; exit 0 ;;
		--print) PRINT=1; shift; continue ;;
		--suite|--models|--ir-root|--model-root|--images-dir|--image-manifest|--pred-manifest)
			(($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
			[[ "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
			case "$1" in
				--suite) SUITE="$2" ;;
				--models) MODELS="$2"; MODELS_EXPLICIT=1 ;;
				--ir-root) REG_IR_ROOT="$2" ;;
				--model-root) REG_MODEL_ROOT="$2" ;;
				--images-dir) REG_IMAGES_DIR="$2" ;;
				--image-manifest) REG_IMAGE_MANIFEST="$2" ;;
				--pred-manifest) REG_PRED_MANIFEST="$2" ;;
			esac
			shift 2; continue ;;
	esac
	FORWARD+=("$1")
	shift
done

[[ -n "$SUITE" ]] || { echo "--suite is required" >&2; exit 2; }

# Resolve family defaults and the matching pipeline flags.
IR_ROOT_DEFAULT="$PROJECT_ROOT/models_ov"
MODEL_ROOT_PRED_DEFAULT="$PROJECT_ROOT/models"
IMAGES_DIR_DEFAULT="$PROJECT_ROOT/test_images"
case "$SUITE" in
	dinov3)
		MODELS="${MODELS:-dinov3-convnext-tiny-pretrain-lvd1689m,dinov3-vits16-pretrain-lvd1689m,dinov3-vits16plus-pretrain-lvd1689m}"
		MODEL_FLAG="--dinov3-models"; FAMILY_VAR="DINO_MODELS"
		IR_ROOT="${REG_IR_ROOT:-$IR_ROOT_DEFAULT}"
		;;
	yolo-world)
		MODELS="${MODELS:-v2.1:640,v2.1:1280,v2:640,v2:1280}"
		MODEL_FLAG="--yolo-world-models"; FAMILY_VAR="YOLO_MODELS"
		IR_ROOT="${REG_IR_ROOT:-$IR_ROOT_DEFAULT}"
		;;
	unet)
		MODELS="${MODELS:-unet_carvana_scale0.5_epoch2,unet_carvana_scale1.0_epoch2}"
		MODEL_FLAG="--unet-models"; FAMILY_VAR="UNET_MODELS"
		IR_ROOT="${REG_IR_ROOT:-$IR_ROOT_DEFAULT}"
		;;
	pred)
		MODELS="${MODELS:-26Q2_round13_run2_e40_no_map_attn,ped_26Q1_history_data_gen4_800v2_issue2_e23_refine,vehicle_q2_release_v0_w_leading_25}"
		MODEL_FLAG="--pred-models"; FAMILY_VAR="PRED_MODELS"
		MODEL_ROOT_PRED="${REG_MODEL_ROOT:-$MODEL_ROOT_PRED_DEFAULT}"
		;;
	*) echo "Unknown suite: $SUITE (expected dinov3|yolo-world|unet|pred)" >&2; exit 2 ;;
esac
[[ -n "$MODELS" ]] || { echo "--models must not be empty" >&2; exit 2; }
IMAGES_DIR="${REG_IMAGES_DIR:-$IMAGES_DIR_DEFAULT}"

# Consistency of the registration with the chosen family.
check_path() {
	local what="$1" path="$2"
	[[ -z "$path" ]] && return 0
	if [[ -e "$path" ]]; then return 0; fi
	if (( PRINT == 1 )); then
		echo "WARNING: $what not found: $path" >&2
	else
		echo "$what not found: $path" >&2
		exit 1
	fi
}
if [[ "$SUITE" == "pred" ]]; then
	check_path "model root" "$MODEL_ROOT_PRED"
else
	check_path "IR root" "$IR_ROOT"
fi
check_path "images directory" "$IMAGES_DIR"
check_path "image manifest" "$REG_IMAGE_MANIFEST"
check_path "prediction manifest" "$REG_PRED_MANIFEST"
if [[ "$SUITE" == "dinov3" || "$SUITE" == "yolo-world" || "$SUITE" == "unet" ]]; then
	[[ -z "$REG_PRED_MANIFEST" ]] || { echo "--pred-manifest requires --suite pred" >&2; exit 2; }
else
	[[ -z "$REG_IMAGE_MANIFEST" ]] || { echo "--image-manifest requires dinov3 or yolo-world" >&2; exit 2; }
fi

registration_lines() {
	printf 'SUITES="%s"\n' "$SUITE"
	printf '%s="%s"\n' "$FAMILY_VAR" "$MODELS"
	if [[ "$SUITE" == "pred" ]]; then
		printf 'MODEL_ROOT_PRED="%s"\n' "$MODEL_ROOT_PRED"
	else
		printf 'IR_ROOT="%s"\n' "$IR_ROOT"
	fi
	printf 'IMAGES_DIR="%s"\n' "$IMAGES_DIR"
	[[ -n "$REG_IMAGE_MANIFEST" ]] && printf 'IMAGE_MANIFEST="%s"\n' "$REG_IMAGE_MANIFEST"
	[[ -n "$REG_PRED_MANIFEST" ]] && printf 'PRED_MANIFEST="%s"\n' "$REG_PRED_MANIFEST"
	return 0
}

if (( PRINT == 1 )); then
	registration_lines
	exit 0
fi

pipeline_args=( --suites "$SUITE" "$MODEL_FLAG" "$MODELS"
	--images-dir "$IMAGES_DIR" )
if [[ "$SUITE" == "pred" ]]; then
	pipeline_args+=( --model-root "$MODEL_ROOT_PRED" )
	[[ -n "$REG_PRED_MANIFEST" ]] && pipeline_args+=( --pred-manifest "$REG_PRED_MANIFEST" )
else
	pipeline_args+=( --ir-root "$IR_ROOT" )
	[[ -n "$REG_IMAGE_MANIFEST" ]] && pipeline_args+=( --image-manifest "$REG_IMAGE_MANIFEST" )
fi
set -- "${pipeline_args[@]}" ${FORWARD[@]+"${FORWARD[@]}"}

exec bash "$TOOLS_DIR/ov_bench_all.sh" "$@"

#!/usr/bin/env bash
# Wrapper mỏng: thiết lập môi trường + cùng CLI với scripts/prepare_data.py.
# Truyền thêm cờ cho prepare_data sau --. Ví dụ:
#   bash scripts/run_pipeline.sh -- --train-end 2020-02-29 --log-level DEBUG

set -euo pipefail

section() {
    echo ""
    echo "  $*"
}

# Giá trị mặc định khớp prepare_data.py khi không truyền cờ (YAML điền train/val/min/filter).
CSV_GLOB="data/raw/*.csv"
SPARK_CONFIG=""
DATA_DIR=""
STRUCT_DIR=""
GRAPH_DIR=""
TARGET_BEHAVIOR="purchase"
MIN_USER_PURCHASES=""
MIN_ITEM_PURCHASES=""
FILTER_ROUNDS=""
TRAIN_END=""
VAL_END=""
LOG_LEVEL="INFO"
CONDA_ENV="recsys_env"
EXTRA_PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            EXTRA_PY_ARGS=("$@")
            break
            ;;
        --csv-glob)              CSV_GLOB="$2";              shift 2 ;;
        --spark-config)          SPARK_CONFIG="$2";          shift 2 ;;
        --data-dir)              DATA_DIR="$2";              shift 2 ;;
        --struct-dir)            STRUCT_DIR="$2";            shift 2 ;;
        --graph-dir)             GRAPH_DIR="$2";             shift 2 ;;
        --target-behavior)       TARGET_BEHAVIOR="$2";       shift 2 ;;
        --min-user-purchases)    MIN_USER_PURCHASES="$2";    shift 2 ;;
        --min-item-purchases)    MIN_ITEM_PURCHASES="$2";    shift 2 ;;
        --filter-rounds)         FILTER_ROUNDS="$2";         shift 2 ;;
        --train-end)             TRAIN_END="$2";             shift 2 ;;
        --val-end)               VAL_END="$2";               shift 2 ;;
        --log-level)             LOG_LEVEL="$2";             shift 2 ;;
        --conda-env)             CONDA_ENV="$2";             shift 2 ;;
        *)
            echo "Đối số không nhận dạng: $1 (dùng -- để chuyển cờ vào prepare_data.py)" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PIPELINE_START=$(date +%s)

section "REES46 BPATMP — Chuẩn bị dữ liệu"
echo "Project root           : ${PROJECT_ROOT}"
echo "Glob CSV               : ${CSV_GLOB}"
echo "Spark config           : ${SPARK_CONFIG}"
echo "Thư mục data           : ${DATA_DIR}"
echo "Thư mục struct         : ${STRUCT_DIR}"
echo "Thư mục graph          : ${GRAPH_DIR}"
echo "Hành vi mục tiêu       : ${TARGET_BEHAVIOR}"
echo "Tối thiểu mua/người    : ${MIN_USER_PURCHASES}"
echo "Tối thiểu mua/sp       : ${MIN_ITEM_PURCHASES}"
echo "Số vòng lọc            : ${FILTER_ROUNDS}"
echo "Kết thúc train         : ${TRAIN_END}"
echo "Kết thúc val           : ${VAL_END}"
echo "Log level              : ${LOG_LEVEL}"
echo "Bắt đầu lúc            : $(date '+%Y-%m-%d %H:%M:%S')"

section "Bước 1/4 — Java (PySpark)"

if ! command -v java &>/dev/null; then
    echo "Không tìm thấy Java trên PATH. PySpark cần Java 8 hoặc 11." >&2
    echo "Cài: sudo apt-get install -y openjdk-11-jdk" >&2
    echo "Rồi: export JAVA_HOME=\$(readlink -f /usr/bin/java | sed 's|/bin/java||')" >&2
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -1)
echo "Đã tìm thấy Java: ${JAVA_VERSION}"

section "Bước 2/4 — Conda (${CONDA_ENV})"

CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "Không tìm thấy conda trên PATH. Cài Miniconda/Anaconda và thử lại." >&2
    exit 1
}

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Đã kích hoạt môi trường conda."
echo "Python : $(python --version)"
echo "PySpark: $(python -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null || echo 'chưa cài — pip install pyspark')"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'chưa cài')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

section "Bước 3/4 — Runtime (PYTHONPATH, thư mục tạm Spark)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/data/spark-temp"
mkdir -p "${PROJECT_ROOT}/data/spark-checkpoints"

echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "PYTHONPATH đã gồm thư mục gốc project."
echo "Thư mục tạm Spark: data/spark-temp  data/spark-checkpoints"

section "Bước 4/4 — python scripts/prepare_data.py"

PREPARE_ARGS=(
    --csv-glob         "${CSV_GLOB}"
    --target-behavior  "${TARGET_BEHAVIOR}"
    --log-level        "${LOG_LEVEL}"
)

[[ -n "$SPARK_CONFIG"        ]] && PREPARE_ARGS+=(--spark-config        "$SPARK_CONFIG")
[[ -n "$DATA_DIR"            ]] && PREPARE_ARGS+=(--data-dir            "$DATA_DIR")
[[ -n "$STRUCT_DIR"          ]] && PREPARE_ARGS+=(--struct-dir          "$STRUCT_DIR")
[[ -n "$GRAPH_DIR"           ]] && PREPARE_ARGS+=(--graph-dir           "$GRAPH_DIR")
[[ -n "$MIN_USER_PURCHASES"  ]] && PREPARE_ARGS+=(--min-user-purchases  "$MIN_USER_PURCHASES")
[[ -n "$MIN_ITEM_PURCHASES"  ]] && PREPARE_ARGS+=(--min-item-purchases  "$MIN_ITEM_PURCHASES")
[[ -n "$FILTER_ROUNDS"       ]] && PREPARE_ARGS+=(--filter-rounds       "$FILTER_ROUNDS")
[[ -n "$TRAIN_END"           ]] && PREPARE_ARGS+=(--train-end           "$TRAIN_END")
[[ -n "$VAL_END"             ]] && PREPARE_ARGS+=(--val-end             "$VAL_END")

python scripts/prepare_data.py "${PREPARE_ARGS[@]}" "${EXTRA_PY_ARGS[@]}"

PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

EFFECTIVE_DATA_DIR="${DATA_DIR:-"data/processed/temporal"}"
EFFECTIVE_STRUCT_DIR="${STRUCT_DIR:-"data/processed/temporal/node_mappings"}"
EFFECTIVE_GRAPH_DIR="${GRAPH_DIR:-"data/processed/temporal/graph"}"

section "Hoàn tất pipeline"
echo "Tổng thời gian : $(( ELAPSED / 60 )) phút $(( ELAPSED % 60 )) giây"
echo "Kết thúc lúc   : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Artefact:"
echo "  ${EFFECTIVE_DATA_DIR}/   — cạnh train .npy, mask, node_counts.json, candidate_item_idx.npy"
echo "  ${EFFECTIVE_GRAPH_DIR}/  — val/test parquet, train_events, item_metadata"
echo "  ${EFFECTIVE_STRUCT_DIR}/ — parquet cấu trúc, *2idx.json"
echo ""
echo "Huấn luyện:"
echo "    python -m src.training.trainer --config config/training.yaml"

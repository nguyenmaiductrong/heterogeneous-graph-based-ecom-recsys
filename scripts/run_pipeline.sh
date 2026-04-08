set -euo pipefail

if [ -t 1 ]; then
    C_RESET='\033[0m'; C_BOLD='\033[1m'; C_GREEN='\033[0;32m'
    C_CYAN='\033[0;36m'; C_YELLOW='\033[0;33m'; C_RED='\033[0;31m'
else
    C_RESET=''; C_BOLD=''; C_GREEN=''; C_CYAN=''; C_YELLOW=''; C_RED=''
fi

log_info()    { echo -e "${C_CYAN}[INFO ]${C_RESET}  $*"; }
log_ok()      { echo -e "${C_GREEN}[  OK ]${C_RESET}  $*"; }
log_warn()    { echo -e "${C_YELLOW}[ WARN]${C_RESET}  $*"; }
log_error()   { echo -e "${C_RED}[ERROR]${C_RESET}  $*" >&2; }
log_section() {
    echo -e "\n${C_BOLD}${C_CYAN}══════════════════════════════════════════════════════════${C_RESET}"
    echo -e "${C_BOLD}${C_CYAN}  $*${C_RESET}"
    echo -e "${C_BOLD}${C_CYAN}══════════════════════════════════════════════════════════${C_RESET}"
}


CSV_GLOB="data/raw/*.csv"
DATA_DIR=""
STRUCT_DIR=""
TARGET_BEHAVIOR="purchase"
MIN_INTERACTIONS=""       
MIN_ITEM_INTERACTIONS=""  
LOG_LEVEL="INFO"
CONDA_ENV="recsys_env"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv-glob)               CSV_GLOB="$2";               shift 2 ;;
        --data-dir)               DATA_DIR="$2";               shift 2 ;;
        --struct-dir)             STRUCT_DIR="$2";             shift 2 ;;
        --target-behavior)        TARGET_BEHAVIOR="$2";        shift 2 ;;
        --min-interactions)       MIN_INTERACTIONS="$2";       shift 2 ;;
        --min-item-interactions)  MIN_ITEM_INTERACTIONS="$2";  shift 2 ;;
        --log-level)              LOG_LEVEL="$2";              shift 2 ;;
        --conda-env)              CONDA_ENV="$2";              shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PIPELINE_START=$(date +%s)

log_section "REES46 BAGNN — Data Preparation Pipeline  [Spark + Pandas hybrid]"
log_info "Project root          : ${PROJECT_ROOT}"
log_info "CSV glob              : ${CSV_GLOB}"
log_info "Data dir              : ${DATA_DIR:-"(auto: data/processed/temporal)"}"
log_info "Struct dir            : ${STRUCT_DIR:-"(auto: data/processed/temporal/node_mappings)"}"
log_info "Target behavior       : ${TARGET_BEHAVIOR}"
log_info "Min interactions      : ${MIN_INTERACTIONS:-"(from spark_config.yaml)"}"
log_info "Min item interactions : ${MIN_ITEM_INTERACTIONS:-"(from spark_config.yaml)"}"
log_info "Log level             : ${LOG_LEVEL}"
log_info "Start time            : $(date '+%Y-%m-%d %H:%M:%S')"

log_section "Step 1/4 — Validating Java runtime (required by PySpark)"

if ! command -v java &>/dev/null; then
    log_error "Java not found on PATH. PySpark requires Java 8 or 11."
    log_error "Install with: sudo apt-get install -y openjdk-11-jdk"
    log_error "Then set: export JAVA_HOME=\$(readlink -f /usr/bin/java | sed 's|/bin/java||')"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -1)
log_ok "Java found: ${JAVA_VERSION}"

log_section "Step 2/4 — Activating Conda environment '${CONDA_ENV}'"

CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    log_error "conda not found on PATH. Install Miniconda/Anaconda and retry."
    exit 1
}

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

log_ok "Conda environment activated."
log_info "Python : $(python --version)"
log_info "PySpark: $(python -c 'import pyspark; print(pyspark.__version__)' 2>/dev/null || echo 'not installed — run: pip install pyspark')"
log_info "Torch  : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
log_info "CUDA   : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"


log_section "Step 3/4 — Configuring runtime environment"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/data/spark-temp"
mkdir -p "${PROJECT_ROOT}/data/spark-checkpoints"

log_ok "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
log_ok "PYTHONPATH includes project root."
log_ok "Spark temp dirs created: data/spark-temp  data/spark-checkpoints"

log_section "Step 4/4 — Running prepare_data.py  [Global Temporal Split]"

PREPARE_ARGS=(
    --csv-glob         "${CSV_GLOB}"
    --target-behavior  "${TARGET_BEHAVIOR}"
    --log-level        "${LOG_LEVEL}"
)

[[ -n "$DATA_DIR"              ]] && PREPARE_ARGS+=(--data-dir              "$DATA_DIR")
[[ -n "$STRUCT_DIR"            ]] && PREPARE_ARGS+=(--struct-dir            "$STRUCT_DIR")
[[ -n "$MIN_INTERACTIONS"      ]] && PREPARE_ARGS+=(--min-interactions      "$MIN_INTERACTIONS")
[[ -n "$MIN_ITEM_INTERACTIONS" ]] && PREPARE_ARGS+=(--min-item-interactions "$MIN_ITEM_INTERACTIONS")

python scripts/prepare_data.py "${PREPARE_ARGS[@]}"

PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))

EFFECTIVE_DATA_DIR="${DATA_DIR:-"data/processed/temporal"}"
EFFECTIVE_STRUCT_DIR="${STRUCT_DIR:-"data/processed/temporal/node_mappings"}"

log_section "Pipeline Complete  [Global Temporal Split]"
log_ok "Total time : $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
log_ok "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
log_info "Artifacts written:"
log_info "  ${EFFECTIVE_DATA_DIR}/"
log_info "    purchase_train_{src,dst}.npy       — purchase training edges"
log_info "    view_train_{src,dst}.npy           — view auxiliary edges (leakage-filtered)"
log_info "    cart_train_{src,dst}.npy           — cart auxiliary edges (leakage-filtered)"
log_info "    test_{user,product}_idx.npy        — held-out test pairs"
log_info "    val_{user,product}_idx.npy         — held-out val pairs"
log_info "    train_mask.pkl                     — per-user seen-item sets (eval masking)"
log_info "    node_counts.json                   — authoritative vocab sizes"
log_info "  ${EFFECTIVE_STRUCT_DIR}/"
log_info "    product_category.parquet           — structural edges (product->category)"
log_info "    product_brand.parquet              — structural edges (product->brand)"
log_info "    {user,item,category,brand}2idx.json"
echo ""
log_ok "Ready for training. Run:"
echo -e "    ${C_BOLD}python -m src.training.trainer --config config/training.yaml${C_RESET}"

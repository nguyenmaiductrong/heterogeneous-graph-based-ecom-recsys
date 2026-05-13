# Dự đoán tương tác khách hàng – sản phẩm thông qua học biểu diễn đồ thị không đồng nhất trên tập dữ liệu thương mại điện tử quy mô lớn REES46

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c.svg)](https://pytorch.org/)
[![PySpark 3.5](https://img.shields.io/badge/PySpark-3.5.1-E25A1C.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Phân tích tập dữ liệu](#2-phân-tích-tập-dữ-liệu)
3. [Pipeline xử lý dữ liệu](#3-pipeline-xử-lý-dữ-liệu)
4. [Kiến trúc mô hình](#4-kiến-trúc-mô-hình)
5. [Cấu trúc dự án](#5-cấu-trúc-dự-án)
6. [Cài đặt](#6-cài-đặt)
7. [Huấn luyện](#7-huấn-luyện)
8. [Đánh giá](#8-đánh-giá)
9. [Trích dẫn](#9-trích-dẫn)

---

## 1. Giới thiệu

Kho mã nguồn này cài đặt mô hình **BPATMP (Behavior-aware Personalized Attention with Temporal Message Passing)** cho bài toán gợi ý trong thương mại điện tử quy mô lớn. Mô hình hoạt động trên **Mạng thông tin không đồng nhất (HIN)** được xây dựng từ tập dữ liệu công khai REES46, gồm 4 loại nút (User, Product, Category, Brand) và 10 loại quan hệ (ba hành vi người dùng + cạnh cấu trúc + cạnh ngược tương ứng).

**Các đóng góp kỹ thuật chính:**

- **Pipeline xử lý dữ liệu dựa trên PySpark**: chuyển hóa hơn 200 triệu bản ghi thô thành đồ thị không đồng nhất có cấu trúc, hỗ trợ giao thức Temporal Split.
- **Low-rank Behavior-aware Weight Decomposition**: `W(φ, β) = W_base + A_φ @ B_β^T` — đặc thù hóa bước truyền thông điệp theo cặp (Node Type, Behavior Type) mà không làm tăng đột biến số tham số.
- **Temporal Attention with Decay-in-logit**: chú ý 4 thành phần kết hợp content match, relation bias, time encoding Fourier và per-behavior time-decay, phân biệt tín hiệu view/cart/purchase theo thời gian.
- **IntentCodebook**: phân rã embedding người dùng thành tổ hợp `n_intents` prototype vectors, cung cấp biểu diễn nhận thức ý định cho từng hành vi.
- **HierarchicalMBCL**: InfoNCE phân cấp hướng (view -> cart -> purchase) với hard-negative mining trong batch, tăng khả năng phân biệt embedding giữa các mức độ hành vi.
- **BPATMPTotalLoss**: hàm mất mát tổng hợp gồm BPR đa hành vi có trọng số, contrastive learning, funnel prior và monotonic decay prior.
- **Mixed-strategy Negative Sampling**: kết hợp uniform, popularity-biased và in-batch hard negatives với lịch sử toàn cục để loại trừ các item đã tương tác.

---

## 2. Phân tích tập dữ liệu

Tập dữ liệu **REES46** là nhật ký sự kiện thương mại điện tử từ tháng **10/2019 đến 04/2020**. Sau tiền xử lý (lọc nhiễu, loại người dùng/sản phẩm ít tương tác), dữ liệu được cấu trúc thành đồ thị không đồng nhất.

### 2.1 Thống kê nút

| Loại nút | Ký hiệu | Số lượng |
|---|---|---|
| User | \|U\| | 203.063 |
| Product | \|I\| | 29.892 |
| Category | \|C\| | 14 |
| Brand | \|B\| | 1.919 |
| **Tổng cộng** | | **234.888** |

### 2.2 Thống kê cạnh (tập train)

| Quan hệ | Loại | Số cạnh | Độ thưa |
|---|---|---|---|
| user -> [view] -> product | Behavioral | 25.956.733 | 0,99573 |
| user -> [cart] -> product | Behavioral | 3.868.050 | 0,99936 |
| user -> [purchase] -> product | Behavioral | 2.436.760 | 0,99960 |
| product -> [belongs_to] -> category | Structural | 29.892 | — |
| product -> [producedBy] -> brand | Structural | 29.892 | — |
| **Tổng cạnh hành vi** | | **32.261.543** | |

Độ thưa: `1 - |E| / (|U| × |I|)`.

### 2.3 Đặc điểm nổi bật

| Thuộc tính | Giá trị |
|---|---|
| Mật độ tương tác (mua hàng) | 4,01 × 10⁻⁴ |
| Tỉ lệ view : cart : purchase | 10,6 : 1,6 : 1,0 |
| Người dùng val (unique) | 47.996 — 114.475 positives |
| Người dùng test (unique) | 23.536 — 42.950 positives |

> **EDA:** [`notebooks/01_graph_eda.ipynb`](notebooks/01_graph_eda.ipynb) — phân phối bậc nút, mất cân bằng hành vi, phân phối theo thời gian.

---

## 3. Pipeline xử lý dữ liệu

Pipeline chuyển file CSV thô thành đồ thị không đồng nhất có cấu trúc. Điểm vào là [`scripts/run_pipeline.sh`](scripts/run_pipeline.sh), gọi [`scripts/prepare_data.py`](scripts/prepare_data.py), logic Spark nằm ở [`src/data_pipeline/`](src/data_pipeline/).

![Data Pipeline Diagram](assets/data-pipeline.svg)

### 3.1 Giao thức đánh giá — Temporal Split

Ranh giới thời gian toàn cục: train đến 29/02/2020, val = tháng 3/2020, test = tháng 4/2020. Phù hợp với môi trường triển khai thực tế, tránh rò rỉ thông tin tương lai.

### 3.2 Cấu trúc đầu ra

Sau khi chạy pipeline, thư mục `data/processed/temporal/` có cấu trúc:

```
data/processed/temporal/
├── view_train_{src,dst,ts}.npy
├── cart_train_{src,dst,ts}.npy
├── purchase_train_{src,dst,ts}.npy
├── val_ground_truth.pkl
├── test_ground_truth.pkl
├── train_mask_purchase_only.pkl
├── train_mask.pkl
├── node_counts.json
│
├── node_mappings/
│   ├── product_category.parquet
│   ├── product_brand.parquet
│   └── *2idx.json
│
└── graph/
    ├── train_events.parquet
    ├── val_ground_truth.parquet
    ├── test_ground_truth.parquet
    └── item_metadata.parquet
```

---

## 4. Kiến trúc mô hình

### 4.1 Lược đồ đồ thị không đồng nhất

Đồ thị gồm **10 loại quan hệ** (5 hướng ngữ nghĩa + 5 cạnh ngược):

```
user  --[view]-------->  product  --[belongs_to]-->  category
user  --[cart]-------->  product  --[producedBy]-->  brand
user  --[purchase]---->  product
product  --[rev_view / rev_cart / rev_purchase]-->  user
category  --[contains]-->  product
brand  --[brands]------>  product
```

### 4.2 Các thành phần mô hình

#### BehaviorAwareWeight — Low-rank Weight Decomposition

Ma trận trọng số cho mỗi bước truyền thông điệp:

```
W(φ, β) = W_base  +  A_φ @ B_β^T

  W_base : ma trận cơ sở chung          (d × d)
  A_φ    : nhân tố đặc thù theo loại nút (d × r),  r = rank
  B_β    : nhân tố đặc thù theo hành vi  (d × r)
```

`φ` là loại nút nguồn, `β ∈ {view, cart, purchase}`. Cạnh cấu trúc chỉ dùng `W_base`.

#### TemporalAttention — Chú ý thời gian với decay-in-logit

```
logit_ij = Q·K/√d + b_ρ + u_{ρ,β}ᵀ Φ(Δt) − λ_β · log(1 + Δt/τ)
α_ij     = scatter_softmax(logit_ij, dst_idx)
gate_ij  = σ(c_{ρ,β} + r_{ρ,β}ᵀ Φ(Δt) − μ_β · log(1 + Δt/τ))
h_t^out  = LayerNorm( Σ_j α_ij · gate_ij · m_j ) + h_t^in
```

`Φ(Δt)` là Fourier time encoding, `λ_β` và `μ_β` là decay rate per behavior được học.

#### IntentCodebook — Phân rã ý định người dùng

Embedding người dùng được chiếu lên `n_intents` prototype vectors qua soft-assignment:

```
z_u = Σ_k  softmax(h_u · C_k / τ)_k  · C_k
```

Cung cấp biểu diễn per-behavior cho `HierarchicalMBCL`.

#### HierarchicalMBCL — Contrastive Learning phân cấp

InfoNCE hướng (weak -> strong) với hard-negative mining:

```
L_CL = Σ_{(weak, strong, w)}  w · InfoNCE(z_weak.detach(), z_strong)
Cặp mặc định: (view-> cart, 0.6), (cart-> purchase, 1.0), (view-> purchase, 1.0)
```

### 4.3 Hàm mất mát tổng hợp

```
L = L_BPR + λ_cl · L_CL + λ_conv · L_conv + λ_mono · L_mono + λ_wd · ||θ||²

  L_BPR  : BPR đa hành vi, trọng số w_b = clip((N_purchase/N_b)^alpha, w_min, 1.0)
  L_CL   : HierarchicalMBCL (λ_cl = 0.15)
  L_conv : Funnel prior — s_view < s_cart < s_purchase (λ_conv = 0.10)
  L_mono : Monotonic decay — λ_view ≥ λ_cart ≥ λ_purchase (λ_mono = 0.05)
```

### 4.4 Siêu tham số

| Tham số | Giá trị |
|---|---|
| Embedding dimension (`embed_dim`) | 256 |
| GNN layers (`n_layers`) | 3 |
| Low-rank factor (`rank`) | 64 |
| Intent prototypes (`n_intents`) | 64 |
| Fourier frequencies (`n_freqs`) | 16 |
| Batch size | 8192 |
| Negative samples per query (`num_neg`) | 32 |
| Neighbor budget hop-1 / hop-2 | 16 / 8 |
| Learning rate | 8e-4 |
| Epochs (max) | 40 |
| Early stopping patience | 10 |
| CL every k steps (`cl_every_k`) | 2 |
| Metrics | HR@1/5/10/20/50, NDCG@1/5/10/20/50 |
| Primary metric | NDCG@20 |

---

## 5. Cấu trúc dự án

```
heterogeneous-graph-based-ecom-recsys/
├── config/
│   ├── training.yaml           # Toàn bộ siêu tham số: model, sampler, loss, training, wandb
│   └── spark_config.yaml       # Cấu hình PySpark pipeline
│
├── scripts/
│   ├── run_training.py         # Điểm vào chính: nạp dữ liệu -> build graph -> train
│   ├── prepare_data.py         # Chạy PySpark pipeline xử lý dữ liệu
│   ├── download_data.py        # Tải file CSV từ Kaggle
│   ├── evaluate.py             # Đánh giá checkpoint trên val/test
│   ├── train.py                # (placeholder)
│   └── run_pipeline.sh         # Shell script thiết lập môi trường + gọi prepare_data.py
│
├── src/
│   ├── core/
│   │   ├── contracts.py        # Dataclass I/O: EvalInput, BEHAVIOR_TYPES, EMBED_DIM
│   │   └── evaluator.py        # TemporalSplitEvaluator: HR@K, NDCG@K full-ranking tiled
│   │
│   ├── data_pipeline/
│   │   ├── spark_utils.py      # SparkSession, schema REES46
│   │   ├── extract.py          # Đọc CSV, làm sạch
│   │   ├── transform.py        # Lọc, chia tập, xây cạnh, mask
│   │   ├── splitter.py         # Temporal split logic
│   │   ├── load.py             # Ghi Parquet / npy
│   │   └── sanity.py           # Kiểm tra tính hợp lệ đồ thị và eval mask
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── neighbor_sampler.py # BehaviorAwareNeighborSampler: 2-hop vectorized CSR sampling
│   │
│   ├── model/
│   │   └── bpatmp.py           # BehaviorAwareWeight, TemporalAttention, BPATMPConv,
│   │                           # BPATMPLayer, IntentCodebook, BehaviorNormalizedAgg,
│   │                           # BPATMPModel
│   │
│   └── training/
│       ├── trainer.py          # TrainConfig, train(), eval_epoch(), export_embeddings()
│       ├── losses.py           # BPATMPTotalLoss, MultiTaskBPRLoss, HierarchicalMBCL,
│       │                       # FunnelPriorLoss, MonotonicDecayPriorLoss,
│       │                       # sample_aligned_negatives_local, build_user_history_csr
│       └── checkpoint_manager.py  # Lưu / nạp checkpoint, W&B artifact upload
│
├── notebooks/
│   ├── 01_graph_eda.ipynb      # EDA: phân phối bậc nút, hành vi, thời gian
│   ├── 03_graph_sage.ipynb
│   └── 04_ml_fraction_matrix.ipynb
│
├── requirements.txt
└── pyproject.toml
```

---

## 6. Cài đặt

### 6.1 Yêu cầu hệ thống

- Python 3.11
- Java 11+ (bắt buộc cho PySpark)
- GPU CUDA (khuyến nghị; CPU-only được hỗ trợ)

### 6.2 Thiết lập môi trường

```bash
git clone https://github.com/nguyenmaiductrong/heterogeneous-graph-based-ecom-recsys
cd heterogeneous-graph-based-ecom-recsys

python3.11 -m venv rees46_env
source rees46_env/bin/activate   # Windows: rees46_env\Scripts\activate

pip install -r requirements.txt

# PyTorch Geometric (điều chỉnh phiên bản CUDA cho phù hợp)
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 6.3 Chuẩn bị dữ liệu

1. Tải file CSV REES46 từ [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
2. Cập nhật đường dẫn trong `config/spark_config.yaml`.
3. Chạy pipeline:

```bash
bash scripts/run_pipeline.sh
```

**Thời gian ước tính:** ~120 phút trên máy 64 GB RAM, 8 CPU cores (PySpark local mode).

---

## 7. Huấn luyện

### 7.1 Cấu hình

Mọi siêu tham số nằm trong `config/training.yaml`. Các trường quan trọng:

```yaml
data:
  data_dir: /content/data/           # thư mục chứa .npy, .pkl, node_counts.json
  struct_dir: /content/data/node_mappings  # product_category.parquet, product_brand.parquet

model:
  embed_dim: 256
  n_layers: 3
  rank: 64
  n_intents: 64
  dropout: 0.2

training:
  device: cuda
  epochs: 40
  batch_size: 8192
  lr: 8.0e-04
  patience: 10
  save_dir: checkpoints-v7

wandb:
  enabled: true
  project: bpatmp-recsys
  run_name: bpatmp-v7
```

### 7.2 Chạy huấn luyện

```bash
python scripts/run_training.py --config config/training.yaml
```

**Luồng thực thi:**

1. Nạp dữ liệu từ `data_dir` (file `.npy` hành vi + cấu trúc Parquet).
2. Xây `HeteroData` với 10 loại quan hệ.
3. Khởi tạo `BPATMPModel` và `BehaviorAwareNeighborSampler`.
4. Chạy vòng lặp train với AMP (`bf16`), gradient clipping, cosine LR schedule.
5. Đánh giá trên val set mỗi epoch (full-ranking, `eval_subsample = 20000` user).
6. Lưu checkpoint tốt nhất vào `checkpoints-v7/best.pt` và upload W&B artifact.

**Checkpoint được lưu tại:**

```
checkpoints-v7/
├── best.pt          # model tốt nhất theo NDCG@20 trên val
├── latest.pt        # checkpoint epoch gần nhất
└── eval_val.json    # metrics của best checkpoint
```

### 7.3 Theo dõi quá trình

Nếu `wandb.enabled: true`, metrics được log lên W&B project `bpatmp-recsys` theo từng epoch:

```
loss/total, loss/bpr, loss/cl, loss/conv, loss/mono, loss/wd
loss/view, loss/cart, loss/purchase
HR@1, HR@5, HR@10, HR@20, HR@50, NDCG@1, NDCG@5, NDCG@10, NDCG@20, NDCG@50
```

Để tắt W&B:

```yaml
wandb:
  enabled: false
```

---

## 8. Đánh giá

### 8.1 Đánh giá từ checkpoint cục bộ

```bash
# Đánh giá trên val set
python scripts/evaluate.py \
    --checkpoint checkpoints-v7/best.pt \
    --split val

# Đánh giá trên test set
python scripts/evaluate.py \
    --checkpoint checkpoints-v7/best.pt \
    --split test
```

### 8.2 Đánh giá từ W&B artifact

```bash
# Dùng alias 'latest'
python scripts/evaluate.py --wandb-artifact latest --split test

# Dùng alias tên cụ thể
python scripts/evaluate.py --wandb-artifact epoch-025 --split val
```

### 8.3 Giao thức đánh giá

- **Full-ranking**: mỗi user được xếp hạng trên toàn bộ item catalog (không sampling).
- **Exclude items**: loại trừ toàn bộ item purchase trong train (`train_mask_purchase_only.pkl`) khỏi ranking.
- **Multi-positive ground truth**: NDCG được tính trên tất cả item đúng trong top-K.
- **Metrics**: HR@K (Hit Rate) và NDCG@K với K ∈ {1, 5, 10, 20, 50}.

---

## 9. Trích dẫn

### Tập dữ liệu

```bibtex
@dataset{kechinov2020rees46,
  author    = {Kechinov, Michael},
  title     = {{eCommerce} Behavior Data from Multi-Category Store},
  year      = {2020},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store}
}
```

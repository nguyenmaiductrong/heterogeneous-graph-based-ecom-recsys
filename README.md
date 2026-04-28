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
7. [Hướng dẫn sử dụng](#7-hướng-dẫn-sử-dụng)
8. [Trích dẫn](#8-trích-dẫn)

---

## 1. Giới thiệu

Kho mã nguồn này chứa toàn bộ cài đặt của mô hình **Mạng nơ-ron đồ thị nhận thức hành vi (BAGNN – Behavior-Aware Graph Neural Network)** phục vụ bài toán dự đoán hành vi mua hàng trong môi trường thương mại điện tử quy mô lớn. Mô hình hoạt động trên **Mạng thông tin không đồng nhất (HIN – Heterogeneous Information Network)** được xây dựng từ tập dữ liệu công khai REES46, bao gồm bốn loại nút (User, Product, Category, Brand) và năm loại quan hệ ngữ nghĩa (view, cart, purchase, belongsTo, producedBy).

**Các đóng góp kỹ thuật chính:**

- **Pipeline xử lý dữ liệu dựa trên PySpark**: có khả năng mở rộng, chuyển hóa hơn 200 triệu bản ghi tương tác thô thành đồ thị không đồng nhất (Heterogeneous Graph) có cấu trúc, hỗ trợ cả hai giao thức đánh giá: Temporal Split và Leave-One-Out (LOO).
- **Chiến lược Low-rank Behavior-aware Weight Decomposition**: `W(φ, β) = W_base + A_φ @ B_β^T` — cho phép mỗi bước truyền thông điệp (Message Passing) đặc thù hóa theo cặp (Node Type, Behavior Type) mà không làm tăng đột biến số lượng tham số.
- **Cơ chế chú ý 4 thành phần**: `[h_src ‖ h_dst ‖ r_emb ‖ β_emb]` kết hợp đồng thời thông tin của nút nguồn, nút đích, ngữ nghĩa quan hệ (Relation Semantics) và nguồn gốc hành vi (Behavior Context) trong một bước tính toán.
- **HierarchyGate**: Thực hiện soft-aggregation các embedding đặc thù theo từng hành vi (View, Cart, Purchase) dựa trên trọng số mức độ liên quan đến ý định mua hàng được học tự động
- **Popularity-biased Negative Sampler** tạo ra các mẫu âm khó với xác suất tỉ lệ thuận với độ phổ biến sản phẩm `p_i ∝ count_i^α`, giúp khắc phục vấn đề mẫu âm dễ (easy negatives) làm loãng tín hiệu huấn luyện.

---

## 2. Phân tích tập dữ liệu

Tập dữ liệu **REES46** là bộ nhật ký sự kiện (event logs) thương mại điện tử quy mô lớn, ghi lại các tương tác thực tế từ tháng **10/2019 đến 04/2020** của một nhà bán lẻ trực tuyến đa ngành. Từ các tệp CSV thô, dữ liệu trải qua quy trình tiền xử lý nghiêm ngặt bao gồm: lọc nhiễu, làm sạch, chuẩn hóa, loại bỏ các người dùng và sản phẩm có quá ít tương tác. Cuối cùng, dữ liệu được cấu trúc hóa dưới dạng đồ thị không đồng nhất (Heterogeneous Graph)

### 2.1 Thống kê nút

| Loại nút | Ký hiệu | Số lượng |
|---|---|---|
| User | \|U\| | 1.964.332 |
| Product | \|I\| | 311.682 |
| Category | \|C\| | 14 |
| Brand | \|B\| | 6.041 |
| **Tổng cộng** | | **2.282.069** |

### 2.2 Thống kê cạnh

| Quan hệ | Loại | Số cạnh | Độ thưa (Sparsity) |
|---|---|---|---|
| user → [view] → product | Behavioral | 185.679.073 | 0,999970 |
| user → [cart] → product | Behavioral | 14.192.808 | 0,999977 |
| user → [purchase] → product | Behavioral | 6.561.484 | **0,999989** |
| product → [belongsTo] → category | Structural | 311.682 | — |
| product → [producedBy] → brand | Structural | 311.682 | — |
| **Tổng cạnh hành vi** | | **206.433.365** | |

Độ thưa được tính theo công thức: `1 - |E_τ| / (|U| × |I|)`.

### 2.3 Đặc điểm nổi bật

| Thuộc tính | Giá trị |
|---|---|
| Mật độ tương tác (mua hàng) | 1,07 × 10⁻⁵ |
| Tỉ lệ view : cart : purchase | 28,3 : 2,2 : 1,0 |
| Người dùng chỉ có 1 lần mua (loại khỏi LOO eval) | 911.558 (46,4%) |
| Người dùng đủ điều kiện đánh giá LOO (≥ 2 lần mua) | 1.052.774 |
| Trung vị số lần mua / người dùng | 2,0 |
| Số lần mua cao nhất / người dùng | 2.119 |

> **Biểu đồ EDA:** Xem [`notebooks/01_graph_eda.ipynb`](notebooks/01_graph_eda.ipynb) để có đầy đủ các phân tích: phân phối bậc nút (Power-law), mất cân bằng lớp tương tác, và phân phối tương tác theo thời gian.

---

## 3. Pipeline xử lý dữ liệu

Pipeline chuyển đổi file CSV thô thành đồ thị không đồng nhất có cấu trúc qua **4 giai đoạn tuần tự**, được cài đặt trong `src/data_pipeline/` và điều phối bởi `scripts/run_pipeline.sh`.

![Data Pipeline Diagram](assets/data-pipeline.svg)

### 3.1 Chi tiết giao thức đánh giá

**Giao thức A — Leave-One-Out (LOO):**
Mỗi người dùng trong tập đánh giá đóng góp đúng một item kiểm tra (lần mua gần nhất) và một item xác thực (lần mua gần thứ hai). Toàn bộ tương tác xem và thêm giỏ hàng vẫn nằm trong đồ thị huấn luyện. Người dùng chỉ có duy nhất một lần mua bị loại khỏi tập đánh giá nhưng vẫn được giữ trong đồ thị huấn luyện. Kết quả thu được **1.052.774 người dùng đánh giá**, mỗi người có đúng một item đúng (ground-truth), được đánh giá trên tập 100 item ứng viên (1 đúng + 99 âm).

**Giao thức B — Temporal Split:**
Sử dụng ranh giới thời gian toàn cục: dữ liệu huấn luyện kết thúc vào ngày 29/02/2020, tập xác thực bao phủ tháng 3/2020, tập kiểm tra bao phủ tháng 4/2020. Giao thức này phù hợp hơn với môi trường triển khai thực tế và tránh rò rỉ thông tin tương lai theo thiết kế.

---

## 4. Kiến trúc mô hình

### 4.1 Lược đồ đồ thị không đồng nhất

Đồ thị đầy đủ gồm **10 loại quan hệ** (5 hướng ngữ nghĩa + 5 cạnh ngược):

```
user  ──[view]────────►  product  ──[belongs_to]──►  category
user  ──[cart]────────►  product  ──[producedBy]──►  brand
user  ──[purchase]────►  product
product  ──[rev_view / rev_cart / rev_purchase]──►  user
category  ──[contains]──►  product
brand  ──[brands]──►  product
```

### 4.2 BAGNN: Mạng nơ-ron đồ thị nhận thức hành vi

Mô hình đầy đủ (`src/model/bagnn.py`) xếp chồng ba thành phần có thể học:

#### Thành phần 1 — BehaviorAwareWeight: Phân rã ma trận truyền thông điệp dạng low-rank

Ma trận trọng số cho mỗi bước truyền thông điệp được phân rã thành:

```
W(φ, β) = W_base  +  A_φ @ B_β^T

  W_base : ma trận trọng số cơ sở chung        (d × d)
  A_φ    : nhân tố đặc thù theo loại nút       (d × r),   r = 16
  B_β    : nhân tố đặc thù theo loại hành vi   (d × r)
```

trong đó `φ` là chỉ số loại nút nguồn và `β ∈ {view=0, cart=1, purchase=2}`. Cạnh cấu trúc chỉ dùng `W_base` (khi `β = -1`).

#### Thành phần 2 — BAGNNConv: Cơ chế chú ý 4 thành phần nhận thức hành vi

Với mỗi quan hệ `r = (s, e, t)`, thông điệp được tính như sau:

```
m_i = W(φ_s, β_e) · h_i^(s)                             (biến đổi nhận thức hành vi)

α_ij = softmax_j( a^T · [m_i ‖ W·h_j^(t) ‖ r_emb ‖ β_emb] )   (chú ý 4 thành phần)

h_t^out = LayerNorm( Σ_j α_ij · m_j ) + h_t^in           (tổng hợp + kết nối dư)
```

`r_emb` là embedding loại quan hệ được học, `β_emb` là embedding loại hành vi được học.

#### Thành phần 3 — HierarchyGate: Tổng hợp mềm theo cấp độ hành vi

Sau khi thu được embedding theo từng hành vi, MLP hai lớp tính cổng mềm:

```
g = softmax( MLP( [z_view ‖ z_cart ‖ z_purchase] ) )   ∈ R^3
z_final = g[0]·z_view + g[1]·z_cart + g[2]·z_purchase
```

### 4.3 Hàm mục tiêu huấn luyện

Hàm mất mát tổng hợp kết hợp BPR có trọng số theo hành vi với chính quy hóa đối chiếu (contrastive):

```
L = Σ_{τ ∈ {view, cart, purchase}}  w_τ · L_BPR^τ  +  λ · L_CL

  w_purchase = 1,0  |  w_cart = 0,3  |  w_view = 0,1
```

Lấy mẫu âm (Negative Sampling) sử dụng **Popularity-biased Hard Negatives**: `p_i ∝ count_i^0,75`, với tỉ lệ dương:âm là 1:99.

### 4.4 Siêu tham số mô hình

| Hyperparameter | Value |
|---|---|
| Embedding Dimension ($d$) | 128 |
| GNN Layers | 2 |
| Low-rank Factor ($r$) | 16 |
| Attention Heads | 4 |
| Dropout | 0,1 |
| Optimizer | Adam |
| Neighbor Sampling Budget (Hop-1) | 15 |
| Neighbor Sampling Budget (Hop-2) | 10 |
| Negative Samples per Query (Evaluation) | 99 |

---

## 5. Cấu trúc dự án

```
heterogeneous-graph-based-ecom-recsys/
├── config/
│   └── spark_config.yaml           # Cấu hình pipeline: đường dẫn, ngưỡng lọc, mốc thời gian
│
├── scripts/
│   └── run_pipeline.sh             # Điểm vào cho toàn bộ pipeline 5 giai đoạn
│
├── src/
│   ├── core/
│   │   ├── contracts.py            # Dataclass có kiểu cho mọi cấu trúc I/O giữa các module
│   │   └── evaluator.py            # LeaveOneOutEvaluator: tính Recall@K, NDCG@K
│   │
│   ├── data_pipeline/
│   │   ├── spark_utils.py          # Khởi tạo SparkSession, schema REES46, nạp config
│   │   ├── extract.py              # Đọc CSV thô / Parquet / ánh xạ nút / mảng cạnh
│   │   ├── transform.py            # Làm sạch, lọc cold-start, xây cạnh, chia tập dữ liệu
│   │   └── load.py                 # Lưu Parquet, mảng .npy, ma trận thưa .npz
│   │
│   ├── graph/
│   │   ├── neighbor_sampler.py     # BehaviorAwareNeighborSampler (2-hop, vector hóa CSR)
│   │   └── contrastive.py          # Hàm mất mát đối chiếu với biểu diễn tăng cường SVD
│   │
│   ├── model/
│   │   ├── bagnn.py                # BehaviorAwareWeight, BAGNNConv, BAGNNLayer, BAGNNModel
│   │   ├── hetero_embedding.py     # Bảng embedding nút không đồng nhất
│   │   ├── hierarchy_gate.py       # HierarchyGate: tổng hợp mềm theo ưu tiên hành vi
│   └── training/
│       ├── trainer.py              # Trainer: AMP, early stopping, lưu/nạp checkpoint
│       ├── losses.py               # BPR loss, PopularityBiasedNegativeSampler
│       └── lightgcn_baseline.py    # Cài đặt mô hình nền LightGCN
│
├── notebooks/
│   └── 01_graph_eda.ipynb          # EDA đầy đủ: thống kê, phân phối bậc, biểu đồ thời gian
│
├── tests/                          # Kiểm thử và tích hợp (pytest)
├── requirements.txt
└── pyproject.toml
```

---

## 6. Cài đặt

### 6.1 Yêu cầu hệ thống

- Python 3.11
- Java 11 trở lên (bắt buộc cho PySpark)
- GPU hỗ trợ CUDA (khuyến nghị cho huấn luyện; CPU-only được hỗ trợ)

### 6.2 Thiết lập môi trường

```bash
# Clone repository
git clone https://github.com/nguyenmaiductrong/heterogeneous-graph-based-ecom-recsys
cd heterogeneous-graph-based-ecom-recsys

# Tạo và kích hoạt môi trường ảo
python3.11 -m venv rees46_env
source rees46_env/bin/activate   # Windows: rees46_env\Scripts\activate

# Cài đặt các phụ thuộc cốt lõi
pip install -r requirements.txt

# Cài đặt PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 6.3 Chuẩn bị dữ liệu

1. Tải các file CSV REES46 từ [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
2. Đặt file vào thư mục dữ liệu thô và cập nhật `config/spark_config.yaml`:

```yaml
paths:
  raw_csv_pattern: "/duong/dan/toi/rees46/raw/*.csv"
  output_dir:      "/duong/dan/toi/rees46/processed"
```

---

## 7. Hướng dẫn sử dụng

### 7.1 Chạy toàn bộ pipeline xử lý dữ liệu

```bash
bash scripts/run_pipeline.sh
```

**Thời gian ước tính:** 120 phút trên máy có 64 GB RAM và 8 CPU cores (PySpark chế độ local).

**Cấu trúc thư mục đầu ra:**

```
processed/
├── cleaned.parquet/          # Tương tác đã làm sạch và lọc cold-start
├── node_mappings/            # user2idx, product2idx, category2idx, brand2idx (.parquet + .csv)
├── edge_lists/               # Mảng cạnh .npy: {hanh_vi}_{split}_src/dst/ts.npy
│                             # Cạnh cấu trúc: belongsTo_src/dst.npy, producedBy_src/dst.npy
├── graph/                    # adj_{hanh_vi}_train.npz + graph_meta.json
├── loo/                      # adj_loo_{hanh_vi}_train.npz, cặp val/test .npy, loo_meta.json
└── statistics/               # node_summary.json
```

### 7.2 Phân tích khám phá dữ liệu (EDA)

```bash
source rees46_env/bin/activate
jupyter notebook notebooks/01_graph_eda.ipynb
```

Các biểu đồ được lưu vào `notebooks/figures/` dưới cả hai định dạng `.pdf` (cho bài báo) và `.png`.

## 8. Trích dẫn
### Trích dẫn tập dữ liệu

```bibtex
@dataset{kechinov2020rees46,
  author    = {Kechinov, Michael},
  title     = {{eCommerce} Behavior Data from Multi-Category Store},
  year      = {2020},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store}
}
```


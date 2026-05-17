# Dự đoán tương tác người dùng - sản phẩm bằng đồ thị không đồng nhất

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![PySpark 3.5.1](https://img.shields.io/badge/PySpark-3.5.1-E25A1C.svg)](https://spark.apache.org/)

Kho mã nguồn này triển khai bài toán dự đoán tương tác giữa người dùng và sản phẩm trên dữ liệu thương mại điện tử REES46. Mục tiêu chính là học biểu diễn trên đồ thị không đồng nhất để xếp hạng khả năng người dùng sẽ tương tác/mua sản phẩm trong tương lai; gợi ý sản phẩm là ứng dụng trực tiếp của kết quả xếp hạng này.

Dự án gồm hai phần chính:

- Pipeline PySpark để xử lý log hành vi `view`, `cart`, `purchase`, tạo temporal split và lưu artefact huấn luyện.
- Mô hình BPATMP trên PyTorch/PyG để tính điểm tương tác user-product, dùng neighbor sampling, temporal attention, contrastive learning phân cấp và đánh giá full-ranking.

## Mục Lục

1. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
2. [Luồng Xử Lý](#luồng-xử-lý)
3. [Dữ Liệu](#dữ-liệu)
4. [Cài Đặt](#cài-đặt)
5. [Chuẩn Bị Dữ Liệu](#chuẩn-bị-dữ-liệu)
6. [Huấn Luyện](#huấn-luyện)
7. [Đánh Giá](#đánh-giá)
8. [Thành Phần Chính](#thành-phần-chính)
9. [Notebook Và Log](#notebook-và-log)

## Cấu Trúc Dự Án

```text
heterogeneous-graph-based-ecom-recsys/
├── config/
│   ├── spark_config.yaml          # Cấu hình Spark, protocol temporal split, filter, graph schema
│   └── training.yaml              # Cấu hình data/model/sampler/loss/training/evaluation/W&B
│
├── scripts/
│   ├── download_data.py           # Tải bộ artefact đã xử lý từ Hugging Face về data/
│   ├── prepare_data.py            # Entry point pipeline PySpark xử lý dữ liệu thô
│   ├── run_pipeline.sh            # Wrapper shell cho prepare_data.py
│   ├── run_training.py            # Entry point huấn luyện BPATMP dự đoán user-product interaction
│   └── evaluate.py                # Đánh giá checkpoint local hoặc W&B artifact
│
├── src/
│   ├── core/
│   │   ├── contracts.py           # Hằng số graph schema, dataclass I/O, EvalInput
│   │   └── evaluator.py           # TemporalSplitEvaluator full-ranking HR@K, NDCG@K
│   │
│   ├── data_pipeline/
│   │   ├── extract.py             # Đọc CSV REES46 và làm sạch dữ liệu
│   │   ├── transform.py           # Lọc, map vocab, tạo split, mask, cạnh cấu trúc
│   │   ├── splitter.py            # Temporal split cho target behavior purchase
│   │   ├── load.py                # Lưu .npy/.pkl/.json/.parquet và verify artefact
│   │   ├── sanity.py              # Kiểm tra leakage, mask, bounds, schema
│   │   └── spark_utils.py         # Tạo SparkSession và load YAML config
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── neighbor_sampler.py    # BehaviorAwareNeighborSampler, CSR sampling 2-hop
│   │
│   ├── model/
│   │   └── bpatmp.py              # BPATMPModel và các block temporal/message passing
│   │
│   └── training/
│       ├── trainer.py             # TrainConfig, train loop, eval_epoch, export embeddings
│       ├── losses.py              # BPR, HierarchicalMBCL, funnel prior, monotonic decay prior
│       └── checkpoint_manager.py  # Lưu/phục hồi checkpoint local và W&B artifact
│
├── notebooks/
│   ├── 01_graph_eda.ipynb
│   ├── 02_graph_sage.ipynb
│   ├── 03_ml_fraction_matrix.ipynb
│   ├── 04_CRGCN_colab.ipynb
│   └── figures/
│       ├── fig_degree_distribution.png
│       ├── fig_interaction_distribution.png
│       ├── fig_temporal_distribution.png
│       └── fig_user_purchase_sequence.png
│
├── data/                          # Không version control: dữ liệu tải về hoặc artefact pipeline
├── assets/                        # Tài nguyên minh họa nếu cần
├── requirements.txt
├── pytest.ini
├── pyproject.toml
├── pipeline_run.log
└── training_history.md
```

## Luồng Xử Lý

```text
CSV thô REES46
  -> src/data_pipeline/extract.py
  -> src/data_pipeline/transform.py
  -> src/data_pipeline/splitter.py
  -> src/data_pipeline/load.py
  -> data artefacts (.npy, .pkl, .json, .parquet)
  -> scripts/run_training.py
  -> src/model/bpatmp.py + src/training/trainer.py
  -> checkpoint mô hình dự đoán tương tác + metrics
  -> scripts/evaluate.py
```

Protocol mặc định trong [config/spark_config.yaml](config/spark_config.yaml):

- `train`: dữ liệu đến hết `2020-02-29`
- `val`: tháng `2020-03`
- `test`: tháng `2020-04`
- Target behavior: `purchase`
- Candidate set: item warm-start trong train
- Ground truth: tất cả sản phẩm được user mua mới trong split đánh giá
- Evaluation: full-ranking với mask purchase train

## Dữ Liệu

Dự án hỗ trợ hai cách chuẩn bị dữ liệu.

### 1. Dùng Artefact Đã Xử Lý

[scripts/download_data.py](scripts/download_data.py) tải bộ dữ liệu đã xử lý về thư mục `data/`.

```bash
pip install huggingface_hub
python scripts/download_data.py
```

Layout sau khi tải:

```text
data/
├── view_train_src.npy
├── view_train_dst.npy
├── view_train_ts.npy
├── cart_train_src.npy
├── cart_train_dst.npy
├── cart_train_ts.npy
├── purchase_train_src.npy
├── purchase_train_dst.npy
├── purchase_train_ts.npy
├── val_user_idx.npy
├── val_product_idx.npy
├── val_timestamp.npy
├── test_user_idx.npy
├── test_product_idx.npy
├── test_timestamp.npy
├── val_ground_truth.pkl
├── test_ground_truth.pkl
├── train_mask_purchase_only.pkl
├── train_mask_seen_all.pkl
├── train_mask.pkl
├── candidate_item_idx.npy
├── node_counts.json
├── graph/
│   ├── train_events.parquet/
│   ├── val_ground_truth.parquet
│   ├── test_ground_truth.parquet
│   └── item_metadata.parquet
└── node_mappings/
    ├── user2idx.json
    ├── item2idx.json
    ├── category2idx.json
    ├── brand2idx.json
    ├── behavior2idx.json
    ├── product_category.parquet
    └── product_brand.parquet
```

Nếu chạy local với layout này, cấu hình đường dẫn trong [config/training.yaml](config/training.yaml) nên trỏ về:

```yaml
data:
  data_dir: data
  struct_dir: data/node_mappings
```

### 2. Tự Chạy Pipeline Từ CSV Thô

Đặt các file CSV REES46 vào `data/raw/`, sau đó chạy:

```bash
bash scripts/run_pipeline.sh
```

Mặc định wrapper gọi [scripts/prepare_data.py](scripts/prepare_data.py) với:

```text
--csv-glob data/raw/*.csv
--data-dir data/processed/temporal
--struct-dir data/processed/temporal/node_mappings
--graph-dir data/processed/temporal/graph
```

Có thể override trực tiếp:

```bash
bash scripts/run_pipeline.sh \
  --csv-glob "data/raw/*.csv" \
  --data-dir data/processed/temporal \
  --struct-dir data/processed/temporal/node_mappings \
  --graph-dir data/processed/temporal/graph \
  --train-end 2020-02-29 \
  --val-end 2020-03-31
```

Sau khi pipeline hoàn tất, cập nhật [config/training.yaml](config/training.yaml):

```yaml
data:
  data_dir: data/processed/temporal
  struct_dir: data/processed/temporal/node_mappings
```

## Cài Đặt

Yêu cầu chính:

- Python 3.11
- Java 8/11+ cho PySpark
- CUDA GPU được khuyến nghị cho huấn luyện

Tạo môi trường conda khớp mặc định của [scripts/run_pipeline.sh](scripts/run_pipeline.sh):

```bash
conda create -n recsys_env python=3.11 -y
conda activate recsys_env
python -m pip install --upgrade pip
```

Cài dependency:

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Thiết lập `PYTHONPATH` khi chạy script từ repo root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Chuẩn Bị Dữ Liệu

Pipeline PySpark gồm 7 giai đoạn trong [scripts/prepare_data.py](scripts/prepare_data.py):

1. Đọc và làm sạch CSV thô.
2. Lọc user/item theo số purchase trong cửa sổ train.
3. Chia temporal split cho purchase.
4. Map vocab, build train events, build cạnh product-category/product-brand.
5. Tạo mask full-ranking: `train_mask_purchase_only.pkl` và `train_mask_seen_all.pkl`.
6. Lưu artefact `.npy`, `.pkl`, `.json`, `.parquet`.
7. Chạy sanity check không dùng Spark.

Lệnh chạy nhanh:

```bash
bash scripts/run_pipeline.sh
```

Wrapper này sẽ `conda activate recsys_env`. Nếu dùng tên môi trường khác, truyền thêm `--conda-env`:

```bash
bash scripts/run_pipeline.sh --conda-env ten_moi_truong
```

Pipeline dùng cấu hình Spark từ [config/spark_config.yaml](config/spark_config.yaml). Nếu máy không đủ RAM, giảm các trường như `driver_memory`, `executor_memory`, `shuffle_partitions`, `default_parallelism`.

## Huấn Luyện

Điểm vào huấn luyện mô hình dự đoán tương tác user-product là [scripts/run_training.py](scripts/run_training.py):

```bash
python scripts/run_training.py --config config/training.yaml
```

Các nhóm cấu hình quan trọng trong [config/training.yaml](config/training.yaml):

- `data`: đường dẫn artefact và `node_counts`
- `model`: `embed_dim`, `n_layers`, `rank`, `n_intents`, `dropout`
- `sampler`: budget neighbor sampling hop-1/hop-2
- `loss`: trọng số BPR/CL/funnel/monotonic prior
- `training`: batch size, learning rate, AMP/bf16, checkpoint dir, early stopping
- `evaluation`: full-ranking metrics và primary metric
- `wandb`: logging và model artifact

Checkpoint local được lưu vào `training.save_dir`:

```text
checkpoints-final-l4/
├── best.pt
├── epoch_000.pt
├── epoch_001.pt
├── ...
└── final_val_metrics.json
```

Nếu `wandb.enabled: true`, checkpoint cũng được upload bằng [src/training/checkpoint_manager.py](src/training/checkpoint_manager.py) với alias `latest` và `epoch-XXX`.

## Đánh Giá

Đánh giá checkpoint local:

```bash
python scripts/evaluate.py \
  --config config/training.yaml \
  --checkpoint checkpoints-final-l4/best.pt \
  --split test
```

Đánh giá từ W&B artifact:

```bash
python scripts/evaluate.py \
  --config config/training.yaml \
  --wandb-artifact latest \
  --split test
```

Evaluator dùng full-ranking tiled scoring trong [src/core/evaluator.py](src/core/evaluator.py):

- HR@1/5/10/20/50
- NDCG@1/5/10/20/50
- Loại các item đã purchase trong train khỏi ranking bằng `train_mask_purchase_only.pkl`
- Hỗ trợ multi-positive ground truth cho mỗi user

## Thành Phần Chính

### Graph Schema

```text
user    --view-------> product
user    --cart-------> product
user    --purchase---> product
product --rev_view---> user
product --rev_cart---> user
product --rev_purchase--> user
product --belongs_to-> category
category --contains--> product
product --producedBy-> brand
brand   --brands-----> product
```

Các relation này được định nghĩa trong [src/core/contracts.py](src/core/contracts.py) và được build trong [scripts/run_training.py](scripts/run_training.py).

### BPATMP Model

[src/model/bpatmp.py](src/model/bpatmp.py) chứa các khối chính:

- `BehaviorAwareWeight`: low-rank transform theo relation và behavior.
- `FourierTimeEncoding`: mã hóa thời gian bằng Fourier feature học được.
- `TemporalAttention`: attention có time bias, decay-in-logit và value gate.
- `BPATMPModel`: mô hình chính để học embedding user/item và tính điểm tương tác user-product.

### Training

[src/training/trainer.py](src/training/trainer.py) xử lý:

- Build dataloader từ triplet `(user, item, behavior, timestamp)`.
- Sample subgraph bằng `BehaviorAwareNeighborSampler`.
- Mixed negative sampling dựa trên history CSR, popularity và local hard signal.
- AMP/bf16, gradient clipping, AdamW, cosine warmup scheduler.
- Eval định kỳ trên validation và early stopping theo `NDCG@20`.

[src/training/losses.py](src/training/losses.py) chứa:

- Multi-task BPR theo hành vi.
- Hierarchical MBCL cho quan hệ `view -> cart -> purchase`.
- Funnel prior.
- Monotonic decay prior.

## Notebook Và Log

- [notebooks/01_graph_eda.ipynb](notebooks/01_graph_eda.ipynb): EDA phân phối tương tác, bậc nút, thời gian.
- [notebooks/02_graph_sage.ipynb](notebooks/02_graph_sage.ipynb): baseline/khảo sát GraphSAGE.
- [notebooks/03_ml_fraction_matrix.ipynb](notebooks/03_ml_fraction_matrix.ipynb): thử nghiệm ma trận/fraction baseline.
- [notebooks/04_CRGCN_colab.ipynb](notebooks/04_CRGCN_colab.ipynb): thử nghiệm CRGCN trên Colab.
- [training_history.md](training_history.md): ghi chú lịch sử huấn luyện.
- [pipeline_run.log](pipeline_run.log): log lần chạy pipeline đã lưu trong repo.

# Demo trực quan BPATMP

Demo web tĩnh **nền sáng**, trình bày theo phong cách [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/):
sơ đồ kiến trúc chạy ngang ở trên, **bấm vào từng lớp** để mở chi tiết “chạy tay” bên dưới (lớp này tính gì, chú ý ra sao, kết quả/loss thế nào). Đi đúng theo **pipeline và mô hình thật** của dự án, gồm 2 tab:

- **① Huấn luyện** — đi từ dữ liệu thô → làm sạch → vocab → tách thời gian → dựng đồ thị không đồng nhất → lấy mẫu láng giềng → **từng lớp của mô hình**: Input Embedding → BehaviorAwareWeight → Temporal Attention → Behavior-Normalized Aggregation → Intent Codebook → hàm mất mát đa nhiệm → embedding đầu ra.
- **② Đánh giá &amp; Gợi ý** — chọn người dùng → xem lịch sử tương tác → xem embedding user → xem embedding sản phẩm ứng viên → tính điểm dot product → danh sách gợi ý xếp hạng.

Mọi vector và con số ở các lớp mô hình (`layers` trong `demo_data.json`) đều đọc trực tiếp từ mô hình `BPATMPModel` đã huấn luyện.

## Nguồn số liệu

Mọi con số trong demo do [`scripts/build_demo_data.py`](../scripts/build_demo_data.py) sinh ra bằng cách **chạy chính code thật** (`BPATMPModel`, `BehaviorAwareNeighborSampler`, `BPATMPTotalLoss`, `TemporalSplitEvaluator`) trên một **mock slice nhỏ** trích từ dữ liệu REES46 trong `data/`. Web chỉ render `demo_data.json`, không tự tính lại mô hình.

Để dễ theo dõi từng con số, demo dùng embedding nhỏ (`d=16`, 2 lớp); kiến trúc giữ nguyên như cấu hình production (`d=256`, 3 lớp).

## Tạo lại dữ liệu demo

```bash
PYTHONPATH=. python scripts/build_demo_data.py
```

Lệnh này ghi đè `demo/demo_data.json`.

## Chạy demo

Demo là web tĩnh nhưng cần fetch `demo_data.json`, vì vậy phải chạy qua HTTP server từ **thư mục gốc repo**. Không mở trực tiếp `demo/index.html` bằng `file://`.

### Chế độ tĩnh

Chế độ này hiển thị toàn bộ flow huấn luyện và suy luận bằng `demo/demo_data.json`.

```bash
cd /home/nmdt/heterogeneous-graph-based-ecom-recsys
python3 -m http.server 8000 --bind 127.0.0.1
```

Sau đó mở trình duyệt:

```text
http://127.0.0.1:8000/demo/
```

Nếu cổng `8000` đang được dùng, đổi sang cổng khác, ví dụ:

```bash
python3 -m http.server 8001 --bind 127.0.0.1
```

Rồi mở `http://127.0.0.1:8001/demo/`.

### Chế độ live checkpoint

Để tab **Đánh giá & Gợi ý** đọc trực tiếp best checkpoint `checkpoints/downloaded/epoch_003.pt`, chạy thêm backend FastAPI:

```bash
python -m uvicorn src.backend.api:app --host 127.0.0.1 --port 8000
```

Sau đó chạy static server ở cổng khác:

```bash
python -m http.server 8001 --bind 127.0.0.1
```

Mở:

```text
http://127.0.0.1:8001/demo/
```

Khi backend sẵn sàng, tab đánh giá sẽ hiện panel **Live checkpoint** với epoch, số embedding, metric trong checkpoint, pipeline backend và top gợi ý tính từ API. Nếu backend chưa bật, demo tự fallback về dữ liệu tĩnh.

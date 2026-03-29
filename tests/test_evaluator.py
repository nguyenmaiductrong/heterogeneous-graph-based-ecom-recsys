import pytest
import torch
import math
from src.core.contracts import EvalInput
from src.core.evaluator import FullRankingEvaluator

def test_evaluator_metrics_and_masking():
    """
    Test tính toán Recall, NDCG và logic Masking (loại bỏ item cũ).
    Sử dụng ma trận nhỏ để có thể tính nhẩm thủ công và so sánh kết quả.
    """
   # 1. Setup Dữ liệu giả lập (2 Users, 4 Items) - KHÔNG ĐIỂM HÒA
    from src.core.contracts import EMBED_DIM
    
    user_embs = torch.zeros(2, EMBED_DIM)
    item_embs = torch.zeros(4, EMBED_DIM)
    
    user_embs[0, 0] = 1.0
    user_embs[1, 1] = 1.0
    
    # Thiết lập điểm số phân cấp rõ ràng, không có con số nào bằng nhau
    item_embs[0, 0] = 10.0; item_embs[0, 1] = 1.0  
    item_embs[1, 0] = 8.0;  item_embs[1, 1] = 5.0  
    item_embs[2, 0] = 5.0;  item_embs[2, 1] = 10.0 
    item_embs[3, 0] = 1.0;  item_embs[3, 1] = 8.0  

    eval_user_ids = torch.tensor([0, 1])
    
    # ---------------------------------------------------------
    # TEST CASE 1: Hiệu ứng của Masking
    # ---------------------------------------------------------
    # Target của User 0 là Item 1 (Score = 8). 
    # Bình thường Item 1 sẽ đứng thứ 2 (sau Item 0 - Score 10).
    # NHƯNG ta sẽ mask Item 0 (coi như user đã view/cart).
    # Lúc này Item 1 vươn lên Rank 1.
    
    # Target của User 1 là Item 2 (Score = 10 -> Rank 1 sẵn).
    ground_truth_1 = {0: 1, 1: 2}
    exclude_items = {0: [0], 1: []} # Mask Item 0 cho User 0

    eval_input_1 = EvalInput(
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        eval_user_ids=eval_user_ids,
        ground_truth=ground_truth_1,
        exclude_items=exclude_items
    )

    evaluator = FullRankingEvaluator(ks=[1, 2], device='cpu')
    metrics_1 = evaluator.evaluate(eval_input_1, batch_size=2)

    # Cả hai user đều có target lọt top 1.
    assert metrics_1['Recall@1'] == 1.0, "Masking logic bị sai, User 0 không đưa Item 1 lên Top 1 được."
    assert metrics_1['NDCG@1'] == 1.0

    # ---------------------------------------------------------
    # TEST CASE 2: Tính toán vị trí (Rank > 1) và công thức NDCG
    # ---------------------------------------------------------
    # Đổi target: 
    # User 0 target Item 3 (Score = 0 -> Rank bét, trượt) -> Hit@2 = 0
    # User 1 target Item 3 (Score = 8 -> Rank 2, trúng) -> Hit@2 = 1, Hit@1 = 0
    ground_truth_2 = {0: 3, 1: 3}
    
    eval_input_2 = EvalInput(
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        eval_user_ids=eval_user_ids,
        ground_truth=ground_truth_2,
        exclude_items=exclude_items
    )

    metrics_2 = evaluator.evaluate(eval_input_2, batch_size=2)

    # Tổng Recall@1 = 0 (Không ai trúng Top 1)
    assert metrics_2['Recall@1'] == 0.0
    
    # Tổng Recall@2 = (0 + 1) / 2 = 0.5
    assert metrics_2['Recall@2'] == 0.5
    
    # NDCG của User 1 tại Rank 2 = 1 / log2(2 + 1) = 1 / log2(3) = 0.6309
    # Tổng NDCG@2 = (0 + 0.6309) / 2 = 0.3154
    expected_ndcg_2 = 0.5 * (1.0 / math.log2(3))
    assert math.isclose(metrics_2['NDCG@2'], expected_ndcg_2, rel_tol=1e-4), "Tính toán NDCG sai lệch"

def test_evaluator_validation_trigger():
    """
    Test xem Evaluator có thực sự gọi hàm validate() của EvalInput để chặn dữ liệu sai không.
    """
    # Cố tình tạo data sai (shape không khớp)
    eval_input_invalid = EvalInput(
        user_embeddings=torch.randn(2, 64), # Sai số chiều so với EMBED_DIM trong contracts (128)
        item_embeddings=torch.randn(4, 128),
        eval_user_ids=torch.tensor([0, 1]),
        ground_truth={0: 1, 1: 2},
        exclude_items={}
    )
    
    evaluator = FullRankingEvaluator(ks=[10], device='cpu')
    
    # Chạy hàm evaluate, kỳ vọng sẽ ném ra lỗi AssertionError từ file contracts.py
    with pytest.raises(AssertionError):
        evaluator.evaluate(eval_input_invalid, batch_size=2)
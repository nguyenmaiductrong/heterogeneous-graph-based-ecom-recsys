import pytest
import torch
import math
from src.core.contracts import EvalInput
from src.core.evaluator import FullRankingEvaluator

def test_evaluator_metrics_and_masking():
    from src.core.contracts import EMBED_DIM
    
    user_embs = torch.zeros(2, EMBED_DIM)
    item_embs = torch.zeros(4, EMBED_DIM)
    
    user_embs[0, 0] = 1.0
    user_embs[1, 1] = 1.0

    item_embs[0, 0] = 10.0; item_embs[0, 1] = 1.0  
    item_embs[1, 0] = 8.0;  item_embs[1, 1] = 5.0  
    item_embs[2, 0] = 5.0;  item_embs[2, 1] = 10.0 
    item_embs[3, 0] = 1.0;  item_embs[3, 1] = 8.0  

    eval_user_ids = torch.tensor([0, 1])
    
    # ---------------------------------------------------------
    # TEST CASE 1: Hiệu ứng của Masking (Phải chạy mode='full')
    # ---------------------------------------------------------
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
    
    metrics_1 = evaluator.evaluate(eval_input_1, batch_size=2, mode="full")

    # Cả hai user đều có target lọt top 1.
    assert metrics_1['HR@1'] == 1.0, "Masking logic bị sai, User 0 không đưa Item 1 lên Top 1 được."
    assert metrics_1['NDCG@1'] == 1.0

    # ---------------------------------------------------------
    # TEST CASE 2: Tính toán vị trí (Rank > 1) và công thức NDCG
    # ---------------------------------------------------------
    ground_truth_2 = {0: 3, 1: 3}
    
    eval_input_2 = EvalInput(
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        eval_user_ids=eval_user_ids,
        ground_truth=ground_truth_2,
        exclude_items=exclude_items
    )

    # [FIX] Đã thêm mode="full" vào đây
    metrics_2 = evaluator.evaluate(eval_input_2, batch_size=2, mode="full")

    # Tổng HR@1 = 0 (Không ai trúng Top 1)
    assert metrics_2['HR@1'] == 0.0

    # Tổng HR@2 = (0 + 1) / 2 = 0.5
    assert metrics_2['HR@2'] == 0.5
    
    # NDCG của User 1 tại Rank 2 = 1 / log2(2 + 1) = 1 / log2(3) = 0.6309
    # Tổng NDCG@2 = (0 + 0.6309) / 2 = 0.3154
    expected_ndcg_2 = 0.5 * (1.0 / math.log2(3))
    assert math.isclose(metrics_2['NDCG@2'], expected_ndcg_2, rel_tol=1e-4), "Tính toán NDCG sai lệch"


def test_evaluator_validation_trigger():
    eval_input_invalid = EvalInput(
        user_embeddings=torch.randn(2, 64), # Sai số chiều so với EMBED_DIM trong contracts
        item_embeddings=torch.randn(4, 128),
        eval_user_ids=torch.tensor([0, 1]),
        ground_truth={0: 1, 1: 2},
        exclude_items={}
    )
    
    evaluator = FullRankingEvaluator(ks=[10], device='cpu')
    
    with pytest.raises(AssertionError):
        # Validation sẽ được gọi ngay đầu hàm evaluate, nên mode nào cũng sẽ catch được lỗi
        evaluator.evaluate(eval_input_invalid, batch_size=2)
import torch
import time
import argparse
from typing import Dict, List

from .contracts import EvalInput

class FullRankingEvaluator:
    def __init__(self, ks: List[int] = [10, 20], device: str = 'cuda'):
        self.ks = ks
        self.max_k = max(ks)git
        self.device = device if torch.cuda.is_available() else 'cpu'

    @torch.no_grad()
    def evaluate(self, eval_input: EvalInput, batch_size: int = 4096) -> Dict[str, float]:
        # 1. Chạy hàm kiểm tra chuẩn hóa từ file contracts.py
        eval_input.validate()

        # 2. Đưa dữ liệu lên thiết bị (GPU/CPU)
        user_embs = eval_input.user_embeddings.to(self.device)
        item_embs = eval_input.item_embeddings.to(self.device)
        
        n_users_eval = eval_input.eval_user_ids.size(0)
        eval_user_ids_cpu = eval_input.eval_user_ids.cpu().numpy() 
        
        # Chuyển ground_truth (Dict) thành Tensor cùng thứ tự với eval_user_ids
        targets = [eval_input.ground_truth[u.item()] for u in eval_input.eval_user_ids]
        test_targets = torch.tensor(targets, device=self.device).view(-1, 1)

        # 3. Khởi tạo Metrics trên GPU
        metrics_gpu = {f'Recall@{k}': torch.tensor(0.0, device=self.device) for k in self.ks}
        metrics_gpu.update({f'NDCG@{k}': torch.tensor(0.0, device=self.device) for k in self.ks})

        ranks = torch.arange(1, self.max_k + 1, device=self.device).float()
        ndcg_weights = 1.0 / torch.log2(ranks + 1.0)

        # 4. Batched Processing
        for i in range(0, n_users_eval, batch_size):
            u_batch_cpu = eval_user_ids_cpu[i:i+batch_size]
            target_batch = test_targets[i:i+batch_size]
            
            # Theo contract, user_embeddings đã có shape (num_eval_users, d) 
            # nên ta có thể slice trực tiếp theo index i thay vì tìm theo user_id
            u_batch_embs = user_embs[i:i+batch_size]
            
            # Dot-product Scores
            scores = torch.matmul(u_batch_embs, item_embs.T)
            
            # Train Masking (-inf) sử dụng exclude_items
            for batch_idx, user_id in enumerate(u_batch_cpu):
                if user_id in eval_input.exclude_items:
                    interacted_items = list(eval_input.exclude_items[user_id])
                    scores[batch_idx, interacted_items] = -1e9

            # Top-K & Hits
            _, topk_indices = torch.topk(scores, k=self.max_k, dim=-1)
            hits = (topk_indices == target_batch) 
            
            for k in self.ks:
                hits_k = hits[:, :k] 
                metrics_gpu[f'Recall@{k}'] += hits_k.sum()
                ndcg_k_scores = hits_k * ndcg_weights[:k]
                metrics_gpu[f'NDCG@{k}'] += ndcg_k_scores.sum()

        # 5. Kéo kết quả về CPU
        metrics = {k: (v.item() / n_users_eval) for k, v in metrics_gpu.items()}
            
        return metrics

def run_testpass():
    print("Khởi chạy Testpass với EvalInput Contract...")
    
    n_users_eval = 1_050_000 # Gần khớp với 1,052,774 users của bạn
    n_items_total = 50_000
    dim = 128 # Đổi thành 128 cho khớp với EMBED_DIM trong contracts.py
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Đang tạo EvalInput giả lập ({device.upper()})...")
    
    # Tạo dữ liệu giả khớp hoàn toàn với contract EvalInput
    eval_user_ids = torch.arange(n_users_eval)
    ground_truth = {i: i % n_items_total for i in range(n_users_eval)}
    exclude_items = {i: [(i + 1) % n_items_total, (i + 2) % n_items_total] for i in range(n_users_eval)}
    
    eval_input = EvalInput(
        user_embeddings=torch.randn(n_users_eval, dim),
        item_embeddings=torch.randn(n_items_total, dim),
        eval_user_ids=eval_user_ids,
        ground_truth=ground_truth,
        exclude_items=exclude_items
    )

    evaluator = FullRankingEvaluator(ks=[10, 20], device=device)
    
    print("Bắt đầu đánh giá...")
    start_time = time.time()
    
    metrics = evaluator.evaluate(eval_input, batch_size=4096)
    
    duration = time.time() - start_time
    
    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (METRICS)")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")
        
    print(f"\nXử lý {n_users_eval:,} users mất {duration:.2f} giây.")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testpass', action='store_true')
    args = parser.parse_args()
    if args.testpass:
        run_testpass()
import gc
import logging
import time
import numpy as np
import torch
from .contracts import EvalInput, EMBED_DIM

logger = logging.getLogger(__name__)

class TemporalSplitEvaluator:

    def __init__(
        self,
        ks: list[int] | None = None,
        num_neg_samples: int = 999,
        device: str = "cpu",
    ):
        self.ks = sorted(ks or [10, 20, 50])
        self.max_k = max(self.ks)
        self.num_neg_samples = num_neg_samples
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)


    @torch.no_grad()
    def evaluate_sampled(
        self,
        eval_input: EvalInput,
        batch_size: int = 2048,
        seed: int = 42,
    ) -> dict[str, float]:
        eval_input.validate()

        rng = np.random.RandomState(seed)
        n_eval = eval_input.eval_user_ids.size(0)
        n_items = eval_input.item_embeddings.size(0)
        n_cand = 1 + self.num_neg_samples

        use_fp16 = self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32
        item_embs = eval_input.item_embeddings.to(
            device=self.device, dtype=dtype
        )

        metrics_sum: dict[str, float] = {
            f"{m}@{k}": 0.0 for m in ("Recall", "NDCG") for k in self.ks
        }

        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            bs = end - start
            batch_uids = eval_input.eval_user_ids[start:end]

            # ── candidate matrix:  col 0 = positive, cols 1.. = negatives
            uid_list = batch_uids.tolist()
            pos_ids = np.array(
                [eval_input.ground_truth[uid] for uid in uid_list],
                dtype=np.int64,
            )
            neg_ids = rng.randint(0, n_items, size=(bs, self.num_neg_samples))

            # fix rare collision where a negative == positive
            collision = neg_ids == pos_ids[:, None]
            n_collision = int(collision.sum())
            if n_collision > 0:
                neg_ids[collision] = rng.randint(0, n_items, size=n_collision)

            candidates = np.empty((bs, n_cand), dtype=np.int64)
            candidates[:, 0] = pos_ids
            candidates[:, 1:] = neg_ids

            cand_t = torch.from_numpy(candidates).to(self.device)
            cand_embs = item_embs[cand_t]          # (bs, n_cand, d)
            batch_uembs = eval_input.user_embeddings[start:end].to(
                device=self.device, dtype=dtype
            )

            scores = torch.sum(
                batch_uembs.unsqueeze(1) * cand_embs, dim=-1
            )  

            pos_score = scores[:, 0:1]                    
            rank = (scores > pos_score).sum(dim=-1) + 1   

            for k in self.ks:
                hit = (rank <= k).float()
                metrics_sum[f"Recall@{k}"] += hit.sum().item()
                ndcg = torch.where(
                    rank <= k,
                    1.0 / torch.log2(rank.float() + 1.0),
                    torch.tensor(0.0, device=self.device),
                )
                metrics_sum[f"NDCG@{k}"] += ndcg.sum().item()

            del batch_uembs, cand_embs, scores, cand_t, candidates, neg_ids

        del item_embs
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {k: v / n_eval for k, v in metrics_sum.items()}

    @torch.no_grad()
    def evaluate_full_ranking(
        self,
        eval_input: EvalInput,
        batch_size: int = 512,
    ) -> dict[str, float]:
        eval_input.validate()

        n_eval = eval_input.eval_user_ids.size(0)

        use_fp16 = self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32
        item_embs = eval_input.item_embeddings.to(
            device=self.device, dtype=dtype
        )

        ranks_range = torch.arange(
            1, self.max_k + 1, device=self.device
        ).float()
        ndcg_weights = 1.0 / torch.log2(ranks_range + 1.0)

        metrics_sum: dict[str, float] = {
            f"{m}@{k}": 0.0 for m in ("Recall", "NDCG") for k in self.ks
        }

        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            batch_uids = eval_input.eval_user_ids[start:end]
            batch_uembs = eval_input.user_embeddings[start:end].to(
                device=self.device, dtype=dtype
            )

            scores = torch.matmul(batch_uembs, item_embs.T)  
            mask_rows: list[int] = []
            mask_cols: list[int] = []
            for idx, uid in enumerate(batch_uids.tolist()):
                items = eval_input.exclude_items.get(uid)
                if items:
                    mask_rows.extend([idx] * len(items))
                    mask_cols.extend(items)
            if mask_rows:
                scores[mask_rows, mask_cols] = -1e9

            targets = torch.tensor(
                [eval_input.ground_truth[uid] for uid in batch_uids.tolist()],
                device=self.device,
            ).view(-1, 1)

            _, topk_idx = torch.topk(scores, k=self.max_k, dim=-1)
            hits = topk_idx == targets  # (bs, max_k)

            for k in self.ks:
                hits_k = hits[:, :k].float()
                metrics_sum[f"Recall@{k}"] += hits_k.sum().item()
                metrics_sum[f"NDCG@{k}"] += (
                    hits_k * ndcg_weights[:k]
                ).sum().item()

            del batch_uembs, scores, topk_idx, hits

        del item_embs
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {k: v / n_eval for k, v in metrics_sum.items()}


    def evaluate(
        self,
        eval_input: EvalInput,
        batch_size: int = 2048,
        mode: str = "sampled",
        seed: int = 42,
    ) -> dict[str, float]:
        if mode == "sampled":
            return self.evaluate_sampled(
                eval_input, batch_size=batch_size, seed=seed,
            )
        if mode == "full":
            return self.evaluate_full_ranking(
                eval_input, batch_size=batch_size,
            )
        raise ValueError(
            f"Unknown mode {mode!r}; use 'sampled' or 'full'"
        )

FullRankingEvaluator = TemporalSplitEvaluator  # backward-compat alias

def run_testpass() -> None:
    """Lightweight smoke test — safe on CPU with <200 MB."""
    print("Running evaluator testpass ...")

    n_eval = 1_000
    n_items = 5_000

    eval_input = EvalInput(
        user_embeddings=torch.randn(n_eval, EMBED_DIM),
        item_embeddings=torch.randn(n_items, EMBED_DIM),
        eval_user_ids=torch.arange(n_eval),
        ground_truth={i: i % n_items for i in range(n_eval)},
        exclude_items={
            i: [(i + 1) % n_items, (i + 2) % n_items]
            for i in range(n_eval)
        },
    )

    evaluator = TemporalSplitEvaluator(ks=[10, 20, 50], device="cpu")

    t0 = time.time()
    m_sampled = evaluator.evaluate(eval_input, mode="sampled")
    t_sampled = time.time() - t0

    print(f"\n{'Sampled (999 neg)':>25s}  ({t_sampled:.2f}s)")
    for k, v in m_sampled.items():
        print(f"  {k}: {v:.4f}")

    t0 = time.time()
    m_full = evaluator.evaluate(eval_input, mode="full", batch_size=512)
    t_full = time.time() - t0

    print(f"\n{'Full Ranking':>25s}  ({t_full:.2f}s)")
    for k, v in m_full.items():
        print(f"  {k}: {v:.4f}")

    del eval_input
    gc.collect()

    print("\nTestpass PASSED")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpass", action="store_true")
    args = parser.parse_args()
    if args.testpass:
        run_testpass()
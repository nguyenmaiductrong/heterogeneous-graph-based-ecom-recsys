import gc
import logging
import time

import numpy as np
import torch

from .contracts import EvalInput, EMBED_DIM

logger = logging.getLogger(__name__)


class TemporalSplitEvaluator:
    """Strict full-rank evaluator. No sampled-negatives mode.

    Metrics (multi-positive ground truth):
        HR@k    — 1 if any positive is in top-k after train-mask exclusion
        NDCG@k  — DCG over all positives in top-k, normalized by ideal DCG
    """

    def __init__(
        self,
        ks: list[int] | None = None,
        device: str = "cpu",
    ):
        self.ks = sorted(ks or [10, 20, 50])
        self.max_k = max(self.ks)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate_full_ranking_tiled(
        self,
        eval_input: EvalInput,
        user_batch: int = 512,
        item_tile: int = 16384,
    ) -> dict[str, float]:
        """Tiled full-rank scoring. Peak VRAM ~ O(user_batch * item_tile)."""
        eval_input.validate()
        n_eval = eval_input.eval_user_ids.size(0)
        n_items = eval_input.item_embeddings.size(0)
        device = self.device

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        item_embs = eval_input.item_embeddings.to(device=device, dtype=dtype)

        uid_to_pos = {int(u): i for i, u in enumerate(eval_input.eval_user_ids.tolist())}
        excl_rows: list[int] = []
        excl_cols: list[int] = []
        for uid, items in eval_input.exclude_items.items():
            pos = uid_to_pos.get(int(uid))
            if pos is None or not items:
                continue
            excl_rows.extend([pos] * len(items))
            excl_cols.extend(int(x) for x in items)
        excl_rows_t = torch.as_tensor(excl_rows, dtype=torch.long, device=device)
        excl_cols_t = torch.as_tensor(excl_cols, dtype=torch.long, device=device)

        gt_lists = [
            EvalInput._as_item_list(eval_input.ground_truth[int(u)])
            for u in eval_input.eval_user_ids.tolist()
        ]
        max_pos = max(len(items) for items in gt_lists)
        gt_padded = torch.full((n_eval, max_pos), -1, dtype=torch.long, device=device)
        gt_counts = torch.empty(n_eval, dtype=torch.long, device=device)
        for row, items in enumerate(gt_lists):
            gt_counts[row] = len(items)
            gt_padded[row, : len(items)] = torch.tensor(items, dtype=torch.long, device=device)

        ndcg_w = 1.0 / torch.log2(torch.arange(1, self.max_k + 1, device=device).float() + 1.0)
        sums: dict[str, float] = {}
        for k in self.ks:
            sums[f"HR@{k}"] = 0.0
            sums[f"NDCG@{k}"] = 0.0

        for u_start in range(0, n_eval, user_batch):
            u_end = min(u_start + user_batch, n_eval)
            B = u_end - u_start
            u_emb = eval_input.user_embeddings[u_start:u_end].to(
                device=device, dtype=dtype
            )

            top_vals = torch.full(
                (B, self.max_k), float("-inf"), device=device, dtype=dtype
            )
            top_idx = torch.full(
                (B, self.max_k), -1, device=device, dtype=torch.long
            )

            in_batch = (excl_rows_t >= u_start) & (excl_rows_t < u_end)
            sub_rows = excl_rows_t[in_batch] - u_start
            sub_cols = excl_cols_t[in_batch]

            for i_start in range(0, n_items, item_tile):
                i_end = min(i_start + item_tile, n_items)
                tile = item_embs[i_start:i_end]
                scores = u_emb @ tile.T

                tile_mask = (sub_cols >= i_start) & (sub_cols < i_end)
                if tile_mask.any():
                    scores[sub_rows[tile_mask], sub_cols[tile_mask] - i_start] = float("-inf")

                k = min(self.max_k, scores.size(1))
                tv, ti = scores.topk(k, dim=-1)
                ti = ti + i_start

                merged_v = torch.cat([top_vals, tv], dim=-1)
                merged_i = torch.cat([top_idx, ti], dim=-1)
                sel = merged_v.topk(self.max_k, dim=-1).indices
                top_vals = merged_v.gather(1, sel)
                top_idx = merged_i.gather(1, sel)

            hits = top_idx == gt_t[u_start:u_end].unsqueeze(1)
            for k in self.ks:
                sums[f"HR@{k}"] += hits[:, :k].any(dim=-1).float().sum().item()
                sums[f"NDCG@{k}"] += (hits[:, :k].float() * ndcg_w[:k]).sum().item()

            del u_emb, top_vals, top_idx, hits

        del item_embs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {k: v / n_eval for k, v in sums.items()}

    def evaluate(
        self,
        eval_input: EvalInput,
        batch_size: int = 512,
        mode: str | None = None,
        seed: int = 42,
    ) -> dict[str, float]:
        if mode is not None and mode not in ("full", "full_tiled"):
            logger.warning(
                "evaluator.evaluate(mode=%r) is ignored — only full-rank is supported.",
                mode,
            )
        return self.evaluate_full_ranking_tiled(eval_input, user_batch=batch_size)


FullRankingEvaluator = TemporalSplitEvaluator


def run_testpass() -> None:
    print("Running evaluator testpass (full-rank only) ...")
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
    m = evaluator.evaluate(eval_input)
    print(f"  Full-rank ({time.time() - t0:.2f}s)")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
    print("Testpass PASSED")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpass", action="store_true")
    args = parser.parse_args()
    if args.testpass:
        run_testpass()

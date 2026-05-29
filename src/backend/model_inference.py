from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path("checkpoints/downloaded/epoch_003.pt")


class EmbeddingModel:
    def __init__(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        self.checkpoint_path = checkpoint_path
        self.epoch = ckpt.get("epoch")
        self.metrics = ckpt.get("metrics", {})
        self.user = F.normalize(state["input_proj.user.weight"].float(), dim=1)
        self.product = F.normalize(state["input_proj.product.weight"].float(), dim=1)

    @property
    def n_users(self) -> int:
        return int(self.user.size(0))

    @property
    def n_products(self) -> int:
        return int(self.product.size(0))

    def user_vector(self, user_id: str) -> Optional[torch.Tensor]:
        if not user_id.startswith("u-"):
            return None
        try:
            user_idx = int(user_id.removeprefix("u-"))
        except ValueError:
            return None
        if 0 <= user_idx < self.n_users:
            return self.user[user_idx]
        return None

    def product_vectors(self, product_indices: list[int]) -> torch.Tensor:
        idx = torch.tensor(product_indices, dtype=torch.long)
        return self.product[idx]

    def product_vector(self, product_idx: int) -> torch.Tensor:
        return self.product[int(product_idx)]


@lru_cache(maxsize=1)
def load_embedding_model() -> EmbeddingModel:
    checkpoint = Path(os.environ.get("BPATMP_CHECKPOINT", DEFAULT_CHECKPOINT))
    return EmbeddingModel(checkpoint)


def model_available() -> bool:
    checkpoint = Path(os.environ.get("BPATMP_CHECKPOINT", DEFAULT_CHECKPOINT))
    return checkpoint.exists()

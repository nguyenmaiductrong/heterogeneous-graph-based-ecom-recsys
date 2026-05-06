"""Tang cuong do thi voi nhieu co gioi han theo behavior (BA-GraphAug).

eps_v = eps_min + (eps_max - eps_min) * sigmoid(a_0 + a_1/sqrt(1+n_purchase) + a_2/sqrt(1+degree) + a_3*n_tilde + a_4*H)
h_tilde = Normalize(h + eps_v * xi), xi ~ N(0,I), ||xi||=1

L_CL = InfoNCE giua 2 view tang cuong
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict


class BoundedBAGraphAug(nn.Module):
    """Tang cuong do thi voi nhieu co gioi han theo behavior.

    Do lon nhieu eps_v moi node nam trong [eps_min, eps_max]
    phu thuoc vao:
    - n_purchase: so luong purchase
    - degree: bac cua node
    - recency: thoi gian chuan hoa tu lan tuong tac cuoi
    - h_norm: proxy entropy cua embedding
    """

    def __init__(
        self,
        eps_min: float = 0.01,
        eps_max: float = 0.5,
        tau_cl: float = 0.2,
    ) -> None:
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.tau_cl = tau_cl

        self.a_0 = nn.Parameter(torch.zeros(1))
        self.raw_a1 = nn.Parameter(torch.zeros(1))
        self.raw_a2 = nn.Parameter(torch.zeros(1))
        self.raw_a3 = nn.Parameter(torch.zeros(1))
        self.raw_a4 = nn.Parameter(torch.zeros(1))

    def compute_eps(
        self,
        n_purchase: Tensor,
        degree: Tensor,
        n_tilde: Tensor,
        h_norm: Tensor,
    ) -> Tensor:
        """Compute per-node noise magnitude.

        Args:
            n_purchase: [N] purchase count per node
            degree: [N] node degree
            n_tilde: [N] normalized recency = log(1+dt_last)/log(1+dt_max)
            h_norm: [N] embedding norm proxy for entropy

        Returns:
            eps: [N] bounded noise magnitudes in [eps_min, eps_max]
        """
        a1 = F.softplus(self.raw_a1)
        a2 = F.softplus(self.raw_a2)
        a3 = F.softplus(self.raw_a3)
        a4 = F.softplus(self.raw_a4)

        logit = (
            self.a_0
            + a1 / torch.sqrt(1.0 + n_purchase.float())
            + a2 / torch.sqrt(1.0 + degree.float())
            + a3 * n_tilde.float()
            + a4 * h_norm.float()
        )

        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(logit)
        return eps

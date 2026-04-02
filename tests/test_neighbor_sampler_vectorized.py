from __future__ import annotations

import time
import pytest
import torch

from src.core.contracts import BEHAVIOR_TO_ID
from src.graph.neighbor_sampler import (
    BehaviorAwareNeighborSampler,
    NeighborSamplerConfig,
    _batch_sample_csr,
)


def _make_toy_graph(device: torch.device) -> dict[tuple[str, str, str], torch.Tensor]:
    """100 users, 200 products, 10 categories. Trace path: U0 -purchase-> P5 -> C2."""
    nu, np_, nc = 100, 200, 10

    def empty_2() -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # User 0: no view/cart edges; purchase -> product 5 only
    ei_view = empty_2()
    ei_cart = empty_2()
    ei_purchase = torch.tensor([[0], [5]], dtype=torch.long, device=device)

    # Other users: random-ish edges so CSR is non-trivial (not required for U0 trace)
    u_other = torch.randint(1, nu, (400,), device=device)
    p_other = torch.randint(0, np_, (400,), device=device)
    ei_view = torch.cat([ei_view, torch.stack([u_other[:200], p_other[:200]])], dim=1)
    ei_cart = torch.cat([ei_cart, torch.stack([u_other[200:350], p_other[200:350]])], dim=1)
    extra_p = torch.stack(
        [torch.randint(1, nu, (80,), device=device), torch.randint(0, np_, (80,), device=device)]
    )
    ei_purchase = torch.cat([ei_purchase, extra_p], dim=1)

    # Product 5 -> category 2
    ei_pc = torch.tensor([[5], [2]], dtype=torch.long, device=device)

    return {
        ("user", "view", "product"): ei_view,
        ("user", "cart", "product"): ei_cart,
        ("user", "purchase", "product"): ei_purchase,
        ("product", "belongs_to", "category"): ei_pc,
    }


def test_purchase_origin_propagates_to_category() -> None:
    """User_0 ->[purchase]-> P5 ->[belongs_to]-> C2: category edge tag must be PURCHASE."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    gen = torch.Generator(device=device).manual_seed(42)

    edges = _make_toy_graph(device)
    cfg = NeighborSamplerConfig(hop1_budget=15, hop2_budget=10, hop1_sample_replace=False)
    sampler = BehaviorAwareNeighborSampler(edge_index_dict=edges, num_nodes_dict=None, config=cfg)

    data = sampler.sample(torch.tensor([0], dtype=torch.long), seed_type="user", generator=gen)

    pc = data["product", "belongs_to", "category"].edge_index
    tags = data["product", "belongs_to", "category"].edge_attr.view(-1)
    px = data["product"].x
    cx = data["category"].x

    # Find edge (global product 5, global category 2)
    p5_local = (px == 5).nonzero(as_tuple=True)[0]
    c2_local = (cx == 2).nonzero(as_tuple=True)[0]
    assert p5_local.numel() >= 1 and c2_local.numel() >= 1
    pl = int(p5_local[0].item())
    cl = int(c2_local[0].item())

    match = ((pc[0] == pl) & (pc[1] == cl)).nonzero(as_tuple=True)[0]
    assert match.numel() >= 1
    eid = int(match[0].item())
    assert int(tags[eid].item()) == BEHAVIOR_TO_ID["purchase"]


def test_batch_sample_csr_shape_and_mask() -> None:
    device = torch.device("cpu")
    ptr = torch.tensor([0, 2, 2, 5], dtype=torch.long, device=device)
    cols = torch.tensor([1, 3, 10, 11, 12], dtype=torch.long, device=device)
    seeds = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    s, v = _batch_sample_csr(ptr, cols, seeds, 4, None, replace=True)
    assert s.shape == (3, 4)
    assert v.shape == (3, 4)
    assert not v[1].any()


@pytest.mark.slow
def test_benchmark_vectorized_vs_legacy() -> None:
    device = torch.device("cpu")
    edges = _make_toy_graph(device)
    cfg = NeighborSamplerConfig(hop1_budget=15, hop2_budget=10, hop1_sample_replace=False)
    sampler = BehaviorAwareNeighborSampler(edge_index_dict=edges, config=cfg)
    gen = torch.Generator(device=device).manual_seed(0)

    n_new = 10_000
    seeds = torch.arange(n_new, dtype=torch.long, device=device) % 100

    t0 = time.perf_counter()
    _ = sampler._sample_user_seeds(seeds, gen)
    t_new = time.perf_counter() - t0

    seeds_l = seeds.clone()
    gen2 = torch.Generator(device=device).manual_seed(0)
    t1 = time.perf_counter()
    _ = sampler._sample_user_seeds_legacy(seeds_l, gen2)
    t_old = time.perf_counter() - t1

    ratio = t_old / t_new if t_new > 0 else float("inf")
    print(f"\n[benchmark] vectorized {n_new} seeds: {t_new:.4f}s")
    print(f"[benchmark] legacy     {n_new} seeds: {t_old:.4f}s")
    print(f"[benchmark] speedup ratio (old/new): {ratio:.2f}x")

    assert t_new > 0 and t_old > 0
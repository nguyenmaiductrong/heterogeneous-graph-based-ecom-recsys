
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from src.core.contracts import BEHAVIOR_TO_ID, BEHAVIOR_TYPES, RELATION_TYPES

_USER_BEHAVIOR_RELS: tuple[tuple[str, str, str], ...] = (
    ("user", "view", "product"),
    ("user", "cart", "product"),
    ("user", "purchase", "product"),
)
_PRODUCT_TO_CATEGORY: tuple[str, str, str] = ("product", "belongs_to", "category")
_PRODUCT_TO_BRAND: tuple[str, str, str] = ("product", "producedBy", "brand")

_RELATION_ALIASES: dict[str, tuple[str, str, str]] = {
    "belongs_to": _PRODUCT_TO_CATEGORY,
    "belongsTo": _PRODUCT_TO_CATEGORY,
    "category": _PRODUCT_TO_CATEGORY,
    "producedBy": _PRODUCT_TO_BRAND,
    "brand": _PRODUCT_TO_BRAND,
}


def _edge_index_to_csr(
    edge_index: Tensor,
    num_src: int,
    *,
    dedupe: bool,
) -> tuple[Tensor, Tensor]:
    if edge_index.numel() == 0:
        ptr = torch.zeros(num_src + 1, dtype=torch.long, device=edge_index.device)
        return ptr, edge_index.new_empty((0,), dtype=torch.long)

    row = edge_index[0].long()
    col = edge_index[1].long()
    if dedupe:
        pairs = torch.stack([row, col], dim=1)
        pairs = torch.unique(pairs, dim=0, sorted=True)
        row, col = pairs[:, 0], pairs[:, 1]
    else:
        order = torch.argsort(row)
        row = row[order]
        col = col[order]

    counts = torch.bincount(row, minlength=num_src)
    ptr = torch.zeros(num_src + 1, dtype=torch.long, device=edge_index.device)
    ptr[1:] = counts.cumsum(dim=0)
    return ptr, col


def _infer_num_nodes(
    edge_index_dict: dict[tuple[str, str, str], Tensor],
    override: dict[str, int] | None,
) -> dict[str, int]:
    if override is not None:
        return dict(override)
    counts: dict[str, int] = {}
    for (_src, _r, _dst), ei in edge_index_dict.items():
        if ei.numel() == 0:
            continue
        counts[_src] = max(counts.get(_src, 0), int(ei[0].max().item()) + 1)
        counts[_dst] = max(counts.get(_dst, 0), int(ei[1].max().item()) + 1)
    return counts

def _batch_sample_csr(
    ptr: Tensor,
    cols: Tensor,
    seeds: Tensor,
    num_samples: int,
    generator: torch.Generator | None,
    *,
    replace: bool = True,
) -> tuple[Tensor, Tensor]:

    device = ptr.device
    seeds = seeds.long().view(-1)
    bsz = seeds.numel()
    lo = ptr[seeds]
    hi = ptr[seeds + 1]
    deg = hi - lo

    out = torch.full((bsz, num_samples), -1, dtype=torch.long, device=device)
    valid = torch.zeros((bsz, num_samples), dtype=torch.bool, device=device)

    zero_deg = deg == 0
    if zero_deg.all():
        return out, valid

    if replace:
        rnd = torch.rand(bsz, num_samples, generator=generator, device=device, dtype=torch.float32)
        safe_deg = deg.clamp(min=1)
        off = (rnd * safe_deg.unsqueeze(1)).floor().long()
        off = off.clamp(max=safe_deg.unsqueeze(1) - 1)
        off = off.masked_fill(zero_deg.unsqueeze(1), 0)
        col_idx = lo.unsqueeze(1) + off
        
        col_idx = col_idx.clamp(max=cols.size(0) - 1)
        
        out = cols[col_idx].masked_fill(zero_deg.unsqueeze(1), -1)
        valid = (~zero_deg.unsqueeze(1)).expand(bsz, num_samples)
        return out, valid

    for i in range(bsz):
        d = int(deg[i].item())
        if d == 0:
            continue
            
        sample_size = min(num_samples, d)
        
        if generator is None:
            perm = torch.randperm(d, device=device)[:sample_size]
        else:
            perm = torch.randperm(d, generator=generator, device=device)[:sample_size]
            
        c_idx = lo[i] + perm
        out[i, :sample_size] = cols[c_idx]
        valid[i, :sample_size] = True

    return out, valid

def _sample_without_replacement(
    pool: Tensor,
    k: int,
    generator: torch.Generator | None,
) -> Tensor:
    n = pool.numel()
    if n == 0:
        return pool
    if k >= n:
        return pool.clone()
    device = pool.device
    if generator is None:
        perm = torch.randperm(n, device=device)
    else:
        perm = torch.randperm(n, device=device, generator=generator)
    return pool[perm[:k]]


@dataclass
class NeighborSamplerConfig:
    hop1_budget: int = 15
    hop2_budget: int = 10
    dedupe_csr: bool = False  # Default to False to prevent OOM on 204M edges
    hop1_sample_replace: bool = False

class BehaviorAwareNeighborSampler:
    def __init__(
        self,
        edge_index_dict: dict[tuple[str, str, str], Tensor] | None = None,
        data: HeteroData | None = None,
        num_nodes_dict: dict[str, int] | None = None,
        *,
        config: NeighborSamplerConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        if (edge_index_dict is None) == (data is None):
            raise ValueError("Provide exactly one of edge_index_dict or data.")

        self._cfg = config or NeighborSamplerConfig()

        if data is not None:
            edge_index_dict = {e: data[e].edge_index.clone() for e in data.edge_types}

        assert edge_index_dict is not None
        dev = device or next(iter(edge_index_dict.values())).device
        self._device = dev
        self._edge_index_dict = {
            k: v.to(device=dev, dtype=torch.long) for k, v in edge_index_dict.items()
        }

        self._num_nodes = _infer_num_nodes(self._edge_index_dict, num_nodes_dict)

        for er in _USER_BEHAVIOR_RELS:
            if er not in self._edge_index_dict:
                raise KeyError(f"Missing required edge type {er}.")
        if _PRODUCT_TO_CATEGORY not in self._edge_index_dict:
            raise KeyError(f"Missing required edge type {_PRODUCT_TO_CATEGORY}.")

        self._csr: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}
        for key, ei in self._edge_index_dict.items():
            src_type = key[0]
            n_src = self._num_nodes.get(src_type, 0)
            if ei.numel() > 0:
                n_src = max(n_src, int(ei[0].max().item()) + 1)
                self._num_nodes[src_type] = max(self._num_nodes.get(src_type, 0), n_src)
            ptr, cols = _edge_index_to_csr(ei, n_src, dedupe=self._cfg.dedupe_csr)
            self._csr[key] = (ptr, cols)

    @property
    def num_nodes_dict(self) -> dict[str, int]:
        return dict(self._num_nodes)

    def _has_csr(self, key: tuple[str, str, str]) -> bool:
        return key in self._csr

    def _vectorized_hop1(
        self,
        user_seeds: Tensor,
        behavior_type: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rel = ("user", behavior_type, "product")
        ptr, cols = self._csr[rel]
        bsz = user_seeds.numel()
        h1 = self._cfg.hop1_budget
        sampled, valid = _batch_sample_csr(
            ptr,
            cols,
            user_seeds,
            h1,
            generator,
            replace=self._cfg.hop1_sample_replace,
        )
        bid = BEHAVIOR_TO_ID[behavior_type]
        origin_tags = torch.full((bsz * h1,), bid, dtype=torch.int8, device=self._device)
        return sampled, valid, origin_tags


    def _vectorized_hop2(
        self,
        product_nodes: Tensor,
        origin_tags: Tensor,
        relation_type: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = _RELATION_ALIASES.get(relation_type)
        if key is None:
            raise ValueError(f"Unknown relation_type: {relation_type}")

        n = product_nodes.numel()
        h2 = self._cfg.hop2_budget

        if key not in self._csr:
            return (
                torch.full((n, h2), -1, dtype=torch.long, device=self._device),
                torch.zeros((n, h2), dtype=torch.bool, device=self._device),
                torch.full((n, h2), -1, dtype=torch.int8, device=self._device),
            )

        ptr, cols = self._csr[key]

        valid_seed = product_nodes >= 0
        safe_seeds = torch.where(
            valid_seed, product_nodes, torch.zeros_like(product_nodes)
        )

        sampled, vmask = _batch_sample_csr(
            ptr,
            cols,
            safe_seeds,
            h2,
            generator,
            replace=True,
        )

        sampled = torch.where(valid_seed.unsqueeze(1), sampled, torch.full_like(sampled, -1))
        vmask = vmask & valid_seed.unsqueeze(1)

        dev = product_nodes.device
        ot = origin_tags.to(device=dev, dtype=torch.int8).view(n, 1).expand(n, h2).clone()
        ot = torch.where(vmask, ot, torch.full_like(ot, -1))
        return sampled, vmask, ot

    def _process_hop2(
        self,
        pg_all: Tensor,
        bi_all: Tensor,
        inv_p: Tensor,
        origin_flat: Tensor,
        relation: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        _empty1 = torch.empty((0,), dtype=torch.long, device=self._device)

        sampled, vmask, orig = self._vectorized_hop2(
            pg_all, origin_flat, relation, generator,
        )

        ridx, cidx = vmask.nonzero(as_tuple=True)
        if ridx.numel() == 0:
            return _empty1, _empty1, _empty1, _empty1, _empty1

        ploc = inv_p[ridx]
        ng = sampled[ridx, cidx]
        bib = bi_all[ridx]
        tag = orig[ridx, cidx].long()

        keys = torch.stack([ng, bib], dim=1)
        uniq, inv = torch.unique(keys, dim=0, return_inverse=True)

        return uniq[:, 0], uniq[:, 1], ploc, inv, tag

    def sample(
        self,
        seed_indices: Tensor,
        *,
        seed_type: str = "user",
        generator: torch.Generator | None = None,
    ) -> HeteroData:
        seeds = seed_indices.long().view(-1).to(self._device)
        if seeds.numel() == 0:
            return _empty_hetero(self._device)

        if seed_type == "user":
            return self._sample_user_seeds(seeds, generator)
        if seed_type == "product":
            return self._sample_product_seeds_vectorized(seeds, generator)
        raise ValueError(f"Unknown seed_type: {seed_type}")

    def _sample_user_seeds(
        self,
        user_seeds: Tensor,
        generator: torch.Generator | None,
    ) -> HeteroData:
        bsz = user_seeds.numel()
        h1 = self._cfg.hop1_budget
        device = self._device
        _empty2 = torch.empty((2, 0), dtype=torch.long, device=device)
        _empty1 = torch.empty((0,), dtype=torch.long, device=device)

        bi_broadcast = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1).expand(bsz, h1)

        hop1: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        bi_chunks: list[Tensor] = []
        pg_chunks: list[Tensor] = []
        origin_parts: list[Tensor] = []

        for beh in BEHAVIOR_TYPES:
            p, m, _ = self._vectorized_hop1(user_seeds, beh, generator)
            bi_flat = bi_broadcast[m]
            pg_flat = p[m]
            hop1[beh] = (p, m, bi_flat, pg_flat)
            if pg_flat.numel() > 0:
                bi_chunks.append(bi_flat)
                pg_chunks.append(pg_flat)
                origin_parts.append(
                    torch.full(
                        (pg_flat.size(0),), BEHAVIOR_TO_ID[beh],
                        dtype=torch.int8, device=device,
                    )
                )

        user_x = user_seeds
        user_b = torch.arange(bsz, device=device, dtype=torch.long)

        if not bi_chunks:
            node_data: dict[str, tuple[Tensor, Tensor]] = {
                "user": (user_x, user_b),
                "product": (_empty1, _empty1),
                "category": (_empty1, _empty1),
                "brand": (_empty1, _empty1),
            }
            edge_data: dict[tuple[str, str, str], tuple[Tensor, Tensor | None]] = {
                rel: (_empty2, None) for rel in RELATION_TYPES
            }
            return _pack_hetero_subgraph(device, node_data, edge_data)

        bi_all = torch.cat(bi_chunks, dim=0)
        pg_all = torch.cat(pg_chunks, dim=0)
        origin_flat = torch.cat(origin_parts, dim=0)

        keys_p = torch.stack([pg_all, bi_all], dim=1)
        uniq_p, inv_p = torch.unique(keys_p, dim=0, return_inverse=True)
        prod_x = uniq_p[:, 0]
        prod_b = uniq_p[:, 1]

        beh_edges: dict[str, list[Tensor]] = {"view": [], "cart": [], "purchase": []}
        offset = 0
        for beh in BEHAVIOR_TYPES:
            _, _, bi_flat, pg_flat = hop1[beh]
            n_e = bi_flat.numel()
            if n_e == 0:
                continue
            ploc = inv_p[offset: offset + n_e]
            offset += n_e
            beh_edges[beh].append(torch.stack([bi_flat, ploc], dim=0))

        def _cat_e(parts: list[Tensor]) -> Tensor:
            return torch.cat(parts, dim=1) if parts else _empty2

        e_view = _cat_e(beh_edges["view"])
        e_cart = _cat_e(beh_edges["cart"])
        e_purchase = _cat_e(beh_edges["purchase"])

        e_rev_view = e_view.flip(0) if e_view.size(1) > 0 else _empty2
        e_rev_cart = e_cart.flip(0) if e_cart.size(1) > 0 else _empty2
        e_rev_purchase = e_purchase.flip(0) if e_purchase.size(1) > 0 else _empty2

        cat_x, cat_b, pc_src, pc_dst, pc_tag = self._process_hop2(
            pg_all, bi_all, inv_p, origin_flat, "belongs_to", generator,
        ) 

        brand_x, brand_b, pb_src, pb_dst, pb_tag = self._process_hop2(
            pg_all, bi_all, inv_p, origin_flat, "brand", generator,
        )

        def _ei(src: Tensor, dst: Tensor) -> Tensor:
            if src.numel() == 0:
                return _empty2
            return torch.stack([src, dst], dim=0)

        def _attr(tag: Tensor) -> Tensor | None:
            if tag.numel() == 0:
                return None
            return tag.view(-1, 1)

        node_data = {
            "user": (user_x, user_b),
            "product": (prod_x, prod_b),
            "category": (cat_x, cat_b),
            "brand": (brand_x, brand_b),
        }

        edge_data = {
            ("user", "view", "product"): (e_view, None),
            ("user", "cart", "product"): (e_cart, None),
            ("user", "purchase", "product"): (e_purchase, None),
            ("product", "rev_view", "user"): (e_rev_view, None),
            ("product", "rev_cart", "user"): (e_rev_cart, None),
            ("product", "rev_purchase", "user"): (e_rev_purchase, None),
            ("product", "belongs_to", "category"): (_ei(pc_src, pc_dst), _attr(pc_tag)),
            ("category", "contains", "product"): (_ei(pc_dst, pc_src), _attr(pc_tag)),
            ("product", "producedBy", "brand"): (_ei(pb_src, pb_dst), _attr(pb_tag)),
            ("brand", "brands", "product"): (_ei(pb_dst, pb_src), _attr(pb_tag)),
        }

        return _pack_hetero_subgraph(device, node_data, edge_data)


    def _sample_user_seeds_legacy(
        self,
        user_seeds: Tensor,
        generator: torch.Generator | None,
    ) -> HeteroData:
        bsz = user_seeds.numel()
        h1, h2 = self._cfg.hop1_budget, self._cfg.hop2_budget
        device = self._device
        _empty2 = torch.empty((2, 0), dtype=torch.long, device=device)
        _empty1 = torch.empty((0,), dtype=torch.long, device=device)

        user_rows: list[int] = []
        user_batch: list[int] = []
        prod_rows: list[int] = []
        prod_batch: list[int] = []
        cat_rows: list[int] = []
        cat_batch: list[int] = []
        brand_rows: list[int] = []
        brand_batch: list[int] = []

        e_view: list[tuple[int, int]] = []
        e_cart: list[tuple[int, int]] = []
        e_purchase: list[tuple[int, int]] = []
        e_pc: list[tuple[int, int, int]] = []
        e_pb: list[tuple[int, int, int]] = []

        user_g2l: dict[tuple[int, int], int] = {}
        prod_g2l: dict[tuple[int, int], int] = {}
        cat_g2l: dict[tuple[int, int], int] = {}
        brand_g2l: dict[tuple[int, int], int] = {}

        def ensure_user(ug: int, bi: int) -> int:
            k = (ug, bi)
            if k not in user_g2l:
                user_g2l[k] = len(user_rows)
                user_rows.append(ug)
                user_batch.append(bi)
            return user_g2l[k]

        def ensure_product(pg: int, bi: int) -> int:
            k = (pg, bi)
            if k not in prod_g2l:
                prod_g2l[k] = len(prod_rows)
                prod_rows.append(pg)
                prod_batch.append(bi)
            return prod_g2l[k]

        def ensure_category(cg: int, bi: int) -> int:
            k = (cg, bi)
            if k not in cat_g2l:
                cat_g2l[k] = len(cat_rows)
                cat_rows.append(cg)
                cat_batch.append(bi)
            return cat_g2l[k]

        def ensure_brand(bg: int, bi: int) -> int:
            k = (bg, bi)
            if k not in brand_g2l:
                brand_g2l[k] = len(brand_rows)
                brand_rows.append(bg)
                brand_batch.append(bi)
            return brand_g2l[k]

        beh_edges = {"view": e_view, "cart": e_cart, "purchase": e_purchase}
        produced_by_csr = self._has_csr(_PRODUCT_TO_BRAND)

        for bi in range(bsz):
            u_global = int(user_seeds[bi].item())
            ul = ensure_user(u_global, bi)

            for beh in BEHAVIOR_TYPES:
                rel = ("user", beh, "product")
                ptr, cols = self._csr[rel]
                lo = int(ptr[u_global].item())
                hi = int(ptr[u_global + 1].item())
                neigh = cols[lo:hi]
                picked = _sample_without_replacement(neigh, h1, generator)
                bid = BEHAVIOR_TO_ID[beh]

                for pg in picked.tolist():
                    pl = ensure_product(pg, bi)
                    beh_edges[beh].append((ul, pl))

                    p_ptr, p_cols = self._csr[_PRODUCT_TO_CATEGORY]
                    lo2 = int(p_ptr[pg].item())
                    hi2 = int(p_ptr[pg + 1].item())
                    c_neigh = p_cols[lo2:hi2]
                    c_picked = _sample_without_replacement(c_neigh, h2, generator)
                    for cg in c_picked.tolist():
                        cl = ensure_category(cg, bi)
                        e_pc.append((pl, cl, bid))

                    if produced_by_csr:
                        b_ptr, b_cols = self._csr[_PRODUCT_TO_BRAND]
                        lo3 = int(b_ptr[pg].item())
                        hi3 = int(b_ptr[pg + 1].item())
                        b_neigh = b_cols[lo3:hi3]
                        b_picked = _sample_without_replacement(b_neigh, h2, generator)
                        for bg in b_picked.tolist():
                            bl = ensure_brand(bg, bi)
                            e_pb.append((pl, bl, bid))

        def _l2t(lst: list[int]) -> Tensor:
            if not lst:
                return _empty1
            return torch.tensor(lst, dtype=torch.long, device=device)

        def _pairs_ei(pairs: list[tuple[int, int]]) -> Tensor:
            if not pairs:
                return _empty2
            return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

        def _triples_unpack(
            triples: list[tuple[int, int, int]],
        ) -> tuple[Tensor, Tensor, Tensor]:
            if not triples:
                return _empty1, _empty1, _empty1
            src = torch.tensor([a for a, _, _ in triples], dtype=torch.long, device=device)
            dst = torch.tensor([b for _, b, _ in triples], dtype=torch.long, device=device)
            tag = torch.tensor([c for _, _, c in triples], dtype=torch.long, device=device)
            return src, dst, tag

        ev = _pairs_ei(e_view)
        ec = _pairs_ei(e_cart)
        ep = _pairs_ei(e_purchase)
        pc_s, pc_d, pc_t = _triples_unpack(e_pc)
        pb_s, pb_d, pb_t = _triples_unpack(e_pb)

        def _ei(src: Tensor, dst: Tensor) -> Tensor:
            if src.numel() == 0:
                return _empty2
            return torch.stack([src, dst], dim=0)

        def _attr(tag: Tensor) -> Tensor | None:
            if tag.numel() == 0:
                return None
            return tag.view(-1, 1)

        node_data = {
            "user": (_l2t(user_rows), _l2t(user_batch)),
            "product": (_l2t(prod_rows), _l2t(prod_batch)),
            "category": (_l2t(cat_rows), _l2t(cat_batch)),
            "brand": (_l2t(brand_rows), _l2t(brand_batch)),
        }
        edge_data: dict[tuple[str, str, str], tuple[Tensor, Tensor | None]] = {
            ("user", "view", "product"): (ev, None),
            ("user", "cart", "product"): (ec, None),
            ("user", "purchase", "product"): (ep, None),
            ("product", "rev_view", "user"): (ev.flip(0) if ev.size(1) > 0 else _empty2, None),
            ("product", "rev_cart", "user"): (ec.flip(0) if ec.size(1) > 0 else _empty2, None),
            ("product", "rev_purchase", "user"): (ep.flip(0) if ep.size(1) > 0 else _empty2, None),
            ("product", "belongs_to", "category"): (_ei(pc_s, pc_d), _attr(pc_t)),
            ("category", "contains", "product"): (_ei(pc_d, pc_s), _attr(pc_t)),
            ("product", "producedBy", "brand"): (_ei(pb_s, pb_d), _attr(pb_t)),
            ("brand", "brands", "product"): (_ei(pb_d, pb_s), _attr(pb_t)),
        }

        return _pack_hetero_subgraph(device, node_data, edge_data)


    def _sample_product_seeds_vectorized(
        self,
        product_seeds: Tensor,
        generator: torch.Generator | None,
    ) -> HeteroData:
        bsz = product_seeds.numel()
        bid = BEHAVIOR_TO_ID["purchase"]
        device = self._device
        _empty2 = torch.empty((2, 0), dtype=torch.long, device=device)
        _empty1 = torch.empty((0,), dtype=torch.long, device=device)

        prod_x = product_seeds
        prod_b = torch.arange(bsz, device=device, dtype=torch.long)
        inv_p = torch.arange(bsz, device=device, dtype=torch.long)
        origin_flat = torch.full((bsz,), bid, dtype=torch.int8, device=device)

        cat_x, cat_b, pc_src, pc_dst, pc_tag = self._process_hop2(
            product_seeds, prod_b, inv_p, origin_flat, "belongs_to", generator,
        )
        brand_x, brand_b, pb_src, pb_dst, pb_tag = self._process_hop2(
            product_seeds, prod_b, inv_p, origin_flat, "brand", generator,
        )

        def _ei(src: Tensor, dst: Tensor) -> Tensor:
            if src.numel() == 0:
                return _empty2
            return torch.stack([src, dst], dim=0)

        def _attr(tag: Tensor) -> Tensor | None:
            if tag.numel() == 0:
                return None
            return tag.view(-1, 1)

        node_data = {
            "user": (_empty1, _empty1),
            "product": (prod_x, prod_b),
            "category": (cat_x, cat_b),
            "brand": (brand_x, brand_b),
        }
        edge_data: dict[tuple[str, str, str], tuple[Tensor, Tensor | None]] = {
            ("user", "view", "product"): (_empty2, None),
            ("user", "cart", "product"): (_empty2, None),
            ("user", "purchase", "product"): (_empty2, None),
            ("product", "rev_view", "user"): (_empty2, None),
            ("product", "rev_cart", "user"): (_empty2, None),
            ("product", "rev_purchase", "user"): (_empty2, None),
            ("product", "belongs_to", "category"): (_ei(pc_src, pc_dst), _attr(pc_tag)),
            ("category", "contains", "product"): (_ei(pc_dst, pc_src), _attr(pc_tag)),
            ("product", "producedBy", "brand"): (_ei(pb_src, pb_dst), _attr(pb_tag)),
            ("brand", "brands", "product"): (_ei(pb_dst, pb_src), _attr(pb_tag)),
        }

        return _pack_hetero_subgraph(device, node_data, edge_data)


def _empty_hetero(device: torch.device) -> HeteroData:
    z = torch.empty(0, dtype=torch.long, device=device)
    node_data: dict[str, tuple[Tensor, Tensor]] = {
        nt: (z.clone(), z.clone()) for nt in ("user", "product", "category", "brand")
    }
    _e2 = torch.empty((2, 0), dtype=torch.long, device=device)
    edge_data: dict[tuple[str, str, str], tuple[Tensor, Tensor | None]] = {
        rel: (_e2.clone(), None) for rel in RELATION_TYPES
    }
    return _pack_hetero_subgraph(device, node_data, edge_data)


def _pack_hetero_subgraph(
    device: torch.device,
    node_data: dict[str, tuple[Tensor, Tensor]],
    edge_data: dict[tuple[str, str, str], tuple[Tensor, Tensor | None]],
) -> HeteroData:
    data = HeteroData()
    _e2 = torch.empty((2, 0), dtype=torch.long, device=device)

    for ntype, (x, batch) in node_data.items():
        data[ntype].x = x.to(device=device, dtype=torch.long)
        data[ntype].batch = batch.to(device=device, dtype=torch.long)
        data[ntype].num_nodes = x.size(0)

    for rel, (ei, attr) in edge_data.items():
        if ei is None or ei.numel() == 0:
            data[rel].edge_index = _e2.clone()
        else:
            data[rel].edge_index = ei.to(device=device, dtype=torch.long)

        if attr is not None and attr.numel() > 0:
            data[rel].edge_attr = attr.to(device=device)

    return data


def collate_hetero_subgraphs(batch: list[HeteroData]) -> Batch:
    return Batch.from_data_list(batch)


HeteroNeighborSampler = BehaviorAwareNeighborSampler
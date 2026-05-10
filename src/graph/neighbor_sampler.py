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
    ptr, cols, _ = _edge_index_to_csr_with_values(
        edge_index,
        num_src,
        dedupe=dedupe,
        edge_values=None,
    )
    return ptr, cols


def _edge_index_to_csr_with_values(
    edge_index: Tensor,
    num_src: int,
    *,
    dedupe: bool,
    edge_values: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    device = edge_index.device
    if edge_index.numel() == 0:
        ptr = torch.zeros(num_src + 1, dtype=torch.long, device=device)
        empty_values = None
        if edge_values is not None:
            empty_values = edge_values.new_empty((0,), dtype=edge_values.dtype)
        return ptr, edge_index.new_empty((0,), dtype=torch.long), empty_values

    row = edge_index[0].long().cpu()
    col = edge_index[1].long().cpu()
    values_cpu = edge_values.detach().cpu().view(-1) if edge_values is not None else None
    if dedupe:
        if values_cpu is not None:
            raise ValueError("dedupe=True is not supported for timestamped CSR edges")
        pairs = torch.stack([row, col], dim=1)
        pairs = torch.unique(pairs, dim=0, sorted=True)
        row, col = pairs[:, 0], pairs[:, 1]
    else:
        order = torch.argsort(row)
        row = row[order]
        col = col[order]
        if values_cpu is not None:
            values_cpu = values_cpu[order]

    counts = torch.bincount(row, minlength=num_src)
    ptr = torch.zeros(num_src + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(dim=0)
    values = values_cpu.to(device) if values_cpu is not None else None
    return ptr.to(device), col.to(device), values


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
    return_positions: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:

    device = ptr.device
    seeds = seeds.long().view(-1)
    bsz = seeds.numel()
    lo = ptr[seeds]
    hi = ptr[seeds + 1]
    deg = hi - lo

    out = torch.full((bsz, num_samples), -1, dtype=torch.long, device=device)
    valid = torch.zeros((bsz, num_samples), dtype=torch.bool, device=device)
    pos_out = torch.full((bsz, num_samples), -1, dtype=torch.long, device=device)

    zero_deg = deg == 0
    if zero_deg.all():
        return (out, valid, pos_out) if return_positions else (out, valid)

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
        pos_out = col_idx.masked_fill(zero_deg.unsqueeze(1), -1)
        return (out, valid, pos_out) if return_positions else (out, valid)

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
        pos_out[i, :sample_size] = c_idx

    return (out, valid, pos_out) if return_positions else (out, valid)


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
            edge_ts_dict = {}
            for e in data.edge_types:
                edge_ts = getattr(data[e], "edge_ts", None)
                if edge_ts is None:
                    edge_ts = getattr(data[e], "edge_time", None)
                if edge_ts is not None and edge_ts.numel() > 0:
                    edge_ts_dict[e] = edge_ts.clone()
        else:
            edge_ts_dict = {}

        assert edge_index_dict is not None
        dev = device or next(iter(edge_index_dict.values())).device
        self._device = dev
        self._edge_index_dict = {
            k: v.to(device=dev, dtype=torch.long) for k, v in edge_index_dict.items()
        }
        self._edge_ts_dict = {
            k: v.to(device=dev, dtype=torch.long).view(-1) for k, v in edge_ts_dict.items()
        }

        self._num_nodes = _infer_num_nodes(self._edge_index_dict, num_nodes_dict)

        for er in _USER_BEHAVIOR_RELS:
            if er not in self._edge_index_dict:
                raise KeyError(f"Missing required edge type {er}.")
        if _PRODUCT_TO_CATEGORY not in self._edge_index_dict:
            raise KeyError(f"Missing required edge type {_PRODUCT_TO_CATEGORY}.")

        self._csr: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}
        self._csr_edge_ts: dict[tuple[str, str, str], Tensor] = {}
        for key, ei in self._edge_index_dict.items():
            src_type = key[0]
            n_src = self._num_nodes.get(src_type, 0)
            if ei.numel() > 0:
                n_src = max(n_src, int(ei[0].max().item()) + 1)
                self._num_nodes[src_type] = max(self._num_nodes.get(src_type, 0), n_src)
            edge_ts = self._edge_ts_dict.get(key)
            ptr, cols, ts_vals = _edge_index_to_csr_with_values(
                ei,
                n_src,
                dedupe=self._cfg.dedupe_csr,
                edge_values=edge_ts,
            )
            self._csr[key] = (ptr, cols)
            if ts_vals is not None:
                self._csr_edge_ts[key] = ts_vals

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        rel = ("user", behavior_type, "product")
        ptr, cols = self._csr[rel]
        bsz = user_seeds.numel()
        h1 = self._cfg.hop1_budget
        sampled, valid, positions = _batch_sample_csr(
            ptr,
            cols,
            user_seeds,
            h1,
            generator,
            replace=self._cfg.hop1_sample_replace,
            return_positions=True,
        )
        bid = BEHAVIOR_TO_ID[behavior_type]
        origin_tags = torch.full((bsz * h1,), bid, dtype=torch.int8, device=self._device)
        edge_ts = None
        ts_vals = self._csr_edge_ts.get(rel)
        if ts_vals is not None:
            safe_pos = positions.clamp(min=0)
            edge_ts = ts_vals[safe_pos].masked_fill(~valid, -1)
        return sampled, valid, origin_tags, edge_ts

    def _vectorized_hop2(
        self,
        product_nodes: Tensor,
        origin_tags: Tensor,
        origin_ts: Tensor | None,
        relation_type: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
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
                None,
            )

        ptr, cols = self._csr[key]

        valid_seed = product_nodes >= 0
        safe_seeds = torch.where(valid_seed, product_nodes, torch.zeros_like(product_nodes))

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
        ts_out = None
        if origin_ts is not None:
            ts_out = origin_ts.to(device=dev, dtype=torch.long).view(n, 1).expand(n, h2).clone()
            ts_out = torch.where(vmask, ts_out, torch.full_like(ts_out, -1))
        return sampled, vmask, ot, ts_out

    def _process_hop2(
        self,
        pg_all: Tensor,
        bi_all: Tensor,
        inv_p: Tensor,
        origin_flat: Tensor,
        origin_ts_flat: Tensor | None,
        relation: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:

        _empty1 = torch.empty((0,), dtype=torch.long, device=self._device)

        sampled, vmask, orig, sampled_ts = self._vectorized_hop2(
            pg_all,
            origin_flat,
            origin_ts_flat,
            relation,
            generator,
        )

        ridx, cidx = vmask.nonzero(as_tuple=True)
        if ridx.numel() == 0:
            return _empty1, _empty1, _empty1, _empty1, _empty1, None

        ploc = inv_p[ridx]
        ng = sampled[ridx, cidx]
        bib = bi_all[ridx]
        tag = orig[ridx, cidx].long()
        ts = sampled_ts[ridx, cidx].long() if sampled_ts is not None else None

        keys = torch.stack([ng, bib], dim=1)
        uniq, inv = torch.unique(keys, dim=0, return_inverse=True)

        return uniq[:, 0], uniq[:, 1], ploc, inv, tag, ts

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

        bi_broadcast = (
            torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1).expand(bsz, h1)
        )

        hop1: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        hop1_ts: dict[str, Tensor | None] = {}
        bi_chunks: list[Tensor] = []
        pg_chunks: list[Tensor] = []
        origin_parts: list[Tensor] = []
        ts_parts: list[Tensor] = []

        for beh in BEHAVIOR_TYPES:
            p, m, _, ts_mat = self._vectorized_hop1(user_seeds, beh, generator)
            bi_flat = bi_broadcast[m]
            pg_flat = p[m]
            hop1[beh] = (p, m, bi_flat, pg_flat)
            ts_flat = ts_mat[m] if ts_mat is not None else None
            hop1_ts[beh] = ts_flat
            if pg_flat.numel() > 0:
                bi_chunks.append(bi_flat)
                pg_chunks.append(pg_flat)
                origin_parts.append(
                    torch.full(
                        (pg_flat.size(0),),
                        BEHAVIOR_TO_ID[beh],
                        dtype=torch.int8,
                        device=device,
                    )
                )
                if ts_flat is not None:
                    ts_parts.append(ts_flat)

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
        origin_ts_flat = (
            torch.cat(ts_parts, dim=0) if ts_parts and len(ts_parts) == len(pg_chunks) else None
        )

        keys_p = torch.stack([pg_all, bi_all], dim=1)
        uniq_p, inv_p = torch.unique(keys_p, dim=0, return_inverse=True)
        prod_x = uniq_p[:, 0]
        prod_b = uniq_p[:, 1]

        beh_edges: dict[str, list[Tensor]] = {"view": [], "cart": [], "purchase": []}
        beh_edge_ts: dict[str, list[Tensor]] = {"view": [], "cart": [], "purchase": []}
        offset = 0
        for beh in BEHAVIOR_TYPES:
            _, _, bi_flat, pg_flat = hop1[beh]
            n_e = bi_flat.numel()
            if n_e == 0:
                continue
            ploc = inv_p[offset : offset + n_e]
            offset += n_e
            beh_edges[beh].append(torch.stack([bi_flat, ploc], dim=0))
            if hop1_ts[beh] is not None:
                beh_edge_ts[beh].append(hop1_ts[beh])

        def _cat_e(parts: list[Tensor]) -> Tensor:
            return torch.cat(parts, dim=1) if parts else _empty2

        def _cat_ts(parts: list[Tensor]) -> Tensor | None:
            return torch.cat(parts, dim=0) if parts else None

        e_view = _cat_e(beh_edges["view"])
        e_cart = _cat_e(beh_edges["cart"])
        e_purchase = _cat_e(beh_edges["purchase"])
        ts_view = _cat_ts(beh_edge_ts["view"])
        ts_cart = _cat_ts(beh_edge_ts["cart"])
        ts_purchase = _cat_ts(beh_edge_ts["purchase"])

        e_rev_view = e_view.flip(0) if e_view.size(1) > 0 else _empty2
        e_rev_cart = e_cart.flip(0) if e_cart.size(1) > 0 else _empty2
        e_rev_purchase = e_purchase.flip(0) if e_purchase.size(1) > 0 else _empty2

        cat_x, cat_b, pc_src, pc_dst, pc_tag, pc_ts = self._process_hop2(
            pg_all,
            bi_all,
            inv_p,
            origin_flat,
            origin_ts_flat,
            "belongs_to",
            generator,
        )

        brand_x, brand_b, pb_src, pb_dst, pb_tag, pb_ts = self._process_hop2(
            pg_all,
            bi_all,
            inv_p,
            origin_flat,
            origin_ts_flat,
            "brand",
            generator,
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
            ("user", "view", "product"): (e_view, None, ts_view),
            ("user", "cart", "product"): (e_cart, None, ts_cart),
            ("user", "purchase", "product"): (e_purchase, None, ts_purchase),
            ("product", "rev_view", "user"): (e_rev_view, None, ts_view),
            ("product", "rev_cart", "user"): (e_rev_cart, None, ts_cart),
            ("product", "rev_purchase", "user"): (e_rev_purchase, None, ts_purchase),
            ("product", "belongs_to", "category"): (_ei(pc_src, pc_dst), _attr(pc_tag), pc_ts),
            ("category", "contains", "product"): (_ei(pc_dst, pc_src), _attr(pc_tag), pc_ts),
            ("product", "producedBy", "brand"): (_ei(pb_src, pb_dst), _attr(pb_tag), pb_ts),
            ("brand", "brands", "product"): (_ei(pb_dst, pb_src), _attr(pb_tag), pb_ts),
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

        cat_x, cat_b, pc_src, pc_dst, pc_tag, pc_ts = self._process_hop2(
            product_seeds,
            prod_b,
            inv_p,
            origin_flat,
            None,
            "belongs_to",
            generator,
        )
        brand_x, brand_b, pb_src, pb_dst, pb_tag, pb_ts = self._process_hop2(
            product_seeds,
            prod_b,
            inv_p,
            origin_flat,
            None,
            "brand",
            generator,
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
            ("product", "belongs_to", "category"): (_ei(pc_src, pc_dst), _attr(pc_tag), pc_ts),
            ("category", "contains", "product"): (_ei(pc_dst, pc_src), _attr(pc_tag), pc_ts),
            ("product", "producedBy", "brand"): (_ei(pb_src, pb_dst), _attr(pb_tag), pb_ts),
            ("brand", "brands", "product"): (_ei(pb_dst, pb_src), _attr(pb_tag), pb_ts),
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
    edge_data: dict[
        tuple[str, str, str],
        tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor | None],
    ],
) -> HeteroData:
    data = HeteroData()
    _e2 = torch.empty((2, 0), dtype=torch.long, device=device)

    for ntype, (x, batch) in node_data.items():
        data[ntype].x = x.to(device=device, dtype=torch.long)
        data[ntype].batch = batch.to(device=device, dtype=torch.long)
        data[ntype].num_nodes = x.size(0)

    for rel, payload in edge_data.items():
        if len(payload) == 2:
            ei, attr = payload
            edge_ts = None
        else:
            ei, attr, edge_ts = payload
        if ei is None or ei.numel() == 0:
            data[rel].edge_index = _e2.clone()
        else:
            # .contiguous() is mandatory: tensors produced by .flip(0) share
            # storage with the original but have a negative stride. PyG's
            # scatter_add_ kernel requires C-contiguous layout; without this
            # call PyTorch makes an implicit copy mid-forward, causing a VRAM
            # spike proportional to edge count on every training step.
            data[rel].edge_index = ei.to(device=device, dtype=torch.long).contiguous()

        if attr is not None and attr.numel() > 0:
            data[rel].edge_attr = attr.to(device=device).contiguous()
        if edge_ts is not None and edge_ts.numel() > 0:
            data[rel].edge_ts = edge_ts.to(device=device, dtype=torch.long).contiguous()

    return data


def collate_hetero_subgraphs(batch: list[HeteroData]) -> Batch:
    return Batch.from_data_list(batch)



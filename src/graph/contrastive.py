import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import (
    BEHAVIOR_TYPES,
    EMBED_DIM,
    SVD_RANK,
    GNNOutput,
    SVDFactors,
)

class ContrastiveLearning(nn.Module):
    """
    InfoNCE Contrastive Learning (P3) cho Heterogeneous Graph RecSys.

    Hai view được đối chiếu:
      - GNN view : per_behavior_emb từ GNNOutput
      - SVD view : g_user = US_k @ (VS_k^T @ E_item)  — O(q*d), không cần ma trận I×J

    Loss:
        L_cl = mean over behaviors [
            InfoNCE(gnn_user_k, svd_user_k) +
            InfoNCE(gnn_item_k, svd_item_k)
        ]

    Args:
        tau      : temperature (default = 0.2)
        proj_dim : chiều projection head (None = không dùng)
        behaviors: list behavior cần tính CL (default = tất cả)
    """
    def __init__(
        self, 
        tau: float = 0.2,
        proj_dim: int | None = None,
        behaviors: list[str] | None = None,
    ):
        super().__init__()
        self.tau = tau
        self.proj_dim = proj_dim
        self.behaviors = behaviors if behaviors is not None else BEHAVIOR_TYPES
        
        if proj_dim is not None:
            self.projectors: nn.Module = nn.Sequential(
                nn.Linear(EMBED_DIM, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            ) 
        else:
            self.projectors = nn.Identity()

    def info_nce(
        self,
        z_anchor: torch.Tensor,  # [N, d] -> gnn_user_k or gnn_item_k
        z_positive: torch.Tensor, # [N, d] -> svd_user_k or svd_item_k
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE với in-batch negatives.

        L = -1/N Σ_i  log( exp(sim(i,i)/τ) / Σ_j exp(sim(i,j)/τ) )

        Với mỗi anchor i:
          - positive  = z_pos[i]
          - negatives = z_pos[j≠i]  (in-batch)
        """
        z_anchor = F.normalize(self.projectors(z_anchor), dim=-1)   # [N, d]
        z_positive = F.normalize(self.projectors(z_positive), dim=-1) # [N, d]
        
        # Tính cosine similarity: sim(i,j) = z_anchor[i] · z_positive[j]
        sim_matrix = torch.matmul(z_anchor, z_positive.T) / self.tau  # [N, N]
        
        # Diagonal = positive pair → label = index
        N = z_anchor.size(0)
        labels = torch.arange(N, device=z_anchor.device)

        # Symmetric: cross_entropy cả 2 chiều
        loss = (
            F.cross_entropy(sim_matrix,   labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2.0

        return loss
    
    # SVD user embedding: g_user = US_k @ (VS_k^T @ E_item) - > O(q*d) thay vì O(I*J)
    def _svd_user_view(
        self,
        svd_factors: SVDFactors,
        item_emb_all: torch.Tensor,  # [N_i, d]
        behavior: str,
    ) -> torch.Tensor:
        """
        g_user = US_k @ (VS_k[:B]^T @ E_item)
        Returns: (N_users, d)
        """ 
        US = svd_factors.US[behavior]  # [N_u, k]
        VS = svd_factors.VS[behavior]  # [N_i, k]
        
        vs_sub = VS[:item_emb_all.size(0)]  # [N_i, k]    
        context = vs_sub.T @ item_emb_all  # [k, d]
        return US @ context  # [N_u, d]
    
    def _svd_item_view(
        self,
        svd_factors: SVDFactors,
        user_emb: torch.Tensor,  # [N_u, d]
        behavior: str,
    ) -> torch.Tensor:
        """
        g_item = VS_k @ (US_k[:B]^T @ E_user)
        Returns: (N_items, d)
        """ 
        US = svd_factors.US[behavior]  # [N_u, k]
        VS = svd_factors.VS[behavior]  # [N_i, k]

        us_sub = US[:user_emb.size(0)]  # [N_u, k]
        context = us_sub.T @ user_emb    # [k, d]
        return VS @ context  # [N_i, d] 
    
    # forward() sẽ trả về dict {behavior: loss_cl} để trainer tổng hợp vào log
    def forward(
        self, 
        gnn_output: GNNOutput,
        svd_factors: SVDFactors,
    ) -> torch.Tensor:
        """
        Args:
            gnn_output : GNNOutput từ GNN encoder
            svd        : SVDFactors từ preprocessing
        Returns:
            cl_loss : scalar
        """
        device = gnn_output.final_user_emb.device
        cl_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for beh in self.behaviors:
            if beh not in gnn_output.per_behavior_emb:
                continue

            gnn_user = gnn_output.per_behavior_emb[beh]["user"]     # (B, d)
            gnn_item = gnn_output.per_behavior_emb[beh]["product"]  # (B, d)

            # SVD view, cắt đúng batch size
            svd_user = self._svd_user_view(svd_factors, gnn_item, beh)[:gnn_user.size(0)]
            svd_item = self._svd_item_view(svd_factors, gnn_user, beh)[:gnn_item.size(0)]

            # info_nce applies self.projectors internally — pass raw embeddings
            cl_loss = cl_loss + self.info_nce(gnn_user, svd_user)
            cl_loss = cl_loss + self.info_nce(gnn_item, svd_item)
            count += 2

        if count > 0:
            cl_loss = cl_loss / count

        return cl_loss


class SvdGnnContrastive(nn.Module):
    """Per-batch InfoNCE between GNN user embeddings and SVD-projected user
    embeddings. Per behavior, per user. Global supervision beyond L-hop.

    Math (per behavior k):
        svd_user[i] = US_k[user_global[i]] @ (VS_k[item_global].T @ item_emb)
    O(q*d) intermediate; never materialises I*J.
    """
    def __init__(
        self,
        svd_factors: SVDFactors,
        behaviors: list[str] | None = None,
        tau: float = 0.2,
    ):
        super().__init__()
        from src.core.contracts import BEHAVIOR_TYPES
        self.behaviors = list(behaviors) if behaviors else list(BEHAVIOR_TYPES)
        self.tau = tau
        for b in self.behaviors:
            if b not in svd_factors.US or b not in svd_factors.VS:
                raise KeyError(f"SVD factors missing behaviour {b!r}")
            self.register_buffer(f"_us_{b}", svd_factors.US[b], persistent=False)
            self.register_buffer(f"_vs_{b}", svd_factors.VS[b], persistent=False)

    def _us(self, beh: str) -> torch.Tensor:
        return getattr(self, f"_us_{beh}")

    def _vs(self, beh: str) -> torch.Tensor:
        return getattr(self, f"_vs_{beh}")

    @staticmethod
    def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        sim = (z_a @ z_b.T) / tau
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

    def forward(
        self,
        beh_embs: dict[str, torch.Tensor],
        user_global: torch.Tensor,
        item_emb: torch.Tensor,
        item_global: torch.Tensor,
    ) -> torch.Tensor:
        device = item_emb.device
        if item_emb.size(0) == 0 or user_global.numel() < 2:
            return torch.zeros((), device=device)

        loss = torch.zeros((), device=device)
        n_terms = 0
        i_g = item_global.long()
        u_g = user_global.long()

        for b in self.behaviors:
            if b not in beh_embs:
                continue
            gnn_user = beh_embs[b]
            B = gnn_user.size(0)
            if B < 2:
                continue
            US_b = self._us(b)
            VS_b = self._vs(b)

            us_rows = US_b[u_g]
            vs_rows = VS_b[i_g]
            context = vs_rows.T @ item_emb
            svd_user = us_rows @ context

            loss = loss + self._info_nce(gnn_user, svd_user, self.tau)
            n_terms += 1

        return loss / max(n_terms, 1)


if __name__ == "__main__":
    import math

    torch.manual_seed(42)

    print("=" * 55)
    print("  ContrastiveLearning — self-test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, d, q, N_u, N_i = 64, EMBED_DIM, SVD_RANK, 200, 500

    gnn_output = GNNOutput(
        per_behavior_emb={
            beh: {
                "user":    torch.randn(B, d, device=device),
                "product": torch.randn(B, d, device=device),
            }
            for beh in BEHAVIOR_TYPES
        },
        final_user_emb=torch.randn(B, d, device=device),
        final_item_emb=torch.randn(B, d, device=device),
    )
    svd = SVDFactors(
        US={beh: torch.randn(N_u, q, device=device) for beh in BEHAVIOR_TYPES},
        VS={beh: torch.randn(N_i, q, device=device) for beh in BEHAVIOR_TYPES},
    )

    # Test 1: không projection
    cl = ContrastiveLearning(tau=0.2).to(device)
    loss = cl(gnn_output, svd)
    assert loss.dim() == 0 and not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() >= 0
    print(f" [PASS] No projection  — loss = {loss.item():.4f}")

    # Test 2: có projection head
    cl_proj = ContrastiveLearning(tau=0.2, proj_dim=64).to(device)
    loss_proj = cl_proj(gnn_output, svd)
    assert not torch.isnan(loss_proj)
    print(f" [PASS] With projection — loss = {loss_proj.item():.4f}")

    # Test 3: tau nhỏ → loss cao hơn
    loss_low  = ContrastiveLearning(tau=0.05).to(device)(gnn_output, svd)
    loss_high = ContrastiveLearning(tau=1.0 ).to(device)(gnn_output, svd)
    assert loss_low.item() >= loss_high.item() - 1e-3
    print(f" [PASS] Temperature    — τ=0.05: {loss_low.item():.4f} >= τ=1.0: {loss_high.item():.4f}")

    # Test 4: gradient flows
    gnn_grad = GNNOutput(
        per_behavior_emb={
            beh: {
                "user":    torch.randn(B, d, device=device, requires_grad=True),
                "product": torch.randn(B, d, device=device, requires_grad=True),
            }
            for beh in BEHAVIOR_TYPES
        },
        final_user_emb=torch.randn(B, d, device=device),
        final_item_emb=torch.randn(B, d, device=device),
    )
    cl(gnn_grad, svd).backward()
    for beh in BEHAVIOR_TYPES:
        for nt in ["user", "product"]:
            assert gnn_grad.per_behavior_emb[beh][nt].grad is not None
    print(f" [PASS] Gradient flows qua all per_behavior_emb")

    # Test 5: random alignment baseline
    print(f" [INFO] Random baseline ≈ ln({B}) = {math.log(B):.4f}, got {loss.item():.4f}")

    print("\n" + "=" * 55)
    print("ALL PASSED — ContrastiveLearning (InfoNCE, τ=0.2)")
    print("=" * 55)
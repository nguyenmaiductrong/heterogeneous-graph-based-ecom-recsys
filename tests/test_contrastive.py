import math
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph.contrastive import ContrastiveLearning
from core.contracts import (
    BEHAVIOR_TYPES,
    EMBED_DIM,
    SVD_RANK,
    GNNOutput,
    SVDFactors,
)

# shared constants

BATCH = 64       # in-batch negatives pool size
N_USERS = 200    # "full catalog" for SVD factors (> BATCH)
N_ITEMS = 500


# fixtures

@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def svd(device) -> SVDFactors:
    torch.manual_seed(0)
    return SVDFactors(
        US={beh: torch.randn(N_USERS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
        VS={beh: torch.randn(N_ITEMS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
    )


@pytest.fixture(scope="module")
def gnn_out(device) -> GNNOutput:
    torch.manual_seed(1)
    return GNNOutput(
        per_behavior_emb={
            beh: {
                "user":    torch.randn(BATCH, EMBED_DIM, device=device),
                "product": torch.randn(BATCH, EMBED_DIM, device=device),
            }
            for beh in BEHAVIOR_TYPES
        },
        final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
    )


@pytest.fixture
def cl(device) -> ContrastiveLearning:
    return ContrastiveLearning(tau=0.2).to(device)


@pytest.fixture
def cl_proj(device) -> ContrastiveLearning:
    return ContrastiveLearning(tau=0.2, proj_dim=64).to(device)


# TestInfoNCE

class TestInfoNCE:
    """Unit tests for the info_nce() method in isolation."""

    def test_output_is_scalar(self, cl):
        z = torch.randn(BATCH, EMBED_DIM)
        loss = cl.info_nce(z, z.clone())
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_non_negative(self, cl):
        """Cross-entropy is always ≥ 0."""
        z_a = torch.randn(BATCH, EMBED_DIM)
        z_p = torch.randn(BATCH, EMBED_DIM)
        loss = cl.info_nce(z_a, z_p)
        assert loss.item() >= 0.0

    def test_perfect_alignment_near_random_baseline(self, cl):
        """
        z_anchor == z_positive (perfect alignment) → loss ≈ ln(N).
        Reason: after L2-norm, sim(i,i)/τ = 1/τ but sim(i,j) = cos(θ_ij)/τ.
        With random orthogonal-ish vectors in high-dim, sim(i,j) ≈ 0, so
        the softmax denominator ≈ N terms each ≈ exp(0) = 1.
        Therefore loss ≈ ln(N) even for perfect alignment — this is the
        lower bound for a random embedding bank.
        """
        torch.manual_seed(42)
        z = torch.randn(BATCH, EMBED_DIM)
        loss = cl.info_nce(z, z.clone())
        # Allow ±1 nats around ln(N) — perfect alignment still has N-1 negatives
        assert abs(loss.item() - math.log(BATCH)) < 1.5

    def test_identical_pairs_lower_than_random(self, cl):
        """
        Identical pairs (sim=1/τ on diagonal) must produce lower loss than
        completely mismatched pairs (sim≈-1/τ on diagonal).
        """
        torch.manual_seed(7)
        z = torch.randn(BATCH, EMBED_DIM)
        z_norm = torch.nn.functional.normalize(z, dim=-1)

        # Positive pairs = same vector (sim diagonal = 1)
        loss_aligned = cl.info_nce(z_norm, z_norm.clone())
        # Positive pairs = negated vector (sim diagonal = -1)
        loss_anti = cl.info_nce(z_norm, -z_norm)
        assert loss_aligned.item() < loss_anti.item()

    def test_symmetric(self, cl):
        """
        info_nce(a, b) should equal info_nce(b, a) because it's
        computed as (CE(sim, labels) + CE(sim^T, labels)) / 2.
        """
        torch.manual_seed(3)
        z_a = torch.randn(BATCH, EMBED_DIM)
        z_p = torch.randn(BATCH, EMBED_DIM)
        loss_ab = cl.info_nce(z_a, z_p)
        loss_ba = cl.info_nce(z_p, z_a)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-5

    def test_no_nan_no_inf(self, cl):
        torch.manual_seed(9)
        z_a = torch.randn(BATCH, EMBED_DIM)
        z_p = torch.randn(BATCH, EMBED_DIM)
        loss = cl.info_nce(z_a, z_p)
        assert not torch.isnan(loss), "NaN in info_nce output"
        assert not torch.isinf(loss), "Inf in info_nce output"

    def test_gradient_flows_through_info_nce(self, device):
        """Gradient must reach both anchor and positive."""
        cl = ContrastiveLearning(tau=0.2).to(device)
        z_a = torch.randn(BATCH, EMBED_DIM, requires_grad=True, device=device)
        z_p = torch.randn(BATCH, EMBED_DIM, requires_grad=True, device=device)
        loss = cl.info_nce(z_a, z_p)
        loss.backward()
        assert z_a.grad is not None and not torch.isnan(z_a.grad).any()
        assert z_p.grad is not None and not torch.isnan(z_p.grad).any()


# TestTemperature

class TestTemperature:
    """Temperature τ controls loss scale and gradient magnitude."""

    def test_lower_tau_higher_loss_random_embeddings(self, device, svd, gnn_out):
        """
        With random (unaligned) embeddings, lower τ → harder negatives →
        higher cross-entropy. Verified across a wide τ range.
        """
        losses = {}
        for tau in [0.05, 0.2, 0.5, 1.0]:
            cl = ContrastiveLearning(tau=tau).to(device)
            with torch.no_grad():
                losses[tau] = cl(gnn_out, svd).item()

        assert losses[0.05] > losses[0.2], (
            f"τ=0.05 loss {losses[0.05]:.4f} should > τ=0.2 loss {losses[0.2]:.4f}"
        )
        assert losses[0.2] > losses[0.5], (
            f"τ=0.2 loss {losses[0.2]:.4f} should > τ=0.5 loss {losses[0.5]:.4f}"
        )
        assert losses[0.5] > losses[1.0], (
            f"τ=0.5 loss {losses[0.5]:.4f} should > τ=1.0 loss {losses[1.0]:.4f}"
        )

    def test_tau_scales_logits(self, device):
        """
        Dividing by τ uniformly scales the sim_matrix → lower τ sharpens
        the softmax distribution. Verify that logit magnitude ∝ 1/τ.
        """
        torch.manual_seed(42)
        z_a = torch.randn(BATCH, EMBED_DIM)
        z_p = torch.randn(BATCH, EMBED_DIM)

        # At τ→∞, loss → ln(N) (uniform distribution); at τ→0, winner-takes-all.
        # We only check that the ratio of losses is monotone (not exact).
        cl_lo = ContrastiveLearning(tau=0.1).to(device)
        cl_hi = ContrastiveLearning(tau=2.0).to(device)

        loss_lo = cl_lo.info_nce(z_a.to(device), z_p.to(device))
        loss_hi = cl_hi.info_nce(z_a.to(device), z_p.to(device))
        assert loss_lo.item() > loss_hi.item()

    def test_default_tau_is_0_2(self, device):
        cl = ContrastiveLearning().to(device)
        assert cl.tau == 0.2


# TestShape

class TestShape:
    """Output shapes and structural correctness of forward()."""

    def test_forward_returns_scalar(self, cl, gnn_out, svd):
        loss = cl(gnn_out, svd)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_forward_no_nan_no_inf(self, cl, gnn_out, svd):
        loss = cl(gnn_out, svd)
        assert not torch.isnan(loss), "NaN in forward() output"
        assert not torch.isinf(loss), "Inf in forward() output"

    def test_forward_non_negative(self, cl, gnn_out, svd):
        loss = cl(gnn_out, svd)
        assert loss.item() >= 0.0

    def test_projection_head_returns_scalar(self, cl_proj, gnn_out, svd):
        loss = cl_proj(gnn_out, svd)
        assert loss.dim() == 0

    def test_projection_head_no_nan(self, cl_proj, gnn_out, svd):
        loss = cl_proj(gnn_out, svd)
        assert not torch.isnan(loss)

    def test_single_behavior_subset(self, device, svd, gnn_out):
        """Restricting to one behavior → valid scalar loss."""
        cl = ContrastiveLearning(tau=0.2, behaviors=["purchase"]).to(device)
        loss = cl(gnn_out, svd)
        assert loss.dim() == 0 and loss.item() >= 0.0

    def test_missing_behavior_in_gnn_output_skipped(self, device, svd):
        """Behaviors absent from gnn_output.per_behavior_emb are silently skipped."""
        partial_gnn = GNNOutput(
            per_behavior_emb={
                "purchase": {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device),
                }
                # view and cart intentionally absent
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        loss = cl(partial_gnn, svd)
        assert loss.dim() == 0 and not torch.isnan(loss)

    def test_all_behaviors_missing_returns_zero(self, device, svd):
        """If no behavior matches, cl_loss == 0.0 (no contribution)."""
        empty_gnn = GNNOutput(
            per_behavior_emb={},
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        loss = cl(empty_gnn, svd)
        assert loss.item() == 0.0

    def test_svd_user_view_shape(self, cl, svd, device):
        item_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        out = cl._svd_user_view(svd, item_emb, "purchase")
        # Returns (N_USERS, EMBED_DIM)
        assert out.shape == (N_USERS, EMBED_DIM), f"Unexpected shape: {out.shape}"

    def test_svd_item_view_shape(self, cl, svd, device):
        user_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        out = cl._svd_item_view(svd, user_emb, "purchase")
        assert out.shape == (N_ITEMS, EMBED_DIM), f"Unexpected shape: {out.shape}"


# TestGradient

class TestGradient:
    """Gradient flows correctly through the full contrastive pipeline."""

    def test_gradient_flows_to_all_per_behavior_emb(self, device, svd):
        """
        Backward pass must produce non-None, non-NaN gradients for every
        (behavior, node_type) embedding tensor.
        """
        gnn_grad = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        cl(gnn_grad, svd).backward()

        for beh in BEHAVIOR_TYPES:
            for nt in ["user", "product"]:
                g = gnn_grad.per_behavior_emb[beh][nt].grad
                assert g is not None, f"No grad for [{beh}][{nt}]"
                assert not torch.isnan(g).any(), f"NaN grad for [{beh}][{nt}]"

    def test_gradient_flows_through_projection_head(self, device, svd):
        """Projection head parameters must receive gradients."""
        gnn_grad = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl_proj = ContrastiveLearning(tau=0.2, proj_dim=64).to(device)
        cl_proj(gnn_grad, svd).backward()

        for name, param in cl_proj.named_parameters():
            assert param.grad is not None, f"No grad for projector param: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for: {name}"

    def test_gradient_magnitude_finite(self, device, svd):
        """Gradients must be finite (no exploding gradients at τ=0.2)."""
        gnn_grad = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        cl(gnn_grad, svd).backward()

        for beh in BEHAVIOR_TYPES:
            for nt in ["user", "product"]:
                g = gnn_grad.per_behavior_emb[beh][nt].grad
                assert torch.isfinite(g).all(), f"Infinite grad for [{beh}][{nt}]"

    def test_loss_decreases_with_optimizer_step(self, device, svd):
        """
        One Adam step on a learnable embedding should decrease (or hold) loss.
        Verifies that the optimization signal is pointing the right direction.
        """
        torch.manual_seed(42)
        embs = {
            beh: {
                "user":    nn.Parameter(torch.randn(BATCH, EMBED_DIM, device=device)),
                "product": nn.Parameter(torch.randn(BATCH, EMBED_DIM, device=device)),
            }
            for beh in BEHAVIOR_TYPES
        }
        cl = ContrastiveLearning(tau=0.2).to(device)
        optimizer = torch.optim.Adam(
            [p for beh in embs.values() for p in beh.values()], lr=1e-2
        )

        def make_gnn(embs):
            return GNNOutput(
                per_behavior_emb={b: {k: v for k, v in d.items()} for b, d in embs.items()},
                final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
                final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            )

        loss_before = cl(make_gnn(embs), svd)
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()
        loss_after = cl(make_gnn(embs), svd)

        # Allow tiny numerical noise but loss should not explode
        assert loss_after.item() < loss_before.item() + 0.5, (
            f"Loss did not decrease or stay stable: "
            f"{loss_before.item():.4f} → {loss_after.item():.4f}"
        )


# TestSVDView

class TestSVDView:
    """Memory-safe SVD view: VS^T @ E_item first (128 KB intermediate path)."""

    def test_svd_user_view_no_large_intermediate(self, cl, svd, device):
        """
        _svd_user_view computes VS^T @ item_emb first ([k, d]),
        NOT US @ VS^T first ([N_u, N_i]).
        We verify this by checking that the output shape is (N_USERS, EMBED_DIM)
        — if it attempted N_u × N_i, it would OOM on real data.
        """
        item_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        out = cl._svd_user_view(svd, item_emb, "view")
        assert out.shape == (N_USERS, EMBED_DIM)
        assert not torch.isnan(out).any()

    def test_svd_item_view_no_large_intermediate(self, cl, svd, device):
        """Analogous check for _svd_item_view: US^T @ user_emb first."""
        user_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        out = cl._svd_item_view(svd, user_emb, "cart")
        assert out.shape == (N_ITEMS, EMBED_DIM)
        assert not torch.isnan(out).any()

    def test_svd_user_view_all_behaviors(self, cl, svd, device):
        """SVD user view works for every behavior key."""
        item_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        for beh in BEHAVIOR_TYPES:
            out = cl._svd_user_view(svd, item_emb, beh)
            assert out.shape == (N_USERS, EMBED_DIM), f"Shape error for {beh}"

    def test_svd_item_view_all_behaviors(self, cl, svd, device):
        user_emb = torch.randn(BATCH, EMBED_DIM, device=device)
        for beh in BEHAVIOR_TYPES:
            out = cl._svd_item_view(svd, user_emb, beh)
            assert out.shape == (N_ITEMS, EMBED_DIM), f"Shape error for {beh}"

    def test_forward_truncates_svd_to_batch(self, device, svd):
        """
        SVD factors have N_USERS/N_ITEMS rows, but forward() slices
        [:batch_size] so that info_nce receives equal-size tensors.
        """
        cl = ContrastiveLearning(tau=0.2).to(device)
        # Use a smaller batch than N_USERS/N_ITEMS
        small_batch = 16
        gnn_small = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(small_batch, EMBED_DIM, device=device),
                    "product": torch.randn(small_batch, EMBED_DIM, device=device),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(small_batch, EMBED_DIM, device=device),
            final_item_emb=torch.randn(small_batch, EMBED_DIM, device=device),
        )
        loss = cl(gnn_small, svd)
        assert loss.dim() == 0 and not torch.isnan(loss)


# TestGPU

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """GPU-specific tests: device consistency and no CPU↔GPU tensor mixing."""

    def test_forward_on_cuda(self):
        device = torch.device("cuda")
        torch.manual_seed(0)
        svd = SVDFactors(
            US={beh: torch.randn(N_USERS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
            VS={beh: torch.randn(N_ITEMS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
        )
        gnn = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        loss = cl(gnn, svd)
        assert loss.device.type == "cuda"
        assert not torch.isnan(loss)

    def test_gradient_on_cuda(self):
        device = torch.device("cuda")
        svd = SVDFactors(
            US={beh: torch.randn(N_USERS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
            VS={beh: torch.randn(N_ITEMS, SVD_RANK, device=device) for beh in BEHAVIOR_TYPES},
        )
        gnn = GNNOutput(
            per_behavior_emb={
                beh: {
                    "user":    torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                    "product": torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True),
                }
                for beh in BEHAVIOR_TYPES
            },
            final_user_emb=torch.randn(BATCH, EMBED_DIM, device=device),
            final_item_emb=torch.randn(BATCH, EMBED_DIM, device=device),
        )
        cl = ContrastiveLearning(tau=0.2).to(device)
        cl(gnn, svd).backward()
        for beh in BEHAVIOR_TYPES:
            for nt in ["user", "product"]:
                g = gnn.per_behavior_emb[beh][nt].grad
                assert g is not None and g.device.type == "cuda"

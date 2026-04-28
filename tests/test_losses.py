import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.losses import (
    PopularityBiasedNegativeSampler,
    bpr_loss,
    MultiTaskBPRLoss,
    BPRTrainingStep,
)


REES46_BEHAVIOR_COUNTS = {
    "view": 185_679_073,
    "cart": 14_192_808,
    "purchase": 4_455_936,
}

NUM_ITEMS_SMALL = 100_000  # Cho unit test, không cần full scale


@pytest.fixture
def sampler():
    """Tạo sampler với power-law distribution giả lập e-commerce."""
    counts = torch.zeros(NUM_ITEMS_SMALL)
    counts[:100] = 10_000       # head items  (top 0.1%)
    counts[100:1_000] = 1_000   # mid-head    (0.1% - 1%)
    counts[1_000:10_000] = 100  # mid-tail    (1% - 10%)
    counts[10_000:] = 1         # tail items  (90%)
    return PopularityBiasedNegativeSampler(
        item_counts={"global": counts},
        num_items=NUM_ITEMS_SMALL,
        alpha=0.75,
    )


@pytest.fixture
def multi_task_loss():
    """MultiTaskBPRLoss với inverse_freq weighting."""
    return MultiTaskBPRLoss(
        behavior_counts=REES46_BEHAVIOR_COUNTS,
    )


@pytest.fixture
def fake_behavior_losses():
    """Giả lập per-behavior BPR losses."""
    return {
        "view": torch.tensor(0.5),
        "cart": torch.tensor(0.6),
        "purchase": torch.tensor(0.7),
    }


class TestBPRLoss:
    """Verify bpr_loss: correctness, numerical stability, gradient flow."""

    def test_perfect_ranking_loss_near_zero(self):
        """Khi pos >> neg, model rank đúng hoàn hảo → loss ≈ 0."""
        pos = torch.tensor([10.0, 10.0])
        neg = torch.tensor([-10.0, -10.0])
        loss = bpr_loss(pos, neg)
        assert loss.item() < 1e-4

    def test_bad_ranking_loss_large(self):
        """Khi pos << neg, model rank sai hoàn toàn → loss lớn."""
        pos = torch.tensor([-10.0, -10.0])
        neg = torch.tensor([10.0, 10.0])
        loss = bpr_loss(pos, neg)
        assert loss.item() > 10.0

    def test_equal_scores_loss_is_ln2(self):
        """
        Khi pos == neg → σ(0) = 0.5 → -ln(0.5) = ln(2) ≈ 0.6931.
        Đây là "random ranking" loss — baseline tham chiếu.
        """
        pos = torch.tensor([1.0, 1.0])
        neg = torch.tensor([1.0, 1.0])
        loss = bpr_loss(pos, neg)
        assert abs(loss.item() - np.log(2)) < 0.01

    def test_multi_negative_shape(self):
        """Support K negatives per positive: pos=[B], neg=[B, K] → scalar."""
        pos = torch.randn(32)
        neg = torch.randn(32, 5)
        loss = bpr_loss(pos, neg)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_gradient_flows_no_nan(self):
        """Gradient phải chảy qua cả pos và neg, không NaN."""
        pos = torch.randn(16, requires_grad=True)
        neg = torch.randn(16, requires_grad=True)
        loss = bpr_loss(pos, neg)
        loss.backward()

        assert pos.grad is not None, "Không có gradient cho pos_scores"
        assert neg.grad is not None, "Không có gradient cho neg_scores"
        assert not torch.isnan(pos.grad).any(), "NaN trong gradient pos"
        assert not torch.isnan(neg.grad).any(), "NaN trong gradient neg"

    def test_float16_numerical_stability(self):
        """
        CRITICAL: Kiểm tra logsigmoid không tạo NaN/Inf trong float16.
        Giá trị ±50 trong float16 là cực trị — sigmoid(100) hay sigmoid(-100)
        sẽ overflow/underflow nếu dùng -log(sigmoid(x)) thay vì logsigmoid(x).
        """
        pos = torch.tensor([50.0, -50.0], dtype=torch.float16)
        neg = torch.tensor([-50.0, 50.0], dtype=torch.float16)
        loss = bpr_loss(pos, neg)
        assert not torch.isnan(loss), "NaN ở float16 — logsigmoid có vấn đề"
        assert not torch.isinf(loss), "Inf ở float16 — logsigmoid có vấn đề"

    def test_output_is_scalar(self):
        """Loss luôn trả về scalar bất kể input shape."""
        for shape in [(8,), (8, 1), (8, 3)]:
            pos = torch.randn(8)
            neg = torch.randn(*shape)
            loss = bpr_loss(pos, neg)
            assert loss.dim() == 0


class TestNegativeSampler:
    """Verify sampler: output shape, valid range, popularity bias."""

    def test_output_shape(self, sampler):
        negs = sampler.sample(batch_size=1024, num_neg=4)
        assert negs.shape == (1024, 4)

    def test_single_negative(self, sampler):
        negs = sampler.sample(batch_size=512, num_neg=1)
        assert negs.shape == (512, 1)

    def test_indices_in_valid_range(self, sampler):
        negs = sampler.sample(batch_size=2048, num_neg=4)
        assert negs.min() >= 0
        assert negs.max() < NUM_ITEMS_SMALL

    def test_popularity_bias_head_oversampled(self, sampler):
        """
        Head items (top 0.1% catalog) phải được sample NHIỀU HƠN 0.1%.
        Nếu uniform, head fraction ≈ 0.1%. Với popularity bias, phải >> 0.1%.
        """
        large_sample = sampler.sample(batch_size=100_000, num_neg=1).flatten()
        head_fraction = (large_sample < 100).float().mean().item()
        # Head = 100 items / 100K items = 0.1% of catalog
        # Với alpha=0.75, head fraction phải > 1% (10x so với uniform)
        assert head_fraction > 0.01, (
            f"Head items chỉ chiếm {head_fraction:.4%} samples — "
            f"popularity bias không hoạt động"
        )

    def test_tail_items_still_sampled(self, sampler):
        """
        Tail items phải vẫn xuất hiện (nhờ +1 smoothing).
        Nếu không có smoothing, tail items có thể có prob = 0.
        """
        large_sample = sampler.sample(batch_size=200_000, num_neg=1).flatten()
        tail_fraction = (large_sample >= 10_000).float().mean().item()
        assert tail_fraction > 0.0, "Tail items không bao giờ được sample — thiếu smoothing?"

    def test_fallback_to_global(self, sampler):
        """Khi behavior không tồn tại, fallback sang 'global'."""
        negs = sampler.sample(batch_size=64, num_neg=1, behavior="nonexistent")
        assert negs.shape == (64, 1)  # Không crash, dùng global distribution


class TestMultiTaskBPRLoss:
    """Verify multi-task loss: weighting, float32 enforcement, edge cases."""

    def test_output_not_nan(self, multi_task_loss, fake_behavior_losses):
        total, _ = multi_task_loss(fake_behavior_losses)
        assert not torch.isnan(total)

    def test_output_is_scalar(self, multi_task_loss, fake_behavior_losses):
        total, _ = multi_task_loss(fake_behavior_losses)
        assert total.dim() == 0

    def test_purchase_weight_highest(self, multi_task_loss, fake_behavior_losses):
        """
        Với inverse_freq weighting trên REES46 (view:cart:purchase = 42:3:1),
        purchase CÓ ÍT DATA NHẤT → weight PHẢI CAO NHẤT.
        """
        _, log = multi_task_loss(fake_behavior_losses)
        assert log["weight/purchase"] > log["weight/cart"] > log["weight/view"], (
            f"Weight order sai: view={log['weight/view']:.4f}, "
            f"cart={log['weight/cart']:.4f}, purchase={log['weight/purchase']:.4f}"
        )

    def test_partial_behaviors_no_crash(self, multi_task_loss):
        """
        Không phải mọi user đều có cả 3 behaviors.
        Loss phải hoạt động khi chỉ có 1 hoặc 2 behaviors trong batch.
        """
        partial = {"view": torch.tensor(0.5)}
        total, log = multi_task_loss(partial)
        assert not torch.isnan(total)
        assert "loss/view" in log
        assert "loss/cart" not in log       # Không có cart → không xuất hiện
        assert "loss/purchase" not in log

    def test_float16_input_forced_to_float32_output(self, multi_task_loss):
        """
        CRITICAL AMP TEST: Dù input losses ở float16 (từ autocast context),
        output PHẢI là float32 (nhờ @autocast(enabled=False)).
        Nếu output ở float16: GradScaler nhân 65536 × loss → overflow.
        """
        fp16_losses = {
            "view": torch.tensor(0.5, dtype=torch.float16),
            "cart": torch.tensor(0.6, dtype=torch.float16),
            "purchase": torch.tensor(0.7, dtype=torch.float16),
        }
        total, _ = multi_task_loss(fp16_losses)
        assert total.dtype == torch.float32, (
            f"Output dtype = {total.dtype}, PHẢI là float32 để AMP không overflow"
        )

    def test_l2_regularization_applied(self, multi_task_loss, fake_behavior_losses):
        """L2 term phải xuất hiện trong log khi model_params được truyền."""
        fake_l2 = torch.tensor(100.0)
        total, log = multi_task_loss(fake_behavior_losses, model_params=fake_l2)
        assert "loss/l2" in log
        assert log["loss/l2"] > 0

    def test_l2_increases_total_loss(self, multi_task_loss, fake_behavior_losses):
        """Total loss VỚI L2 phải > total loss KHÔNG L2."""
        total_no_l2, _ = multi_task_loss(fake_behavior_losses, model_params=None)
        total_with_l2, _ = multi_task_loss(
            fake_behavior_losses, model_params=torch.tensor(1000.0)
        )
        assert total_with_l2.item() > total_no_l2.item()

    def test_weights_sum_to_num_behaviors(self, multi_task_loss):
        """
        Inverse_freq weights được normalize sao cho sum = num_behaviors.
        Điều này giữ gradient scale tổng thể tương đương single-task.
        """
        weights = multi_task_loss.task_weights
        assert abs(weights.sum().item() - 3.0) < 0.01, (
            f"Sum of weights = {weights.sum().item()}, expected 3.0"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
class TestAMPFullPipeline:
    """End-to-end test: model → loss → backward → update, no NaN/Inf."""

    NUM_USERS = 10_000
    NUM_ITEMS = 50_000
    DIM = 64

    class DummyGNNModel(nn.Module):
        """Mô phỏng GNN encoder: chỉ có embedding tables."""
        def __init__(self, num_users, num_items, dim):
            super().__init__()
            self.user_emb = nn.Embedding(num_users, dim)
            self.item_emb = nn.Embedding(num_items, dim)
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.item_emb.weight)

        def embedding_l2_norm(self):
            return (
                self.user_emb.weight.pow(2).sum()
                + self.item_emb.weight.pow(2).sum()
            )

    @pytest.fixture
    def gpu_setup(self):
        """Tạo model, optimizer, sampler, loss trên GPU."""
        device = torch.device("cuda")

        model = self.DummyGNNModel(
            self.NUM_USERS, self.NUM_ITEMS, self.DIM
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        item_counts = {
            "view": torch.randint(1, 1000, (self.NUM_ITEMS,)),
            "cart": torch.randint(1, 100, (self.NUM_ITEMS,)),
            "purchase": torch.randint(1, 10, (self.NUM_ITEMS,)),
        }
        sampler = PopularityBiasedNegativeSampler(
            item_counts={**item_counts, "global": item_counts["view"]},
            num_items=self.NUM_ITEMS,
            alpha=0.75,
        )
        mt_loss = MultiTaskBPRLoss(
            behavior_counts=REES46_BEHAVIOR_COUNTS,
        ).to(device)

        trainer = BPRTrainingStep(
            multi_task_loss=mt_loss,
            neg_sampler=sampler,
            num_neg=4,
            amp_enabled=True,
        )

        return model, optimizer, trainer, device

    def _make_batch(self, device):
        """Tạo 1 batch giả lập với 3 behaviors, batch size khác nhau."""
        batch = {}
        for beh, bs in [("view", 2048), ("cart", 512), ("purchase", 128)]:
            batch[beh] = {
                "user": torch.randint(0, self.NUM_USERS, (bs,), device=device),
                "pos_item": torch.randint(0, self.NUM_ITEMS, (bs,), device=device),
            }
        return batch

    def test_five_steps_no_nan(self, gpu_setup):
        """5 training steps liên tiếp không được có NaN/Inf."""
        model, optimizer, trainer, device = gpu_setup

        for step_i in range(5):
            batch = self._make_batch(device)
            log = trainer.step(
                model, optimizer, batch,
                model.user_emb.weight, model.item_emb.weight,
            )
            assert not np.isnan(log["loss/total"]), f"NaN tại step {step_i}"
            assert not np.isinf(log["loss/total"]), f"Inf tại step {step_i}"

    def test_loss_decreases_over_steps(self, gpu_setup):
        """Loss trung bình 10 steps cuối < 10 steps đầu (model đang học)."""
        model, optimizer, trainer, device = gpu_setup

        losses_early = []
        losses_late = []
        for step_i in range(100):
            batch = self._make_batch(device)
            log = trainer.step(
                model, optimizer, batch,
                model.user_emb.weight, model.item_emb.weight,
            )
            if step_i < 10:
                losses_early.append(log["loss/total"])
            if step_i >= 90:
                losses_late.append(log["loss/total"])

        avg_early = np.mean(losses_early)
        avg_late = np.mean(losses_late)
        # Không assert strictly decreasing (noisy), chỉ check late < early
        assert avg_late < avg_early, (
            f"Loss không giảm: early={avg_early:.4f}, late={avg_late:.4f}"
        )

    def test_grad_norm_finite(self, gpu_setup):
        """Gradient norm phải hữu hạn sau clip."""
        model, optimizer, trainer, device = gpu_setup
        batch = self._make_batch(device)
        log = trainer.step(
            model, optimizer, batch,
            model.user_emb.weight, model.item_emb.weight,
        )
        assert np.isfinite(log["grad_norm"]), f"Gradient norm = {log['grad_norm']}"
        assert log["grad_norm"] <= trainer.max_grad_norm + 0.01  # clip hoạt động

    def test_amp_scale_positive(self, gpu_setup):
        """AMP scale factor phải dương (nếu = 0, training dừng)."""
        model, optimizer, trainer, device = gpu_setup
        batch = self._make_batch(device)
        log = trainer.step(
            model, optimizer, batch,
            model.user_emb.weight, model.item_emb.weight,
        )
        assert log["amp_scale"] > 0, f"AMP scale = {log['amp_scale']} — training sẽ dừng"

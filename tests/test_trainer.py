import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock
import os
import sys

# ĐÃ BỎ COMMENT ĐỂ IMPORT ĐƯỢC TRAINER
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from training.trainer import Trainer  # Cần đảm bảo file thật của bạn tên là trainer.py trong src/training

NUM_USERS = 100
NUM_ITEMS = 500
DIM = 16

class DummyModel(nn.Module):
    """Giả lập Model trả về embedding matrix cho toàn bộ user và item."""
    def __init__(self):
        super().__init__()
        self.user_emb = nn.Embedding(NUM_USERS, DIM)
        self.item_emb = nn.Embedding(NUM_ITEMS, DIM)
        # Khởi tạo trọng số ngẫu nhiên để test validate thay đổi
        nn.init.normal_(self.user_emb.weight)
        nn.init.normal_(self.item_emb.weight)

    def forward(self, graph_data):
        # Trainer gọi: user_emb, item_emb_all = self.model(graph_data)
        return self.user_emb.weight, self.item_emb.weight

@pytest.fixture
def device():
    # Force về CPU để chạy Unit Test nhanh chóng ở bất kỳ đâu
    return torch.device("cpu")

@pytest.fixture
def dummy_batch(device):
    """Batch chuẩn format chứa interaction của E-commerce."""
    return {
        "graph": torch.randn(1, 10).to(device), # Mock graph data
        "interactions": {
            "purchase": {
                "user": torch.tensor([1, 10, 20], device=device),
                "pos_item": torch.tensor([50, 100, 250], device=device)
            },
            "view": {
                "user": torch.tensor([2, 3], device=device),
                "pos_item": torch.tensor([10, 11], device=device)
            }
        }
    }

@pytest.fixture
def mock_bpr_step():
    """Giả lập BPRTrainingStep để không phải chạy logic Loss thật."""
    step_mock = MagicMock()
    # step() trả về dictionary chứa log_dict
    step_mock.step.return_value = {"loss/total": 0.35, "loss/purchase": 0.2}
    # ĐÃ FIX CẢNH BÁO CỦA PYTORCH 2.X
    step_mock.scaler = torch.amp.GradScaler('cuda', enabled=False) 
    return step_mock

@pytest.fixture
def trainer_config(tmp_path, mock_bpr_step):
    """Tạo config chuẩn, dùng tmp_path của pytest để tự động dọn rác checkpoint."""
    return {
        "epochs": 3,
        "patience": 2,
        "save_dir": str(tmp_path / "checkpoints"),
        "bpr_step": mock_bpr_step,
        "amp": False, # Tắt AMP khi test local/CPU để tránh lỗi GradScaler
        "use_wandb": False, # Ép tắt wandb để test độc lập hoàn toàn
    }

class TestTrainer:
    """Suite kiểm thử cho luồng chạy của class Trainer. Không còn phụ thuộc vào wandb."""

    def test_trainer_initialization(self, trainer_config, device):
        """Kiểm tra khởi tạo tạo đủ thư mục."""
        model = DummyModel().to(device)
        trainer = Trainer(model, None, [], [], device, trainer_config)
        
        assert Path(trainer_config["save_dir"]).exists()
        assert trainer.amp_enabled is False

    def test_train_epoch_success(self, trainer_config, dummy_batch, device):
        """Test train_epoch chạy qua dataloader và gọi bpr_step chính xác."""
        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        train_loader = [dummy_batch, dummy_batch] # 2 batches
        
        trainer = Trainer(model, optimizer, train_loader, [], device, trainer_config)
        avg_loss = trainer.train_epoch()
        
        # Loss mock là 0.35 cho mỗi batch -> trung bình là 0.35
        assert abs(avg_loss - 0.35) < 1e-5
        assert trainer_config["bpr_step"].step.call_count == 2

    def test_validate_metrics(self, trainer_config, dummy_batch, device):
        """Test logic full-catalogue ranking trả về Recall@10, Recall@20 hợp lệ."""
        model = DummyModel().to(device)
        val_loader = [dummy_batch]
        
        trainer = Trainer(model, None, [], val_loader, device, trainer_config)
        avg_loss, r10, r20 = trainer.validate()
        
        assert isinstance(avg_loss, float)
        assert 0.0 <= r10 <= 1.0
        assert 0.0 <= r20 <= 1.0
        assert r10 <= r20, "Recall@10 không thể lớn hơn Recall@20"

    def test_validate_skip_missing_purchase(self, trainer_config, device):
        """Nếu batch không có behavior 'purchase', validate phải bỏ qua an toàn."""
        model = DummyModel().to(device)
        batch_no_purchase = {
            "graph": torch.randn(1, 10).to(device),
            "interactions": {"view": {"user": torch.tensor([1]), "pos_item": torch.tensor([5])}}
        }
        
        trainer = Trainer(model, None, [], [batch_no_purchase], device, trainer_config)
        avg_loss, r10, r20 = trainer.validate()
        
        # Không có purchase user nào -> num_users = max(0, 1) = 1
        assert r10 == 0.0
        assert r20 == 0.0
        assert avg_loss == 0.0

    def test_checkpoint_save_and_load(self, trainer_config, device):
        """Đảm bảo trạng thái training (epoch, best_recall) được bảo toàn khi save/load."""
        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, [], [], device, trainer_config)
        
        # Mô phỏng trạng thái
        trainer.best_recall = 0.88
        trainer.save_checkpoint(epoch=7, is_best=True)
        
        # Khởi tạo Trainer mới và load lại
        new_trainer = Trainer(model, optimizer, [], [], device, trainer_config)
        loaded_epoch = new_trainer.load_checkpoint(Path(trainer_config["save_dir"]) / "checkpoint_best.pt")
        
        assert loaded_epoch == 7
        assert new_trainer.best_recall == 0.88

    def test_early_stopping_behavior(self, trainer_config, dummy_batch, device):
        """Test Early Stopping: Nếu Recall không tăng sau 'patience' lần, quá trình phải break."""
        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Force config
        trainer_config["epochs"] = 10
        trainer_config["patience"] = 1
        
        trainer = Trainer(model, optimizer, [dummy_batch], [dummy_batch], device, trainer_config)
        
        # Mock hàm validate trả về Recall giảm dần
        trainer.validate = MagicMock(side_effect=[
            (0.1, 0.5, 0.6), 
            (0.1, 0.4, 0.5), 
            (0.1, 0.3, 0.4) # Epoch này sẽ không bao giờ được chạy tới
        ])
        
        trainer.train()
        
        assert trainer.best_epoch == 0
        assert trainer.patience_counter >= trainer_config["patience"]
        # Chỉ chạy đúng 2 epoch (0 và 1)
        assert trainer.validate.call_count == 2
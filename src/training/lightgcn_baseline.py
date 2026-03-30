import os
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from src.core.contracts import EMBED_DIM

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 128, num_layers: int = 3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.num_layers = num_layers

        # Khởi tạo Embeddings chuẩn cho LightGCN 
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, norm_adj: torch.sparse.FloatTensor):
        # Nối user và item embeddings thành 1 ma trận (N_users + N_items, D)
        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs_list = [all_embs]

        # Message Passing (Nhân ma trận thưa)
        for layer in range(self.num_layers):
            all_embs = torch.sparse.mm(norm_adj, all_embs)
            embs_list.append(all_embs)

        # Trung bình cộng các layers (Đặc trưng cốt lõi của LightGCN)
        lightout = torch.stack(embs_list, dim=1).mean(dim=1)
        
        user_final, item_final = torch.split(lightout, [self.num_users, self.num_items])
        return user_final, item_final

def build_normalized_adj(user_np, item_np, num_users, num_items):
    R = sp.coo_matrix((np.ones(len(user_np)), (user_np, item_np)), shape=(num_users, num_items))
    
    # Ma trận A = [0 R; R^T 0]
    A = sp.bmat([[None, R], [R.T, None]])
    A = A.todok()
    A.setdiag(0)
    A = A.tocoo()

    # Tính D^{-0.5}
    rowsum = np.array(A.sum(1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    D_mat = sp.diags(d_inv)

    # D^{-0.5} A D^{-0.5}
    norm_adj = D_mat.dot(A).dot(D_mat).tocoo()
    
    # Chuyển sang PyTorch Sparse Tensor
    indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

def bpr_loss(users, pos_items, neg_items, user_emb, item_emb):
    u_e = user_emb[users]
    pos_e = item_emb[pos_items]
    neg_e = item_emb[neg_items]

    pos_scores = torch.sum(u_e * pos_e, dim=1)
    neg_scores = torch.sum(u_e * neg_e, dim=1)
    
    # L2 Regularization (tránh overfitting)
    reg_loss = (1/2) * (u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / float(len(users))
    
    # BPR Loss = -mean(log(sigmoid(pos_score - neg_score)))
    bpr = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    return bpr, reg_loss

def sample_negatives_fast(num_samples, num_items, device):
    return torch.randint(0, num_items, (num_samples,), device=device)

def train(data_dir: str, epochs: int, batch_size: int, lr: float):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device.type.upper()}")

    # 1. Load Dữ liệu Purchase-only
    train_u = np.load(os.path.join(data_dir, "loo_purchase_train_src.npy"))
    train_i = np.load(os.path.join(data_dir, "loo_purchase_train_dst.npy"))
    
    num_users = int(max(train_u)) + 1
    num_items = int(max(train_i)) + 1
    print(f"  - Edges: {len(train_u):,}")
    print(f"  - Users: {num_users:,} | Items: {num_items:,}")

    with open(os.path.join(data_dir, "train_mask.pkl"), 'rb') as f:
        train_mask = pickle.load(f)

    # 2. Build Graph & Model
    norm_adj = build_normalized_adj(train_u, train_i, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, dim=EMBED_DIM, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    num_edges = len(train_u)
    indices = np.arange(num_edges)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        
        total_loss = 0.0
        start_time = time.time()

        for i in range(0, num_edges, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_users = train_u[batch_idx]
            b_pos_items = train_i[batch_idx]
            
            # GPU Fast Sampling (THÊM VÀO ĐÂY)
            b_neg_items_t = sample_negatives_fast(len(b_users), num_items, device)
            
            # Move to GPU
            b_users_t = torch.tensor(b_users, device=device)
            b_pos_items_t = torch.tensor(b_pos_items, device=device)
            b_neg_items_t = b_neg_items_t.to(device)

            optimizer.zero_grad()
            
            # Forward
            user_emb_final, item_emb_final = model(norm_adj)
            
            # Compute Loss
            bpr, reg = bpr_loss(b_users_t, b_pos_items_t, b_neg_items_t, user_emb_final, item_emb_final)
            loss = bpr + 1e-4 * reg # 1e-4 là weight decay chuẩn
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / (num_edges / batch_size)
        print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # AUTO-SAVE mỗi 10 Epochs
        if epoch % 10 == 0:
            ckpt_path = "checkpoints/lightgcn_purchase_latest.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Đã ghi đè Checkpoint mới nhất tại: {ckpt_path} (Epoch {epoch})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục chứa loo npy files")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size, args.lr)
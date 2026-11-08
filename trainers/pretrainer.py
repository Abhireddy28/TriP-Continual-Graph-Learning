import torch
import torch.nn.functional as F

def random_edge_dropout(edge_index, p=0.2):
    """Independently drops edges with probability p."""
    if p <= 0.0 or edge_index.shape[1] == 0:
        return edge_index
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask]

def random_feature_masking(x, p=0.2):
    """Independently masks feature dimensions with probability p."""
    if p <= 0.0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask

class SelfSupervisedPretrainer:
    """
    Performs Self-Supervised Contrastive Pre-training on Task 1 (Section 4.2).
    Uses Edge Dropout and Feature Masking to produce two views G_1^(1) and G_1^(2).
    Trained with InfoNCE loss without task label leakage, then freezes the GNN.
    """
    def __init__(self, model, lr=0.001, epochs=80, edge_drop_p=0.2, 
                 feat_mask_p=0.2, temperature=0.5, weight_decay=5e-4):
        self.model = model
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.edge_drop_p = float(edge_drop_p)
        self.feat_mask_p = float(feat_mask_p)
        self.temperature = float(temperature)
        self.weight_decay = float(weight_decay)

    def train(self, x, edge_index, pretrain_nodes):
        self.model.backbone.train()
        optimizer = torch.optim.Adam(
            self.model.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        print(f"[Pre-training] Starting self-supervised pre-training on Task 1 "
              f"({len(pretrain_nodes)} nodes, {self.epochs} epochs)...")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            # Mini-batch subsampling for contrastive InfoNCE if pretrain nodes > 4096 (e.g. Arxiv/Reddit)
            if len(pretrain_nodes) > 4096:
                perm = torch.randperm(len(pretrain_nodes), device=x.device)[:4096]
                batch_nodes = pretrain_nodes[perm]
            else:
                batch_nodes = pretrain_nodes

            # View 1
            x_1 = random_feature_masking(x, self.feat_mask_p)
            edge_1 = random_edge_dropout(edge_index, self.edge_drop_p)
            h_1 = self.model.backbone(x_1, edge_1)
            h_1 = F.normalize(h_1[batch_nodes], p=2, dim=-1)

            # View 2
            x_2 = random_feature_masking(x, self.feat_mask_p)
            edge_2 = random_edge_dropout(edge_index, self.edge_drop_p)
            h_2 = self.model.backbone(x_2, edge_2)
            h_2 = F.normalize(h_2[batch_nodes], p=2, dim=-1)

            # InfoNCE contrastive loss over nodes in pretrain set
            sim_matrix = torch.matmul(h_1, h_2.t()) / self.temperature
            labels = torch.arange(len(batch_nodes), device=x.device)
            loss = F.cross_entropy(sim_matrix, labels)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == self.epochs:
                print(f"  [Epoch {epoch:03d}/{self.epochs:03d}] Pre-train InfoNCE Loss: {loss.item():.4f}")

        # Permanently freeze backbone parameters
        self.model.freeze_backbone()
        print("[Pre-training] Completed successfully.")

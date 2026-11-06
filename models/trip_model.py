import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GCNBackbone, SGCBackbone
from .prompts import FeaturePrompt, ClassPrompt
from .prototypes import TaskPrototypeBuffer, ClassPrototypeBuffer

class TriPFramework(nn.Module):
    """
    TriP: Triple-Prompt Framework Aligning Pre-training and Class-Incremental Objectives.
    
    Components:
      1. Pre-trained and Frozen GNN Backbone (GCN / SGC)
      2. Task-specific Feature Prompts (GPF-plus)
      3. Task-specific Class Prompts (Gating)
      4. Unified Prompt Template Loss (Positive & Negative Prototype Alignment)
      5. Task Prototype and Class Prototype Buffers
    """
    def __init__(self, in_features, hidden_dim=256, backbone_type="GCN", 
                 prompt_k=2, temperature=0.5, alpha=0.4, device="cpu"):
        super(TriPFramework, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.alpha = alpha
        self.prompt_k = prompt_k
        self.device = device

        # 1. Backbone GNN
        if backbone_type.upper() == "GCN":
            self.backbone = GCNBackbone(in_features, hidden_dim=hidden_dim).to(device)
        elif backbone_type.upper() == "SGC":
            self.backbone = SGCBackbone(in_features, hidden_dim=hidden_dim).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        # 2. Prompts storage per task
        self.feature_prompts = nn.ModuleDict()
        self.class_prompts = nn.ModuleDict()

        # 3. Buffers for prototypes
        self.task_prototype_buffer = TaskPrototypeBuffer()
        self.class_prototype_buffer = ClassPrototypeBuffer()

    def freeze_backbone(self):
        """Freezes all parameters of the GNN backbone permanently after pre-training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("[TriP] Backbone GNN successfully frozen.")

    def add_task_prompts(self, task_id):
        """Creates lightweight Feature and Class prompt modules for task t."""
        str_id = str(task_id)
        fp = FeaturePrompt(self.in_features, num_prompts_k=self.prompt_k).to(self.device)
        cp = ClassPrompt(self.hidden_dim).to(self.device)
        self.feature_prompts[str_id] = fp
        self.class_prompts[str_id] = cp

    def get_task_prompts(self, task_id):
        str_id = str(task_id)
        return self.feature_prompts[str_id], self.class_prompts[str_id]

    def encode(self, x, edge_index, task_id):
        """
        Calculates class-prompted representations:
          h_tilde_v = f_theta(x_v + p_v) * q_t^cls  (Eq. 15)
        """
        str_id = str(task_id)
        fp = self.feature_prompts[str_id]
        cp = self.class_prompts[str_id]

        # 1. Feature Prompt modification
        x_prompted = fp(x)
        # 2. Frozen GNN encoding
        h = self.backbone(x_prompted, edge_index)
        # 3. Class Prompt feature-wise gating
        h_tilde = cp(h)
        # L2-normalize for stable cosine similarity
        h_tilde = F.normalize(h_tilde, p=2, dim=-1)
        return h_tilde

    def compute_unified_prompt_loss(self, h_tilde, y, train_nodes, current_classes, previous_classes):
        """
        Unified Prompt Template Loss (Section 4.5, Eq. 9-14):
          L_v = -log (s_v^+ / (s_v^+ + s_v^{-,curr})) - alpha * log (s_v^+ / (s_v^+ + s_v^{-,pre}))
        """
        train_h = h_tilde[train_nodes]
        train_y = y[train_nodes]

        # Current task class prototypes: shape (len(current_classes), D)
        curr_protos = [self.class_prototype_buffer.class_prototypes[c] for c in current_classes]
        curr_protos = torch.stack(curr_protos, dim=0).to(self.device)

        # Dot product with prototypes (cosine similarity since normalized): (N_train, |C_t|)
        sims_curr = torch.matmul(train_h, curr_protos.t()) / self.temperature

        # Create mapping of current class IDs to local indices [0, 1, ...]
        c_map = {c: i for i, c in enumerate(current_classes)}
        local_y = torch.tensor([c_map[val.item()] for val in train_y], device=self.device)

        # Positive similarity: exp(h_tilde_v^T c_{t, y_v} / tau)
        s_pos = torch.exp(sims_curr[torch.arange(len(train_nodes)), local_y])

        # Current negative similarity: sum_{c' != y_v} exp(h_tilde_v^T c_{t, c'} / tau)
        exp_sims_curr = torch.exp(sims_curr)
        # Sum of all in curr task minus positive
        s_neg_curr = exp_sims_curr.sum(dim=-1) - s_pos
        s_neg_curr = torch.clamp(s_neg_curr, min=1e-8)

        loss_curr = -torch.log(s_pos / (s_pos + s_neg_curr + 1e-8))

        # Previous classes loss term (Eq. 11 & Eq. 13)
        if len(previous_classes) > 0:
            prev_protos = [self.class_prototype_buffer.class_prototypes[c] for c in previous_classes]
            prev_protos = torch.stack(prev_protos, dim=0).to(self.device)
            sims_prev = torch.matmul(train_h, prev_protos.t()) / self.temperature
            s_neg_pre = torch.exp(sims_prev).sum(dim=-1)
            s_neg_pre = torch.clamp(s_neg_pre, min=1e-8)
            loss_pre = -torch.log(s_pos / (s_pos + s_neg_pre + 1e-8))
            total_loss = loss_curr + self.alpha * loss_pre
        else:
            total_loss = loss_curr

        return total_loss.mean()

    def predict(self, x, edge_index, test_nodes, observed_classes):
        """
        Class-incremental inference without task labels (Section 4.6):
          1. Retrieve closest task prototype: t* = argmax cos(m_test, m_i)
          2. Compute embeddings with prompts from task t*
          3. Predict class with highest prototype similarity across ALL observed classes
        """
        self.eval()
        with torch.no_grad():
            test_x = x[test_nodes]
            # 1. Query most relevant task prompt
            t_star = self.task_prototype_buffer.query_task(test_x)

            # 2. Extract prompted node embeddings using task t* prompts
            h_tilde = self.encode(x, edge_index, t_star)
            test_h = h_tilde[test_nodes]

            # 3. Classify against all observed class prototypes
            all_protos = self.class_prototype_buffer.get_all_prototypes_matrix(observed_classes).to(self.device)
            # Scores: (N_test, |observed_classes|)
            scores = torch.matmul(test_h, all_protos.t()) / self.temperature
            preds_idx = torch.argmax(scores, dim=-1)

            # Map predicted index back to actual class ID
            pred_classes = torch.tensor([observed_classes[i] for i in preds_idx.cpu().numpy()], device=self.device)
            return pred_classes

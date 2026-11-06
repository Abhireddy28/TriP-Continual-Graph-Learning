import torch
import torch.nn.functional as F

class TaskPrototypeBuffer:
    """
    Maintains and queries task-level prototypes:
      m_t = 1/|V_t^train| sum_{v in V_t^train} x_v    (Eq. 3)
    Used during inference to select the most relevant task prompt without task IDs:
      m_test = 1/|V_test| sum_{v in V_test} x_v       (Eq. 16)
      t* = argmax cos(m_test, m_i)                    (Eq. 17)
    """
    def __init__(self):
        self.prototypes = {}  # task_id -> tensor of shape (d,)

    def register_task_prototype(self, task_id, x, train_nodes):
        task_x = x[train_nodes]
        m_t = task_x.mean(dim=0).detach()
        self.prototypes[task_id] = m_t

    def query_task(self, test_x):
        """
        Given test node features of shape (N, d) or (d,),
        identifies the most similar task ID t*.
        """
        if test_x.dim() == 2:
            m_test = test_x.mean(dim=0)
        else:
            m_test = test_x

        best_task_id = 0
        best_sim = -float('inf')

        for task_id, m_t in self.prototypes.items():
            sim = F.cosine_similarity(m_test.unsqueeze(0), m_t.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_task_id = task_id

        return best_task_id

class ClassPrototypeBuffer:
    """
    Maintains class-level prototypes for the Unified Prompt Template and Inference:
      c_{t,c} = 1/|V_c| sum_{v in V_c} h_tilde_v       (Eq. 8)
    """
    def __init__(self):
        self.class_prototypes = {}  # class_id -> tensor of shape (D,)

    def update_prototypes(self, h_tilde, y, train_nodes, current_classes):
        """
        Updates prototypes for classes in the current task.
        """
        train_h = h_tilde[train_nodes]
        train_y = y[train_nodes]

        for c in current_classes:
            idx = (train_y == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                c_proto = train_h[idx].mean(dim=0).detach()
                # Normalize prototype vector for stable cosine similarity
                c_proto = F.normalize(c_proto, p=2, dim=-1)
                self.class_prototypes[c] = c_proto

    def get_all_prototypes_matrix(self, observed_classes):
        """
        Returns stacked prototypes tensor of shape (|observed_classes|, D)
        and matching class id list.
        """
        proto_list = []
        for c in observed_classes:
            proto_list.append(self.class_prototypes[c])
        return torch.stack(proto_list, dim=0)

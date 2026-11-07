import torch
import torch.nn.functional as F

def compute_sinkhorn_wasserstein(X, Y, reg=0.1, max_iter=20):
    """
    Computes regularized 2-Wasserstein distance between empirical distributions
    represented by point sets X and Y via the Sinkhorn-Knopp algorithm.
    X: shape (N, d), Y: shape (M, d)
    """
    N = X.shape[0]
    M = Y.shape[0]
    a = torch.ones(N, device=X.device) / N
    b = torch.ones(M, device=Y.device) / M

    # Pairwise squared Euclidean cost matrix
    C = torch.cdist(X, Y, p=2).pow(2)
    # Kernel matrix
    K = torch.exp(-C / reg)

    u = torch.ones_like(a)
    for _ in range(max_iter):
        v = b / (torch.matmul(K.t(), u) + 1e-8)
        u = a / (torch.matmul(K, v) + 1e-8)

    transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
    w_dist = (transport_plan * C).sum()
    return w_dist.item()

class TaskPrototypeBuffer:
    """
    [Upgrade 3] Dual-Mode Task Prototype Buffer:
      Mode 1: Standard TriP centroid mean-pooling (Eq. 3 & Eq. 17)
      Mode 2: Optimal Transport / Wasserstein Distribution Matching via Sinkhorn distance
              Eliminates task confusion caused by mean-pooling compression (Section 5.7)
    """
    def __init__(self, num_support_nodes=24, use_wasserstein=True):
        self.prototypes = {}          # task_id -> mean tensor of shape (d,)
        self.support_samples = {}     # task_id -> support distribution tensor of shape (S, d)
        self.num_support_nodes = num_support_nodes
        self.use_wasserstein = use_wasserstein

    def register_task_prototype(self, task_id, x, train_nodes):
        task_x = x[train_nodes]
        m_t = task_x.mean(dim=0).detach()
        self.prototypes[task_id] = m_t

        # Store support distribution for Optimal Transport matching
        n_samples = min(len(train_nodes), self.num_support_nodes)
        # Uniformly sampled representative support indices
        perm = torch.randperm(len(train_nodes))[:n_samples]
        self.support_samples[task_id] = task_x[perm].detach()

    def query_task(self, test_x):
        """
        Given test node features of shape (N, d) or (d,),
        identifies the most similar task ID t*.
        """
        # If Wasserstein mode is enabled and we have multi-node batch
        if self.use_wasserstein and test_x.dim() == 2 and test_x.shape[0] > 1:
            return self.query_task_wasserstein(test_x)
        
        # Standard TriP Centroid Cosine Matching (Eq. 16 & 17)
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

    def query_task_wasserstein(self, test_x):
        """
        Queries task ID via minimal Wasserstein transport cost to support distributions.
        """
        n_test_samples = min(test_x.shape[0], self.num_support_nodes)
        perm = torch.randperm(test_x.shape[0])[:n_test_samples]
        sub_test_x = test_x[perm]

        best_task_id = 0
        min_w_dist = float('inf')

        for task_id, supp in self.support_samples.items():
            dist = compute_sinkhorn_wasserstein(sub_test_x, supp, reg=0.1, max_iter=15)
            if dist < min_w_dist:
                min_w_dist = dist
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

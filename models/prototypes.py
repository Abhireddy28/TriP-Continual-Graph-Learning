import torch
import torch.nn.functional as F

def compute_sinkhorn_wasserstein(X, Y, reg=0.2, max_iter=20):
    """
    Computes regularized 2-Wasserstein distance between empirical distributions
    represented by point sets X and Y via the Sinkhorn-Knopp algorithm.
    Uses L2-normalized cosine distance cost to prevent dimensional overflow/underflow.
    X: shape (N, d), Y: shape (M, d)
    """
    # L2 normalize vectors to prevent dimensional underflow when d is large (e.g. d=8710)
    X_norm = F.normalize(X, p=2, dim=-1)
    Y_norm = F.normalize(Y, p=2, dim=-1)

    # Cosine distance cost matrix bounded in [0, 2]
    C = 1.0 - torch.matmul(X_norm, Y_norm.t())
    C = torch.clamp(C, min=0.0)

    # Kernel matrix
    K = torch.exp(-C / reg)

    N = X.shape[0]
    M = Y.shape[0]
    a = torch.ones(N, device=X.device) / N
    b = torch.ones(M, device=Y.device) / M

    u = torch.ones_like(a)
    for _ in range(max_iter):
        v = b / (torch.matmul(K.t(), u) + 1e-8)
        u = a / (torch.matmul(K, v) + 1e-8)

    transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
    w_dist = (transport_plan * C).sum()
    return w_dist.item()

class TaskPrototypeBuffer:
    """
    [TAPOR Module] Dual-Mode Margin-Aware Task Prototype Router:
      Mode 1: Standard centroid cosine matching (TriP / TPP baseline)
      Mode 2: Margin-Aware Optimal Transport (TAPOR):
              Computes top-1 and top-2 cosine similarities: Delta = s_1 - s_2.
              - If Delta >= delta (clear winner): routes via fast centroid cosine.
              - If Delta < delta (near-tie ambiguity): triggers Sinkhorn Wasserstein fallback
                over support distribution S_t to resolve task identity without confusion.
      
      Support Set Selection:
        - 'topk_l2': K nodes closest to centroid m_t
        - 'random': uniformly sampled random nodes
        - 'empty': S = 0 (centroid-only baseline)
    """
    def __init__(self, num_support_nodes=24, use_wasserstein=True, 
                 margin_threshold=0.08, ot_weight=0.15, support_mode="topk_l2"):
        self.prototypes = {}          # task_id -> mean tensor of shape (d,)
        self.support_samples = {}     # task_id -> support distribution tensor of shape (S, d)
        self.num_support_nodes = num_support_nodes
        self.use_wasserstein = use_wasserstein
        self.margin_threshold = margin_threshold
        self.ot_weight = ot_weight
        self.support_mode = support_mode.lower()
        
        # Diagnostic tracking
        self.total_queries = 0
        self.sinkhorn_triggers = 0
        self.task_id_correct_count = 0

    def reset_stats(self):
        self.total_queries = 0
        self.sinkhorn_triggers = 0
        self.task_id_correct_count = 0

    def register_task_prototype(self, task_id, x, train_nodes):
        task_x = x[train_nodes]
        m_t = task_x.mean(dim=0).detach()
        self.prototypes[task_id] = m_t

        if self.support_mode == "empty" or self.num_support_nodes <= 0:
            self.support_samples[task_id] = None
            return

        n_samples = min(len(train_nodes), self.num_support_nodes)
        if self.support_mode == "topk_l2":
            # Nodes with smallest L2 distance to centroid m_t (medoids)
            dists = torch.norm(task_x - m_t.unsqueeze(0), dim=-1)
            topk_idx = torch.topk(dists, k=n_samples, largest=False).indices
            self.support_samples[task_id] = task_x[topk_idx].detach()
        else:
            # Uniform random sampling
            perm = torch.randperm(len(train_nodes))[:n_samples]
            self.support_samples[task_id] = task_x[perm].detach()

    def query_task(self, test_x, true_task_id=None):
        """
        Given test node features of shape (N, d) or (d,),
        identifies the most relevant task ID t*.
        Returns t_star.
        """
        self.total_queries += 1

        if test_x.dim() == 2:
            m_test = test_x.mean(dim=0)
        else:
            m_test = test_x

        # 1. Compute cosine similarities with all registered centroids
        sims = {}
        for task_id, m_t in self.prototypes.items():
            sim = F.cosine_similarity(m_test.unsqueeze(0), m_t.unsqueeze(0)).item()
            sims[task_id] = sim

        sorted_tasks = sorted(sims.items(), key=lambda item: item[1], reverse=True)
        top1_task, top1_sim = sorted_tasks[0]

        # If only 1 task seen or OT disabled or empty support, return top-1 centroid
        if len(sorted_tasks) <= 1 or not self.use_wasserstein or self.support_mode == "empty":
            t_star = top1_task
            if true_task_id is not None and t_star == true_task_id:
                self.task_id_correct_count += 1
            return t_star

        top2_task, top2_sim = sorted_tasks[1]
        margin = top1_sim - top2_sim

        # 2. Near-Tie Margin Gate: If margin >= delta, unambiguous split -> fast centroid routing
        if margin >= self.margin_threshold:
            t_star = top1_task
            if true_task_id is not None and t_star == true_task_id:
                self.task_id_correct_count += 1
            return t_star

        # 3. Near-Tie Fallback: Trigger Optimal Transport (Sinkhorn) on ambiguous tasks
        self.sinkhorn_triggers += 1
        n_test_samples = min(test_x.shape[0], self.num_support_nodes)
        perm = torch.randperm(test_x.shape[0])[:n_test_samples]
        sub_test_x = test_x[perm]

        best_task_id = top1_task
        best_score = -float('inf')

        # Evaluate candidate tasks that are within near-tie range
        candidate_tasks = [t for t, s in sorted_tasks if (top1_sim - s) <= self.margin_threshold * 1.5]
        if len(candidate_tasks) < 2:
            candidate_tasks = [top1_task, top2_task]

        w_weight = self.ot_weight
        for task_id in candidate_tasks:
            supp = self.support_samples.get(task_id, None)
            if supp is None:
                continue
            cos_s = sims[task_id]
            w_dist = compute_sinkhorn_wasserstein(sub_test_x, supp, reg=0.2, max_iter=15)
            score = cos_s - w_weight * w_dist
            if score > best_score:
                best_score = score
                best_task_id = task_id

        t_star = best_task_id
        if true_task_id is not None and t_star == true_task_id:
            self.task_id_correct_count += 1
        return t_star

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
        Supports both full-graph representations and indexed sub-representations.
        """
        if h_tilde.shape[0] == len(train_nodes):
            train_h = h_tilde
            train_y = y[train_nodes] if y.shape[0] != len(train_nodes) else y
        else:
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

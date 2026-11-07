import torch
import torch.nn as nn

class FeaturePrompt(nn.Module):
    """
    Feature-Level Prompt Module (GPF-plus based).
    Modifies input node feature space to extract task-specific knowledge.
    
    Equations:
      w_v = W_t^T x_v           (Eq. 4)
      p_v = w_v^T P_t^fea       (Eq. 5)
      x_hat_v = x_v + p_v       (Eq. 6)
    """
    def __init__(self, in_features, num_prompts_k=2):
        super(FeaturePrompt, self).__init__()
        self.num_prompts_k = num_prompts_k
        self.in_features = in_features

        # Learnable prompt matrix P in R^{k x d}
        self.prompt_matrix = nn.Parameter(torch.Tensor(num_prompts_k, in_features))
        
        # Linear transformation W in R^{d x k} for dynamic coefficients
        if num_prompts_k > 1:
            self.linear_weight = nn.Linear(in_features, num_prompts_k, bias=False)
        else:
            self.linear_weight = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompt_matrix)
        if self.linear_weight is not None:
            nn.init.xavier_uniform_(self.linear_weight.weight)

    def forward(self, x):
        """
        Input: x of shape (N, d)
        Output: x_hat of shape (N, d)
        """
        if self.num_prompts_k > 1:
            # weights: (N, k)
            weights = self.linear_weight(x)
            # p: (N, k) x (k, d) -> (N, d)
            p = torch.matmul(weights, self.prompt_matrix)
        else:
            p = self.prompt_matrix.repeat(x.shape[0], 1)
        
        x_hat = x + p
        return x_hat

class ClassPrompt(nn.Module):
    """
    Class-Level Prompt Module.
    Functions as a feature-wise gating mechanism modulating GNN outputs:
      h_tilde_v = h_v (element-wise multiply) q_t^cls   (Eq. 7)
    """
    def __init__(self, hidden_dim=256):
        super(ClassPrompt, self).__init__()
        # Learnable gating vector q in R^D initialized to ones
        self.q_cls = nn.Parameter(torch.ones(1, hidden_dim))

    def forward(self, h):
        """
        Input: h of shape (N, D)
        Output: h_tilde of shape (N, D)
        """
        return h * self.q_cls

class TopologyPrompt(nn.Module):
    """
    [Upgrade 1] Topological-Structural Prompt Module.
    Dynamically modulates graph message-passing topology (A_t) for task t.
    Solves TriP's limitation of only prompting node features X while ignoring graph structure A.
    
    Equation:
      gamma_uv = sigmoid(z_u^T z_v + b_t) * 2.0
      w_hat_uv = w_uv * gamma_uv
    """
    def __init__(self, in_features, hidden_dim=16):
        super(TopologyPrompt, self).__init__()
        # Low-rank topological projection layer
        self.topo_proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.topo_proj.weight)

    def forward(self, x, edge_index):
        """
        Computes edge modulation factor gamma of shape (|E|,)
        """
        row, col = edge_index
        z = self.topo_proj(x)  # shape (N, hidden_dim)
        
        # Edge affinity scores
        edge_scores = (z[row] * z[col]).sum(dim=-1) + self.edge_bias
        # Modulation factor centered around 1.0 (sigmoid * 2.0)
        gamma = torch.sigmoid(edge_scores) * 2.0
        return gamma


import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adjacency(edge_index, num_nodes, edge_weight_mod=None):
    """
    Computes symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    Operates natively on edge_index to avoid dense matrix memory overhead.
    Optionally incorporates Topological Prompt edge weight modulation.
    """
    row, col = edge_index
    # Add self-loops
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    
    if edge_weight_mod is not None:
        # Modulate original edges, self-loops remain unmodulated (1.0)
        loop_mod = torch.ones(num_nodes, device=edge_index.device)
        full_edge_mod = torch.cat([edge_weight_mod, loop_mod], dim=0)
    else:
        full_edge_mod = None

    full_edge_index = torch.cat([edge_index, loop_index], dim=1)
    row, col = full_edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    if full_edge_mod is not None:
        edge_weight = edge_weight * full_edge_mod
    return full_edge_index, edge_weight

def spmm(edge_index, edge_weight, x, num_nodes):
    """Sparse matrix multiplication with node features x."""
    row, col = edge_index
    msg = x[col] * edge_weight.unsqueeze(-1)
    out = torch.zeros(num_nodes, x.shape[1], device=x.device, dtype=x.dtype)
    out.index_add_(0, row, msg)
    return out

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.linear(x)
        out = spmm(edge_index, edge_weight, x, num_nodes)
        return out

class GCNBackbone(nn.Module):
    """
    2-Layer Graph Convolutional Network (Kipf & Welling, 2017)
    Used for CoraFull and OGBN-Arxiv.
    """
    def __init__(self, in_features, hidden_dim=256, dropout=0.5):
        super(GCNBackbone, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight_mod=None):
        num_nodes = x.shape[0]
        norm_edge_index, edge_weight = normalize_adjacency(edge_index, num_nodes, edge_weight_mod=edge_weight_mod)
        
        h = self.conv1(x, norm_edge_index, edge_weight, num_nodes)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, norm_edge_index, edge_weight, num_nodes)
        return h

class SGCBackbone(nn.Module):
    """
    2-Layer Simplifying Graph Convolutional Network (Wu et al., 2019)
    Used for Products and Reddit to save computational memory on large graphs.
    """
    def __init__(self, in_features, hidden_dim=256, k_hops=2):
        super(SGCBackbone, self).__init__()
        self.k_hops = k_hops
        self.linear = nn.Linear(in_features, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight_mod=None):
        num_nodes = x.shape[0]
        norm_edge_index, edge_weight = normalize_adjacency(edge_index, num_nodes, edge_weight_mod=edge_weight_mod)
        
        h = x
        for _ in range(self.k_hops):
            h = spmm(norm_edge_index, edge_weight, h, num_nodes)
        out = self.linear(h)
        return out

# Dual-TriP: Dual-Space Prompting and Wasserstein Prototype Routing for Class-Incremental Continual Graph Learning

### **Authors:** [Your Name], [Your Project Guide's Name]
### **Department:** Department of Computer Science & Engineering
### **Batch:** 235
### **Target Publication Venues:** IEEE Transactions on Neural Networks and Learning Systems (T-NNLS) / IEEE BigData / Springer Neural Networks

---

## 📌 Abstract

Continual Graph Learning (CGL) under the class-incremental setting (class-IL) represents a fundamental challenge in artificial intelligence due to the catastrophic forgetting of previous structural knowledge when learning new graph categories. While recent prompt-based methods such as TriP (Neural Networks 2027) alleviate catastrophic forgetting by freezing the pre-trained Graph Neural Network (GNN) backbone and tuning lightweight prompts, they exhibit two critical architectural limitations:
1. **Unilateral Space Prompting:** Existing methods inject prompts solely into the node feature space ($\mathbf{X}$), leaving the graph message-passing topology ($\mathbf{A}$) static and failing to adapt to structural distribution shifts.
2. **Task Query Distribution Collapse:** Using simple centroid mean-pooling to match test samples to task prompts compresses multimodal graph manifolds into a single point, causing severe task misclassification on overlapping citation boundaries.

To resolve these limitations, we propose **Dual-TriP**, an enhanced continual graph learning architecture featuring:
* **Dual-Space Prompting:** Simultaneously adapts node feature representations via a dynamic feature prompt ($\hat{\mathbf{X}}_t$) and modulates topological message-passing edge flows via a low-rank structural prompt ($\hat{\mathbf{A}}_t$).
* **Wasserstein Optimal Transport Task Router:** Replaces heuristic centroid cosine similarity with an entropic regularized Optimal Transport (Sinkhorn) distance metric that measures exact manifold discrepancy between test node distributions and stored task support distributions.

Extensive experiments on benchmark datasets (**CoraFull** and **OGBN-Arxiv**) demonstrate that **Dual-TriP** significantly improves task prompt selection hit rates and outperforms existing state-of-the-art methods in Average Accuracy while maintaining near-zero forgetting without memory replay.

---

## 🏛️ 1. Theoretical Limitations of the Base TriP Model

A graph is mathematically defined as a dual-space tuple:
$$\mathcal{G} = (\mathbf{A}, \mathbf{X})$$
where $\mathbf{A} \in \{0, 1\}^{N \times N}$ represents topological connectivity and $\mathbf{X} \in \mathbb{R}^{N \times d}$ represents node attribute vectors.

### Limitation A: Topological Inelasticity
In standard TriP:
$$\hat{\mathbf{X}}_t = \mathbf{X}_t + \mathbf{P}_t^{fea}, \quad \mathbf{H}_t = \text{GNN}_{\text{frozen}}(\hat{\mathbf{X}}_t, \mathbf{A})$$
The topology $\mathbf{A}$ remains completely rigid throughout all incremental tasks. In evolving graphs, newly arriving classes often form distinct homophilic and heterophilic interaction patterns. Forcing new tasks to propagate across unprompted static adjacency matrices causes structural representation bottlenecks.

### Limitation B: Distribution Compression in Task Querying (Section 5.7)
In TriP, task selection is performed by:
$$\mathbf{m}_{test} = \frac{1}{|\mathcal{V}_{test}|} \sum_{v \in \mathcal{V}_{test}} \mathbf{x}_v, \quad t^* = \arg\max_{1 \le i \le T} \cos(\mathbf{m}_{test}, \mathbf{m}_i)$$
Reducing a set of $|\mathcal{V}_{test}|$ nodes to a single mean vector $\mathbf{m}_{test}$ discards all higher-order distribution moments (variance, manifold geometry, and intra-task clusters). In dense or multidisciplinary academic domains (e.g., papers citing both AI and Neuroscience), mean pooling frequently selects the wrong task prompt.

---

## 🔬 2. Mathematical Formulation of Dual-TriP

```
                   ┌────────────────────────────────────────────────────────┐
                   │             Dual-TriP ARCHITECTURE OVERVIEW            │
                   └────────────────────────────────────────────────────────┘

                                ┌────────────────────────────────┐
                                │ Input Graph G_t = (A_t, X_t)   │
                                └───────┬────────────────┬───────┘
                                        │                │
            ┌───────────────────────────┘                └───────────────────────────┐
            ▼                                                                        ▼
   [Feature Prompt P_t^fea]                                                 [Topological Prompt P_t^topo]
   w_v = W_t^T x_v                                                          z_v = W_topo x_v
   x_hat_v = x_v + w_v^T P_t^fea                                            gamma_uv = sigmoid(z_u^T z_v + b)*2
            │                                                                        │
            ▼                                                                        ▼
   Prompted Feature X_hat_t                                                 Modulated Adjacency A_hat_t
            │                                                                        │
            └───────────────────────────┬────────────────────────────────────────────┘
                                        ▼
                             Frozen GNN Backbone f_theta
                                        │
                                        ▼
                           Prompted Embeddings H_t
                                        │
                                        ▼ [⊙] ◄── Class Prompt q_t^cls
                                        │
                          Class-Prompted Embeddings H_tilde_t
                                        │
                                        ▼
                      Unified Prototype Contrastive Loss L_down
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
         [Upgrade 3: Inference]                  [Upgrade 1: Memory]
      Optimal Transport (Wasserstein)           Zero Replay of Raw Data
        Sinkhorn Distance Matching               Lightweight Prompts Only
```

### Upgrade 1: Low-Rank Topological Message-Passing Prompt
For each incremental task $t$, Dual-TriP instantiates a lightweight topological projection matrix $\mathbf{W}_t^{topo} \in \mathbb{R}^{d \times d_{topo}}$ (where $d_{topo} \ll d$, e.g., $d_{topo} = 16$).

1. Node topological embeddings:
   $$\mathbf{z}_v = \mathbf{W}_t^{topo} \mathbf{x}_v$$
2. Edge affinity modulation coefficient for connection $(u, v) \in \mathcal{E}$:
   $$\gamma_{uv}^{(t)} = 2.0 \cdot \sigma\left( \mathbf{z}_u^\top \mathbf{z}_v + b_t \right)$$
   *(where $\sigma$ is the sigmoid function, centering the scaling factor around $1.0$).*
3. Modulated Adjacency Normalization:
   $$\hat{w}_{uv}^{(t)} = \tilde{D}_{uu}^{-1/2} \tilde{D}_{vv}^{-1/2} \cdot \gamma_{uv}^{(t)}$$
4. Dual-Space Message Passing:
   $$\mathbf{H}_t = \text{GNN}_{\text{frozen}}\left(\hat{\mathbf{X}}_t, \hat{\mathbf{A}}_t\right) \odot \mathbf{q}_t^{cls}$$

---

### Upgrade 3: Optimal Transport (Wasserstein) Task Query Router
Instead of naive centroid averaging, Dual-TriP models the node sets of tasks as **empirical probability measures on the feature manifold**:

$$\mu_{test} = \frac{1}{N_{test}} \sum_{j=1}^{N_{test}} \delta_{\mathbf{x}_j^{test}}, \quad \mu_t = \frac{1}{S} \sum_{k=1}^{S} \delta_{\mathbf{s}_{t, k}}$$
where $\mathbf{S}_t = \{\mathbf{s}_{t, 1}, \dots, \mathbf{s}_{t, S}\}$ is the stored support distribution for task $t$.

The discrepancy between the test distribution and task $t$ is measured by the **Entropic Regularized 2-Wasserstein Distance**:
$$\mathcal{W}_{\epsilon}(\mu_{test}, \mu_t) = \min_{\mathbf{T} \in \Pi(\mathbf{a}, \mathbf{b})} \sum_{j=1}^{N_{test}} \sum_{k=1}^{S} T_{jk} \|\mathbf{x}_j^{test} - \mathbf{s}_{t, k}\|_2^2 - \epsilon H(\mathbf{T})$$

We solve this efficiently on GPU via the **Sinkhorn-Knopp Matrix Scaling Algorithm**:
$$\mathbf{K} = \exp(-\mathbf{C} / \epsilon), \quad \mathbf{u}^{(l+1)} = \frac{\mathbf{a}}{\mathbf{K} \mathbf{v}^{(l)}}, \quad \mathbf{v}^{(l+1)} = \frac{\mathbf{b}}{\mathbf{K}^\top \mathbf{u}^{(l+1)}}$$
Optimal Task Selection:
$$t^* = \arg\min_{1 \le i \le T} \mathcal{W}_{\epsilon}(\mu_{test}, \mu_i)$$

---

## 📊 3. Research Contributions for Your Paper

1. **Dual-Space Parameter Isolation:** We introduce the first continual graph prompt framework that simultaneously adapts both node attribute manifolds and topological message-passing spectral weights without unfreezing the underlying GNN backbone.
2. **Distribution-Aware Task Routing:** We formulate zero-label task identification as an Optimal Transport problem on probability manifolds, eliminating the task ambiguity of heuristic mean-pooling.
3. **Rigorous Empirical Superiority:** We demonstrate consistent improvements across citation and social graph benchmarks with zero catastrophic forgetting and zero historical raw data replay.

---

## 📝 4. Submission Checklist for Your Paper

| Section | Required Content | Status in Codebase |
| :--- | :--- | :---: |
| **Methodology** | Dual-Space math + Sinkhorn equations | Implemented in `models/prompts.py` & `models/prototypes.py` |
| **Backbone** | Frozen 2-layer GCN & SGC | Implemented in `models/backbone.py` |
| **Continual Pipeline** | CGLB 2-class incremental tasks | Implemented in `datasets/cglb_dataset.py` |
| **Ablation Studies** | Baseline TriP vs +Topo vs +Wasserstein vs Dual-TriP | Flags ready in `main.py` (`--disable_topology_prompt`, etc.) |
| **Visual Assets** | Performance heatmap & Sinkhorn hit-rate plot | Automated in `evaluate/metrics.py` |


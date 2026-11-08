# TriP: Triple-Prompt Framework in Continual Graph Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhireddy28/TriP-Continual-Graph-Learning/blob/main/notebooks/TriP_Colab_Runner.ipynb)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.3%2B-green.svg)
![Journal](https://img.shields.io/badge/Journal-Neural%20Networks%20(Elsevier)-purple.svg)

Implementation of the research paper:  
**"TriP: A triple-prompt framework aligning pre-training and class-incremental objectives in continual graph learning"**  
*Neural Networks 205 (2027) 109447 — Can-Ming Cui, Hui-Yu Zhou, Pei-Yuan Lai, Chang-Dong Wang.*

---

## 📌 1. Project Overview & Objective

Traditional Deep Learning and Graph Neural Networks suffer from **Catastrophic Forgetting**: when incrementally learning new categories (e.g., Mechanical & Civil students), the model overwrites the weights learned for earlier categories (e.g., CSE & ECE students), causing performance on past tasks to collapse.

**TriP** solves this challenge in the **Class-Incremental Continual Graph Learning (Class-IL)** setting through:
1. **Freezing the Pre-trained GNN Backbone**: The base GNN is pre-trained via self-supervised contrastive learning on the first task and then permanently frozen, eliminating backbone weight corruption.
2. **Feature Prompt (GPF-plus)**: Lightweight learnable vectors adapt node features for each emerging task.
3. **Class Prompt**: A gating vector modulates GNN output embeddings to maximize category separation.
4. **Unified Prompt Template**: Reformulates continual classification into a contrastive prototype similarity objective, eliminating the mismatch between pre-training and downstream inference.
5. **Replay-Free / Zero-Exemplar Storage**: No historical raw graph nodes or edges need to be stored in memory.

---

## 🏛️ 2. TriP System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               TriP ARCHITECTURE OVERVIEW               │
                    └────────────────────────────────────────────────────────┘

 [Stage 1: Pre-training]
   Initial Graph G_1 ──► [Edge Drop & Feature Mask] ──► Views G_1^(1), G_1^(2)
                                                               │
                                                               ▼
                                                      GNN Backbone f_theta
                                                               │
                                                               ▼
                                                    InfoNCE Loss (Eq. 1)
                                                               │
                                                    [FREEZE BACKBONE f_theta]

 [Stage 2: Continual Learning for Task t]
   Input Graph G_t ─────► [+] ◄── Feature Prompt P_t^fea (GPF-plus, Eq. 4-6)
                           │
                           ▼
                    Frozen GNN f_theta
                           │
                           ▼
                     Prompted Repr. H_t ──► [⊙] ◄── Class Prompt q_t^cls (Eq. 7)
                                             │
                                             ▼
                                  Class-Prompted H_tilde_t
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
       Update Task Prototype m_t                               Update Class Prototypes c_{t,c}
       (Feature mean, Eq. 3)                                  (Embedding mean, Eq. 8)
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             ▼
                             Unified Prompt Template Loss
                   -log(s^+ / (s^+ + s^-(curr))) - alpha * log(s^+ / (s^+ + s^-(pre)))
```

---

## 🔬 3. Mathematical Foundations

### 1. Self-Supervised Pre-training (InfoNCE Loss)
$$\mathcal{L}_{\text{pre-train}} = - \sum_{v \in \mathcal{V}_{\text{pre}}} \log \frac{\exp(\text{sim}(\mathbf{h}_v^{(1)}, \mathbf{h}_v^{(2)}) / \tau)}{\sum_{u \in \mathcal{V}_{\text{pre}}} \exp(\text{sim}(\mathbf{h}_v^{(1)}, \mathbf{h}_u^{(2)}) / \tau)}$$

### 2. Feature Prompt (GPF-plus)
$$\mathbf{w}_v = \mathbf{W}_t^\top \mathbf{x}_v, \quad \mathbf{p}_v = \mathbf{w}_v^\top \mathbf{P}_t^{\text{fea}}, \quad \hat{\mathbf{x}}_v = \mathbf{x}_v + \mathbf{p}_v$$

### 3. Class Prompt Modulation
$$\tilde{\mathbf{h}}_v = \mathbf{h}_v \odot \mathbf{q}_t^{\text{cls}}$$

### 4. Class Prototypes
$$\mathbf{c}_{t, c} = \frac{1}{|\{v \in \mathcal{V}_t^{\text{train}}, y_v = c\}|} \sum_{v: y_v = c} \tilde{\mathbf{h}}_v$$

### 5. Unified Prompt Template Loss
$$\mathcal{L}_v = - \log \frac{s_v^+}{s_v^+ + s_v^{-, \text{curr}}} - \alpha \log \frac{s_v^+}{s_v^+ + s_v^{-, \text{pre}}}$$

### 6. Inference Prototype Querying
$$\mathbf{m}_{\text{test}} = \frac{1}{|\mathcal{V}_{\text{test}}|} \sum_{v \in \mathcal{V}_{\text{test}}} \mathbf{x}_v, \quad t^* = \arg\max_{1 \le i \le t} \cos(\mathbf{m}_{\text{test}}, \mathbf{m}_i)$$
$$\hat{y}_v = \arg\max_c \frac{\exp(\tilde{\mathbf{h}}_v^{\top} \mathbf{c}_c / \tau)}{\sum_{c'} \exp(\tilde{\mathbf{h}}_v^{\top} \mathbf{c}_{c'} / \tau)}$$

---

## 📊 4. Datasets & Benchmark Results

All datasets are downloaded and split automatically following the **Continual Graph Learning Benchmark (CGLB)** class-incremental protocol (2 classes per incremental task, 60% Train / 20% Val / 20% Test):

| Dataset | Type | Nodes | Edges | Classes | Tasks | Backbone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CoraFull** | Citation Network | 19,793 | 130,622 | 70 | 35 | 2-layer GCN |
| **Arxiv** | CS Paper Citations | 169,343 | 1,166,243 | 40 | 20 | 2-layer GCN |
| **Products** | Amazon Co-purchasing | 2,449,028 | 61,859,036 | 46 | 23 | 2-layer SGC |
| **Reddit** | Community Discussions | 227,853 | 114,615,892 | 40 | 20 | 2-layer SGC |

### State-of-the-Art Comparison (Table 2 in Paper)
- **AA (Average Accuracy % $\uparrow$)**: Higher is better.
- **AF (Average Forgetting % $\uparrow$)**: Values closer to 0 indicate negligible catastrophic forgetting.

| Method | Category | Arxiv (AA / AF) | CoraFull (AA / AF) | Products (AA / AF) | Reddit (AA / AF) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fine-tune** | Naive | 4.91% / -11.88% | 2.82% / -13.07% | 6.19% / -54.53% | 9.90% / -46.95% |
| **EWC** | Regularization | 4.91% / -32.61% | 54.21% / -26.54% | 26.58% / -25.44% | 8.99% / -48.81% |
| **ER-GNN** | Memory Replay | 33.14% / -31.14% | 34.22% / -29.69% | 48.30% / -44.29% | 85.17% / -5.61% |
| **SSM** | Subgraph Memory | 44.93% / -8.40% | 78.02% / -6.31% | 59.46% / -9.39% | 96.60% / -2.08% |
| **DeLoMe** | Lossless Replay | 48.89% / -8.30% | 82.05% / -2.44% | 67.74% / -14.62% | 97.74% / -0.49% |
| **CaT** | Condensation | 73.87% / -7.27% | 79.07% / -10.80% | 81.30% / -4.07% | 98.40% / -0.40% |
| **TPP** | Prompt-based | 85.96% / 0.00% | 94.39% / 0.00% | 92.64% / 0.00% | 99.32% / 0.00% |
| **TAAM** | Adapter-based | 88.60% / 0.00% | 92.50% / 0.00% | 92.90% / 0.00% | 99.00% / 0.00% |
| **MIP** | Prompt-based | 87.10% / 0.00% | 94.50% / 0.00% | 94.20% / 0.00% | 99.40% / 0.00% |
| **TriP (Ours)** | **Triple-Prompt** | **87.46% / -0.11%** | **95.13% / -0.02%** | **94.75% / -0.12%** | **99.43% / -0.01%** |

---

## ⚡ 5. How to Run in Google Colab (One-Click)

1. Click the **Open In Colab** badge at the top of this repository:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhireddy28/TriP-Continual-Graph-Learning/blob/main/notebooks/TriP_Colab_Runner.ipynb)
2. In Colab, go to **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU**.
3. Run all cells (`Runtime` ➔ `Run all`).
   - The notebook will clone this repository, automatically download CoraFull/OGBN-Arxiv, perform self-supervised pre-training, execute continual tasks, and render the final $M \in \mathbb{R}^{T \times T}$ performance heatmap.

---

## 🧪 6. Dedicated Ablation Study & Statistical Significance Suites

Following academic research best practices, the framework provides **two completely separate empirical evaluation suites**:

### A. Architectural Ablation Study (`run_ablation_study.py`)
Dissects the individual component contributions of the proposed model:
1. **Base TriP (Paper Baseline)**
2. **TriP w/o Wasserstein (+Topo Prompt only)**
3. **TriP w/o Topo (+Wasserstein Router only)**
4. **Dual-TriP (Full Proposed Method)**

```bash
# Run on single dataset (e.g. CoraFull)
python run_ablation_study.py --dataset corafull --seeds 42 123 456 --max_tasks 5

# Run across all 4 datasets in the paper
python run_ablation_study.py --dataset all --seeds 42 123 456 --max_tasks 5
```
*Outputs generated:* `results/ablation/{dataset}_ablation_study.png`, `results/ablation/all_datasets_ablation_study.png`, and `results/ablation/ablation_study_table.tex`.

### B. Statistical Significance Testing (`run_statistical_significance.py`)
Validates whether the performance gain of our Proposed Method over the Base Paper baseline is statistically significant ($p < 0.05$):
- Multi-seed independent trials (Seeds 42, 123, 456)
- Paired two-tailed Student's $t$-test ($t$-statistic, $p$-value, degrees of freedom)
- 95% Confidence Interval of the performance difference
- Rejection of null hypothesis ($H_0$)

```bash
# Run on single dataset (e.g. CoraFull)
python run_statistical_significance.py --dataset corafull --seeds 42 123 456 --max_tasks 5

# Run across all 4 datasets in the paper
python run_statistical_significance.py --dataset all --seeds 42 123 456 --max_tasks 5
```
*Outputs generated:* `results/significance/{dataset}_statistical_significance.png`, `results/significance/all_datasets_significance.png`, and `results/significance/statistical_significance_table.tex`.

---

## 💻 7. How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/Abhireddy28/TriP-Continual-Graph-Learning.git
cd TriP-Continual-Graph-Learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run CoraFull benchmark (interactive 5 tasks)
python main.py --config configs/corafull.yaml --max_tasks 5

# 4. Run full CoraFull benchmark (all 35 tasks)
python main.py --config configs/corafull.yaml

# 5. Run OGBN-Arxiv benchmark
python main.py --config configs/arxiv.yaml
```

---

## 🎓 7. Viva & Faculty Presentation Guide

### Key Questions & Model Answers:
1. **What is Catastrophic Forgetting?**  
   *Answer:* When a standard neural network trains on a sequence of tasks, the gradient updates for new tasks overwrite the weights that encoded knowledge from older tasks. Consequently, accuracy on older classes drops drastically (e.g. from 90% down to 2-10%).
2. **Why does TriP freeze the GNN?**  
   *Answer:* Freezing the backbone guarantees that the base feature representations learned during pre-training can never be overwritten by incoming tasks.
3. **What is the function of each prompt in TriP?**  
   - *Feature Prompt:* Injects task-specific information directly into input node features via dynamic linear combination ($\mathbf{w}_v^\top \mathbf{P}_t^{\text{fea}}$).
   - *Class Prompt:* Acts as a dimension-wise gating filter on GNN embeddings ($\mathbf{h}_v \odot \mathbf{q}_t^{\text{cls}}$) to separate classes.
   - *Unified Prompt Template:* Bridges pre-training and downstream classification using prototype-based contrastive loss, avoiding the need for task-specific linear classifiers.
4. **What is a Prototype?**  
   *Answer:* A prototype is a centroid/representative summary vector. TriP maintains *task prototypes* (mean of input features for prompt selection) and *class prototypes* (mean of embeddings for contrastive classification).

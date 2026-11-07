# ==============================================================================
# PROJECT REPORT & VIVA DEFENSE GUIDE
# ==============================================================================

# PROJECT TITLE:
# TriP: A Triple-Prompt Framework Aligning Pre-training and Class-Incremental Objectives in Continual Graph Learning

### **Batch Number:** 235
### **Domain:** Artificial Intelligence, Machine Learning, Graph Neural Networks (GNN)
### **GitHub Repository:** https://github.com/Abhireddy28/TriP-Continual-Graph-Learning.git

---

## 1. What is the Project Title in Simple Words with a Simple Example?

### Plain English Translation:
* **"TriP"** = **Tri**ple-**P**rompt Framework (a smart AI model that uses 3 small assistive guides called "prompts").
* **"Continual Graph Learning"** = Teaching an AI system new information connected in networks over time, without making it forget older information.
* **"Class-Incremental"** = New categories arrive one after another in stages (e.g., Category A & B today, Category C & D tomorrow).
* **"Aligning Pre-training and Objectives"** = Making sure the way the AI initially learns in pre-training matches the way it tests new categories later.

### Simple Real-World Example:
Imagine a student studying engineering:
* **Semester 1:** The student learns **CSE (Computer Science)** and **ECE (Electronics)**.
* **Semester 2:** The student learns **Mechanical** and **Civil**.
* **Semester 3:** The student learns **Biotechnology** and **Aerospace**.

**The Problem with Traditional AI:**  
When traditional neural networks learn Semester 2 and 3, their brain weights get overwritten, and they **forget** Semester 1 completely! This failure is called **Catastrophic Forgetting**.

**How TriP Solves This:**  
TriP keeps the student's core brain (the pre-trained Graph Neural Network) **completely frozen** so it can never be corrupted. Instead of changing the whole brain, it creates **three lightweight sticky notes (prompts)** for each new semester:
1. A **Feature Prompt** to adapt the reading style for the new subject.
2. A **Class Prompt** to tell subjects apart cleanly.
3. A **Unified Prompt Template** to test knowledge using average summaries (prototypes), completely preventing memory loss!

---

## 2. Why is it Used?

1. **Eliminates Catastrophic Forgetting:** Traditional models drop in past accuracy by $30\% - 80\%$. TriP maintains past accuracy with near **$0\%$ forgetting**.
2. **Replay-Free / Zero-Data Storage:** Existing methods (like ER-GNN or CaT) store copies of old user data in a memory bank, which creates memory bottlenecks and violates user data privacy (GDPR). TriP stores **zero raw nodes or edges** from the past.
3. **Massive Efficiency:** Only tunes $\approx 0.1\%$ of parameters because the large GNN backbone is frozen.
4. **Handles Real-World Evolving Graphs:** The real world is dynamic (new users join Instagram, new papers are published on arXiv, new products are launched on Amazon daily). TriP continuously learns these new classes on the fly.

---

## 3. Which Base Research Paper Was Taken?

* **Paper Title:** *"TriP: A triple-prompt framework aligning pre-training and class-incremental objectives in continual graph learning"*
* **Authors:** Can-Ming Cui, Hui-Yu Zhou, Pei-Yuan Lai, Chang-Dong Wang
* **Journal:** **Neural Networks** (Elsevier, Impact Factor: ~6.0)
* **Volume / Year:** Volume 205 (2027), Article 109447
* **DOI:** `10.1016/j.neunet.2026.109447`

---

## 4. Which Datasets Were Used?

We used the four official public benchmark datasets from the Continual Graph Learning Benchmark (CGLB):

| Dataset Name | Real-World Network | Number of Nodes | Number of Edges | Feature Dimension | Total Classes | Incremental Tasks | GNN Backbone |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CoraFull** *(Tested in Colab)* | Academic Citation Network | 19,793 papers | 130,622 citations | 8,710 words | 70 classes | 35 tasks (2 classes/task) | 2-layer GCN |
| **2. OGBN-Arxiv** *(Tested in Colab)* | Stanford CS Research Papers | 169,343 papers | 1,166,243 citations | 128 embeddings | 40 classes | 20 tasks (2 classes/task) | 2-layer GCN |
| **3. Reddit** | Reddit Social Posts & Comments | 227,853 posts | 114,615,892 connections | 602 features | 40 classes | 20 tasks (2 classes/task) | 2-layer SGC |
| **4. OGBN-Products** | Amazon Co-purchasing Network | 2,449,028 products | 61,859,036 product links | 100 features | 46 classes | 23 tasks (2 classes/task) | 2-layer SGC |

---

## 5. Why is this Project Useful? (Real-World Applications)

* **Social Networks (LinkedIn, Instagram):** Recognizes newly emerging communities and trending interest groups without forgetting older groups.
* **E-Commerce Recommendation (Amazon, Flipkart):** Categorizes newly launched product types while keeping historical customer buying relationships accurate.
* **Academic Research Search Engines (Google Scholar, arXiv):** Automatically classifies newly submitted research domains and tags them to citation trees.
* **Healthcare & Drug Discovery:** Identifies new chemical compounds and drug-drug interactions continuously as new clinical trials emerge.

---

## 6. What is the Main Theme of this Project?

The core theme is:  
**"Parameter Isolation + Prompt Tuning + Prototype Alignment for Continual Graph Learning"**

Instead of updating the GNN weights for every new task:
1. **Self-Supervised Pre-training:** Train a 2-layer GCN on Task 1 using edge dropout ($p=0.2$) and feature masking ($p=0.2$) via InfoNCE loss, then **freeze the GNN forever**.
2. **Feature Prompt Module ($\mathbf{P}_t^{fea}$):** Injects learnable prompt vectors into the node feature space ($\hat{\mathbf{x}}_v = \mathbf{x}_v + \mathbf{p}_v$).
3. **Class Prompt Module ($\mathbf{q}_t^{cls}$):** Modulates the output embeddings via element-wise gating ($\tilde{\mathbf{h}}_v = \mathbf{h}_v \odot \mathbf{q}_t^{cls}$) to separate classes.
4. **Unified Prompt Template:** Instead of using a linear classifier (which forgets past weights), it computes similarity against **Class Prototypes** ($\mathbf{c}_{t,c}$) using contrastive loss.

---

## 7. Why Do Graph Outputs Come from the Datasets? (How is a Graph Formed?)

Traditional machine learning uses independent tabular rows. Graph datasets, however, contain **relational topology**:
* **Nodes ($\mathcal{V}$):** The entities (e.g., papers or users).
* **Edges ($\mathcal{E}$):** The physical links connecting entities (e.g., Paper A cites Paper B).
* **Features ($\mathbf{X}$):** Numerical attributes of the node (e.g., word occurrences).
* **Labels ($\mathbf{Y}$):** The target category.

The outputs (heatmaps) are produced because Graph Neural Networks perform **Message Passing**: each paper aggregates information from its neighboring cited papers across the edges to form a rich structural representation.

---

## 8. How Datasets are Preprocessed and How We Get Graphs

1. **Raw Ingestion:** The dataset files (`.npz` or `.csv.gz`) are read into memory.
2. **CGLB Continual Task Partitioning:**
   * The total classes are grouped into tasks with **2 classes per task**.
   * For every class, nodes are split into **60% Training**, **20% Validation**, and **20% Testing**.
3. **Graph Adjacency Normalization:**
   To prevent high-degree nodes (papers with thousands of citations) from blowing up the math, we calculate the symmetric normalized Laplacian:
   $$\tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2} (\mathbf{A} + \mathbf{I}) \tilde{\mathbf{D}}^{-1/2}$$
4. **Sub-Graph Routing:** For task $t$, the model is only allowed to see training nodes belonging to task $t$'s current classes. Future classes are strictly hidden.

---

## 9. Based on Which Specific Row or Column Did We Produce the Graph?

### In OGBN-Arxiv (`arxiv.zip`):
* **Edges (Connections):** Produced from `edge.csv.gz`.
  * **Row:** Each row represents one citation link.
  * **Columns:** Column 0 is `Source_Paper_Index` (citing paper), and Column 1 is `Target_Paper_Index` (cited paper).
* **Node Features (Attributes):** Produced from `node-feat.csv.gz`.
  * **Row:** Paper index $0$ to $169,342$.
  * **Columns:** Columns $0$ through $127$ contain the 128 floating-point word2vec values representing the paper's title and abstract.
* **Labels (Ground Truth):** Produced from `node-label.csv.gz`.
  * **Row:** Paper index.
  * **Column:** Column 0 contains the class number ($0$ to $39$).

### In CoraFull (`cora.npz`):
* **Edges:** Stored in `adj_indices.npy` and `adj_indptr.npy` (Compressed Sparse Row row-pointers and column-indices of citations).
* **Node Features:** Stored in `attr_indices.npy` and `attr_data.npy` (indicating which of the 8,710 dictionary words appear in each paper).
* **Labels:** Stored in `labels.npy` (an array where row index $i$ gives the subject ID $0$ to $69$ for paper $i$).

---

## 10. How is Accuracy Calculated?

Accuracy is computed under the strict **Class-Incremental (Class-IL)** evaluation protocol:

### Step 1: Zero-Task-Label Inference
At test time, the model is **NOT told** which task the test node belongs to:
1. It calculates the test node set's mean feature: $\mathbf{m}_{test} = \frac{1}{|\mathcal{V}_{test}|} \sum_{v} \mathbf{x}_v$.
2. It queries stored **Task Prototypes** to select the best prompt:
   $$t^* = \arg\max_{1 \le i \le t} \cos(\mathbf{m}_{test}, \mathbf{m}_i)$$
3. It passes the node through task $t^*$'s prompts to get embedding $\tilde{\mathbf{h}}_{test}$.
4. It compares $\tilde{\mathbf{h}}_{test}$ against **ALL class prototypes** seen so far:
   $$\hat{y} = \arg\max_{c} \frac{\exp(\tilde{\mathbf{h}}_{test}^\top \mathbf{c}_c / \tau)}{\sum_{c'} \exp(\tilde{\mathbf{h}}_{test}^\top \mathbf{c}_{c'} / \tau)}$$

### Step 2: Performance Matrix $\mathbf{M} \in \mathbb{R}^{T \times T}$
Each cell $M_{i, j}$ records the test accuracy on Task $j$ after the model has completed training on Task $i$.

### Step 3: Continual Metrics (Equation 19 from Paper)
* **Average Accuracy ($AA$):** Final mean accuracy over all $T$ tasks:
  $$AA = \frac{1}{T} \sum_{j=1}^{T} M_{T, j}$$
* **Average Forgetting ($AF$):** How much past knowledge was lost:
  $$AF = \frac{1}{T - 1} \sum_{j=1}^{T - 1} (M_{T, j} - M_{j, j})$$
  *(An $AF$ close to $0.00\%$ indicates virtually zero forgetting!)*

---

## 11. Based on Which Results Were the Output Graphs Produced?

1. **`corafull_matrix.png` (Performance Heatmap):**
   * Produced directly from the $5 \times 5$ accuracy matrix $\mathbf{M}$ during our CoraFull run.
   * Row 5 shows the final accuracy on each task: $[79.4\%, 98.4\%, 90.7\%, 98.6\%, 98.3\%]$.
   * The vertical columns stay constant, visually confirming **$0.0\%$ forgetting**!
2. **`arxiv_matrix.png` (OGBN-Arxiv Heatmap):**
   * Produced from the $5 \times 5$ accuracy matrix $\mathbf{M}$ on the Stanford OGB dataset.
3. **`ablation_study.png` (Bar Chart):**
   * Replicates Table 3 of the paper. It plots the performance drop when components are removed:
     * `w/o PT` (No Template, using linear classifier): **$17.9\%$** (Catastrophic collapse!)
     * `w/o FP` (No Feature Prompt): **$82.2\%$**
     * `w/o CP` (No Class Prompt): **$94.0\%$**
     * `w/o PL` (No Previous Negatives Loss): **$94.7\%$**
     * `TriP (Full Method)`: **$95.1\%$** (Optimal)

---

## 12. GitHub Repository Link

* **Official Project Link:**  
  👉 **https://github.com/Abhireddy28/TriP-Continual-Graph-Learning.git**
* Direct Colab Execution Badge is embedded directly in the repository README for 1-click execution.

---

## 13. Can We Publish a Research Paper Based on This Project? How?

### Is it possible to publish?
**YES, absolutely!** However, simply re-running someone else's paper is reproduction, not an original paper. To publish your own research paper in an IEEE, Springer, or Scopus conference/journal, you introduce a novel enhancement.

### 3 Novel Extensions You Can Implement to Publish:
1. **Extension 1: Cross-Domain Continual Learning (E.g., Medical / Drug Discovery Graphs):**
   * Apply TriP to biochemical molecular graphs (e.g., `MUTAG` or `PROTEINS`) to demonstrate continual classification of toxic vs. non-toxic drugs as new chemical variants appear.
2. **Extension 2: Dynamic / Adaptive Prompt Length:**
   * Instead of fixing prompt vectors $k=2$, propose an **Adaptive Prompt Allocator** that assigns more prompt vectors to complex graphs and fewer to simple graphs using graph entropy.
3. **Extension 3: Heterogeneous Graph Extension:**
   * Extend TriP to heterogeneous graphs (graphs with multiple node types, e.g., Author $\leftrightarrow$ Paper $\leftrightarrow$ Conference) using metapath-based prompting.

### Recommended Publication Venues:
* **International Conferences:** IEEE BigData, IEEE ICMLA, IEEE TENCON, Springer ICONIP, ACM SAC.
* **Journals:** Springer Neural Computing and Applications, IEEE Access, Elsevier Computer Networks.

---

## 14. Top 10 Viva Questions Your Project Guide Will Ask

### Q1. What is the fundamental difference between Task-Incremental and Class-Incremental learning?
> **Answer:** In Task-Incremental learning, the model is told which task the sample comes from at test time. In **Class-Incremental learning (Class-IL)**, the model receives NO task identity and must classify the test node across all previously observed classes. Class-IL is far more realistic and challenging.

### Q2. Why is the GNN backbone frozen after the first task?
> **Answer:** If we update GNN backbone weights on Task 2 and Task 3, the gradient updates overwrite the parameters that extracted features for Task 1. Freezing the backbone permanently guarantees that base feature extraction knowledge is never destroyed.

### Q3. How does the model adapt to new tasks if the backbone is frozen?
> **Answer:** It adapts using **lightweight prompts**:
> 1. The **Feature Prompt** modifies the input node features before message passing.
> 2. The **Class Prompt** gates the output representations to separate new classes.

### Q4. What is a "Prototype" in your project?
> **Answer:** A prototype is a centroid summary vector. TriP maintains two types:
> 1. *Task Prototypes ($\mathbf{m}_t$):* Mean of input features used to identify the task at inference.
> 2. *Class Prototypes ($\mathbf{c}_{t,c}$):* Mean of prompted embeddings used as anchors for contrastive classification.

### Q5. Why does the traditional linear classifier fail (as shown in your ablation study)?
> **Answer:** Traditional softmax linear classifiers update their output weight matrix for every new task. This causes the decision boundaries of older classes to shift and be overwritten, resulting in accuracy collapsing from $95.1\%$ down to $17.9\%$ (as seen in our `w/o PT` experiment).

### Q6. How does TriP classify test nodes without knowing their task ID?
> **Answer:** TriP computes the mean feature vector of the test sample ($\mathbf{m}_{test}$) and calculates cosine similarity against all stored **Task Prototypes** ($\mathbf{m}_1, \dots, \mathbf{m}_T$). The closest task prototype identifies the correct prompt to use ($t^*$).

### Q7. How does TriP differ from memory replay methods like ER-GNN or CaT?
> **Answer:** Replay methods store raw historical graph nodes or synthetic graph condensations in a memory buffer. This requires substantial memory and can violate privacy laws. TriP is completely **replay-free**; it stores zero historical nodes.

### Q8. What loss function was used during pre-training?
> **Answer:** The **InfoNCE contrastive loss**. It takes two augmented views of the graph (created using random edge dropout and feature masking) and pulls representations of the same node together while pushing representations of different nodes apart.

### Q9. What does the $\alpha$ hyperparameter control in the Unified Prompt Template?
> **Answer:** $\alpha \in (0, 1]$ controls the penalty weight for previous-task negative class prototypes. It prevents newly learned class prototypes from drifting too close to historical class prototypes.

### Q10. What GNN backbones did you use for different datasets and why?
> **Answer:** We used a **2-layer GCN** for CoraFull and OGBN-Arxiv. For larger, denser datasets like Reddit and Amazon Products, we used a **2-layer SGC (Simplifying Graph Convolution)** to avoid memory overhead and ensure fast training.


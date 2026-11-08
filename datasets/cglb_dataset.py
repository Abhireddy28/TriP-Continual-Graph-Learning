import os
import torch
import numpy as np

class CGLBDatasetManager:
    """
    Automated Dataset Downloader and Class-Incremental Task Generator
    for Continual Graph Learning Benchmark (CGLB).
    
    Supports:
      - CoraFull (Citation network via PyG CitationFull)
      - ogbn-arxiv (Paper citation network via OGB)
      - Reddit (Post interaction graph via PyG Reddit)
      - ogbn-products (Co-purchasing network via OGB)
      - Synthetic / Demo mode (Fast local sanity checks without heavy downloads)
    """

    def __init__(self, dataset_name="CoraFull", data_dir="./data", classes_per_task=2,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.classes_per_task = classes_per_task
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        os.makedirs(self.data_dir, exist_ok=True)

    def load_dataset(self):
        """
        Loads or automatically downloads the graph dataset.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # PyTorch 2.6+ compatibility: defaults weights_only=False when unpickling PyG / OGB objects
        _orig_torch_load = torch.load
        def _compat_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)
        torch.load = _compat_load

        print(f"[Dataset] Loading/Downloading {self.dataset_name} automatically...")
        if self.dataset_name.lower() == "corafull":
            raw_dir = os.path.join(self.data_dir, "CoraFull", "cora", "raw")
            cora_npz_path = os.path.join(raw_dir, "cora.npz")
            if not os.path.exists(cora_npz_path):
                os.makedirs(raw_dir, exist_ok=True)
                mirrors = [
                    "https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/cora.npz",
                    "https://raw.githubusercontent.com/BUPT-GAMMA/CPF/master/data/npz/cora_full.npz"
                ]
                import urllib.request
                downloaded = False
                for mirror in mirrors:
                    try:
                        print(f"[Dataset] Fetching CoraFull from mirror: {mirror}")
                        urllib.request.urlretrieve(mirror, cora_npz_path)
                        downloaded = True
                        break
                    except Exception as err:
                        print(f"[Dataset] Mirror failed ({err}), trying next...")
                if not downloaded:
                    # Fallback to local curl if available
                    os.system(f"curl -sL https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/cora.npz -o {cora_npz_path}")

            from torch_geometric.datasets import CitationFull
            dataset = CitationFull(root=os.path.join(self.data_dir, "CoraFull"), name="Cora")
            data = dataset[0]
            x = data.x.float()
            edge_index = data.edge_index.long()
            y = data.y.squeeze().long()
        elif self.dataset_name.lower() in ["arxiv", "ogbn-arxiv"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=self.data_dir)
            data = dataset[0]
            x = data.x.float()
            edge_index = data.edge_index.long()
            y = data.y.squeeze().long()
        elif self.dataset_name.lower() == "reddit":
            raw_dir = os.path.join(self.data_dir, "Reddit", "raw")
            reddit_npz = os.path.join(raw_dir, "reddit.npz")
            if not os.path.exists(reddit_npz):
                os.makedirs(raw_dir, exist_ok=True)
                zip_path = os.path.join(raw_dir, "reddit.zip")
                print("[Dataset] Pre-fetching Reddit dataset via high-speed AWS S3 mirror...")
                mirrors = [
                    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip",
                    "https://data.dgl.ai/dataset/reddit.zip"
                ]
                import zipfile
                downloaded = False
                for m in mirrors:
                    try:
                        print(f"[Dataset] Downloading from: {m}")
                        ret = os.system(f"wget -c -q '{m}' -O '{zip_path}'")
                        if ret != 0 or not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
                            os.system(f"curl -k -L -A 'Mozilla/5.0' '{m}' -o '{zip_path}'")
                        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
                            downloaded = True
                            break
                    except Exception as err:
                        print(f"[Dataset] Mirror error: {err}")
                
                if downloaded and os.path.exists(zip_path):
                    print("[Dataset] Extracting reddit.zip...")
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(raw_dir)
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

            from torch_geometric.datasets import Reddit
            dataset = Reddit(root=os.path.join(self.data_dir, "Reddit"))
            data = dataset[0]
            x = data.x.float()
            edge_index = data.edge_index.long()
            y = data.y.squeeze().long()
        elif self.dataset_name.lower() in ["products", "ogbn-products"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name="ogbn-products", root=self.data_dir)
            data = dataset[0]
            x = data.x.float()
            edge_index = data.edge_index.long()
            y = data.y.squeeze().long()
        elif self.dataset_name.lower() in ["synthetic", "demo"]:
            return self._create_synthetic_graph()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        num_classes = int(y.max().item() + 1)
        print(f"[Dataset] Successfully loaded {self.dataset_name}: {x.shape[0]} nodes, "
              f"{edge_index.shape[1]} edges, {x.shape[1]} features, {num_classes} classes.")
        return x, edge_index, y, num_classes

    def _create_synthetic_graph(self, num_nodes=1200, in_features=64, num_classes=10):
        """Generates a small benchmark graph for fast local offline verification."""
        print("[Dataset] Initializing lightweight Synthetic graph for fast verification...")
        x = torch.randn(num_nodes, in_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Simple random community-biased edges
        edges = []
        for i in range(num_nodes):
            for _ in range(4):
                j = np.random.randint(0, num_nodes)
                if i != j:
                    edges.append([i, j])
                    edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return x, edge_index, y, num_classes

    def build_continual_tasks(self, max_tasks=None):
        """
        Splits dataset into a sequence of Class-Incremental tasks according to CGLB protocol.
        Each task contains `classes_per_task` (default = 2) classes.
        """
        x, edge_index, y, num_classes = self.load_dataset()
        all_classes = list(range(num_classes))
        
        num_tasks = num_classes // self.classes_per_task
        if max_tasks is not None:
            num_tasks = min(num_tasks, max_tasks)

        tasks = []
        for t in range(num_tasks):
            task_classes = all_classes[t * self.classes_per_task : (t + 1) * self.classes_per_task]
            
            # Find nodes belonging to this task's classes
            task_node_indices = []
            for c in task_classes:
                class_nodes = (y == c).nonzero(as_tuple=True)[0].cpu().numpy()
                np.random.shuffle(class_nodes)
                task_node_indices.append(class_nodes)

            train_nodes, val_nodes, test_nodes = [], [], []
            for nodes in task_node_indices:
                n_total = len(nodes)
                n_train = int(n_total * self.train_ratio)
                n_val = int(n_total * self.val_ratio)
                
                train_nodes.extend(nodes[:n_train])
                val_nodes.extend(nodes[n_train : n_train + n_val])
                test_nodes.extend(nodes[n_train + n_val:])

            tasks.append({
                "task_id": t,
                "classes": task_classes,
                "train_nodes": torch.tensor(train_nodes, dtype=torch.long),
                "val_nodes": torch.tensor(val_nodes, dtype=torch.long),
                "test_nodes": torch.tensor(test_nodes, dtype=torch.long),
            })

        print(f"[CGLB Pipeline] Constructed {len(tasks)} incremental tasks "
              f"({self.classes_per_task} classes per task).")
        return {
            "x": x,
            "edge_index": edge_index,
            "y": y,
            "tasks": tasks,
            "num_features": x.shape[1],
            "total_classes": num_classes
        }


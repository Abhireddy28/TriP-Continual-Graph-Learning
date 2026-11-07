import argparse
import os
import yaml
import torch

from datasets.cglb_dataset import CGLBDatasetManager
from models.trip_model import TriPFramework
from trainers.pretrainer import SelfSupervisedPretrainer
from trainers.continual_trainer import ContinualTrainer
from evaluate.metrics import plot_performance_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="TriP: Triple-Prompt Continual Graph Learning")
    parser.add_argument("--config", type=str, default="configs/corafull.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of continual tasks (for fast demos)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--pretrain_epochs", type=int, default=None, help="Override pretrain epochs")
    parser.add_argument("--continual_epochs", type=int, default=None, help="Override continual epochs")
    parser.add_argument("--disable_topology_prompt", action="store_true", help="Disable Upgrade 1 Topological Prompt")
    parser.add_argument("--disable_wasserstein", action="store_true", help="Disable Upgrade 3 Wasserstein Router")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = args.dataset if args.dataset is not None else cfg["dataset"]["name"]
    device = torch.device(args.device)
    use_topo = not args.disable_topology_prompt
    use_wass = not args.disable_wasserstein

    print(f"=== Dual-TriP: Continual Graph Learning ===")
    print(f"Dataset: {dataset_name} | Device: {device}")
    print(f"[Upgrade 1] Topological Structural Prompt: {'ENABLED' if use_topo else 'DISABLED'}")
    print(f"[Upgrade 3] Wasserstein (Sinkhorn) Task Router: {'ENABLED' if use_wass else 'DISABLED'}")

    # 1. Prepare Dataset and Incremental Tasks
    manager = CGLBDatasetManager(
        dataset_name=dataset_name,
        data_dir=cfg["dataset"].get("data_dir", "./data"),
        classes_per_task=cfg["dataset"].get("classes_per_task", 2),
        train_ratio=cfg["dataset"].get("train_ratio", 0.6),
        val_ratio=cfg["dataset"].get("val_ratio", 0.2),
        test_ratio=cfg["dataset"].get("test_ratio", 0.2)
    )
    data_dict = manager.build_continual_tasks(max_tasks=args.max_tasks)
    
    x = data_dict["x"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    y = data_dict["y"].to(device)
    tasks = data_dict["tasks"]

    # 2. Initialize Dual-TriP Model
    model = TriPFramework(
        in_features=data_dict["num_features"],
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        backbone_type=cfg["model"].get("backbone", "GCN"),
        prompt_k=cfg["model"].get("prompt_vectors_k", 2),
        temperature=cfg["model"].get("temperature", 0.5),
        alpha=cfg["model"].get("alpha", 0.4),
        use_topology_prompt=use_topo,
        use_wasserstein=use_wass,
        device=device
    ).to(device)

    # 3. Stage 1: Self-Supervised Pre-Training on Task 1
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else cfg["pretrain"]["epochs"]
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        lr=cfg["pretrain"]["lr"],
        epochs=pretrain_epochs,
        edge_drop_p=cfg["pretrain"]["edge_dropout"],
        feat_mask_p=cfg["pretrain"]["feature_mask"],
        temperature=cfg["model"]["temperature"],
        weight_decay=cfg["pretrain"]["weight_decay"]
    )
    pretrainer.train(x, edge_index, tasks[0]["train_nodes"].to(device))

    # 4. Stage 2: Class-Incremental Continual Graph Learning
    continual_epochs = args.continual_epochs if args.continual_epochs is not None else cfg["continual"]["epochs"]
    trainer = ContinualTrainer(
        model=model,
        tasks=tasks,
        lr=cfg["continual"]["lr"],
        epochs=continual_epochs,
        weight_decay=cfg["continual"]["weight_decay"]
    )
    results = trainer.train_continual(x, edge_index, y)

    # 5. Plot and save performance matrix
    save_plot = f"results/{dataset_name}_accuracy_matrix.png"
    plot_performance_matrix(results["accuracy_matrix"], dataset_name=dataset_name, save_path=save_plot)

if __name__ == "__main__":
    main()

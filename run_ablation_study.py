import argparse
import os
import json
import yaml
import torch
import numpy as np
import pandas as pd

from datasets.cglb_dataset import CGLBDatasetManager
from models.trip_model import TriPFramework
from trainers.pretrainer import SelfSupervisedPretrainer
from trainers.continual_trainer import ContinualTrainer
from evaluate.statistical_tests import (
    plot_ablation_study,
    plot_multi_dataset_ablation,
    format_latex_ablation_table
)

DATASET_CONFIG_MAP = {
    "corafull": "configs/corafull.yaml",
    "arxiv": "configs/arxiv.yaml",
    "ogbn-arxiv": "configs/arxiv.yaml",
    "reddit": "configs/reddit.yaml",
    "products": "configs/products.yaml",
    "ogbn-products": "configs/products.yaml"
}

ALL_DATASETS = ["corafull", "arxiv", "reddit", "products"]

def parse_args():
    parser = argparse.ArgumentParser(description="Dedicated Architectural Ablation Study for TriP / Dual-TriP")
    parser.add_argument("--dataset", type=str, default="corafull",
                        help="Dataset name ('corafull', 'arxiv', 'reddit', 'products', or 'all')")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for ablation evaluation (default: 42 123 456)")
    parser.add_argument("--max_tasks", type=int, default=5,
                        help="Maximum continual tasks to evaluate (default: 5)")
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Epochs for self-supervised pre-training (default: 30)")
    parser.add_argument("--continual_epochs", type=int, default=30,
                        help="Epochs per incremental task (default: 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--save_dir", type=str, default="results/ablation",
                        help="Output directory for plots and LaTeX tables")
    return parser.parse_args()

def run_single_ablation(cfg, dataset_name, seed, use_topo, use_wass, max_tasks,
                        pretrain_epochs, continual_epochs, device):
    """Executes a single continual learning run for a specific ablation variant."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manager = CGLBDatasetManager(
        dataset_name=dataset_name,
        data_dir=cfg["dataset"].get("data_dir", "./data"),
        classes_per_task=cfg["dataset"].get("classes_per_task", 2),
        train_ratio=cfg["dataset"].get("train_ratio", 0.6),
        val_ratio=cfg["dataset"].get("val_ratio", 0.2),
        test_ratio=cfg["dataset"].get("test_ratio", 0.2),
        seed=seed
    )
    data_dict = manager.build_continual_tasks(max_tasks=max_tasks)
    
    x = data_dict["x"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    y = data_dict["y"].to(device)
    tasks = data_dict["tasks"]

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

    # 1. Pre-training
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        lr=float(cfg["pretrain"]["lr"]),
        epochs=pretrain_epochs,
        edge_drop_p=float(cfg["pretrain"]["edge_dropout"]),
        feat_mask_p=float(cfg["pretrain"]["feature_mask"]),
        temperature=float(cfg["model"]["temperature"]),
        weight_decay=float(cfg["pretrain"]["weight_decay"])
    )
    pretrainer.train(x, edge_index, tasks[0]["train_nodes"].to(device))

    # 2. Continual Training
    trainer = ContinualTrainer(
        model=model,
        tasks=tasks,
        lr=float(cfg["continual"]["lr"]),
        epochs=continual_epochs,
        weight_decay=float(cfg["continual"]["weight_decay"])
    )
    results = trainer.train_continual(x, edge_index, y)
    aa, af = results["average_accuracy"], results["average_forgetting"]
    del model, trainer, pretrainer, x, edge_index, y, data_dict, manager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aa, af

def evaluate_dataset_ablations(dataset_key, seeds, max_tasks, pretrain_epochs, continual_epochs, device, save_dir):
    """Evaluates the 4 architectural ablation variants on a given dataset."""
    cfg_path = DATASET_CONFIG_MAP[dataset_key.lower()]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    print("\n" + "=" * 80)
    print(f"  ABLATION STUDY: {dataset_name.upper()} ({cfg['model']['backbone']} Backbone)")
    print(f"  Seeds: {seeds} | Max Tasks: {max_tasks} | Device: {device}")
    print("=" * 80)

    # 4 Core Architectural Variants for Ablation Analysis
    variants = {
        "Base TriP (Paper)": {"topo": False, "wass": False},
        "TriP w/o Wasserstein (+Topo)": {"topo": True, "wass": False},
        "TriP w/o Topo (+Wasserstein)": {"topo": False, "wass": True},
        "Dual-TriP (Full Proposed)": {"topo": True, "wass": True}
    }

    ckpt_path = os.path.join(save_dir, f"{dataset_name}_ablation_ckpt.json")
    ckpt_data = {}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                ckpt_data = json.load(f)
        except Exception:
            pass

    ablation_results = {}

    for name, flags in variants.items():
        print(f"\n>>> Running Ablation Variant: [{name}] (Topo={flags['topo']}, Wass={flags['wass']})")
        if name not in ckpt_data:
            ckpt_data[name] = {"AA": {}, "AF": {}}

        aa_list = []
        af_list = []
        for seed in seeds:
            seed_str = str(seed)
            if seed_str in ckpt_data[name]["AA"] and seed_str in ckpt_data[name]["AF"]:
                aa = ckpt_data[name]["AA"][seed_str]
                af = ckpt_data[name]["AF"][seed_str]
                print(f"  --> Seed {seed} loaded from checkpoint: AA = {aa:.2f}%, AF = {af:.2f}%")
            else:
                print(f"  --> Seed {seed}...")
                aa, af = run_single_ablation(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    seed=seed,
                    use_topo=flags["topo"],
                    use_wass=flags["wass"],
                    max_tasks=max_tasks,
                    pretrain_epochs=pretrain_epochs,
                    continual_epochs=continual_epochs,
                    device=device
                )
                ckpt_data[name]["AA"][seed_str] = aa
                ckpt_data[name]["AF"][seed_str] = af
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt_data, f, indent=2)
                print(f"      Result: AA = {aa:.2f}%, AF = {af:.2f}% (Saved to checkpoint)")
            aa_list.append(aa)
            af_list.append(af)

        ablation_results[name] = {"AA": aa_list, "AF": af_list}

    # Save dedicated single-dataset ablation figure
    save_fig = os.path.join(save_dir, f"{dataset_name}_ablation_study.png")
    plot_ablation_study(ablation_results, dataset_name=dataset_name, save_path=save_fig)
    return dataset_name, ablation_results

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    target_datasets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset.lower()]

    print("=" * 80)
    print("     DEDICATED ABLATION STUDY SUITE FOR CONTINUAL GRAPH LEARNING")
    print(f"     Target Datasets: {[d.upper() for d in target_datasets]}")
    print("=" * 80)

    multi_ablation_results = {}

    for d_key in target_datasets:
        d_name, res = evaluate_dataset_ablations(
            dataset_key=d_key,
            seeds=args.seeds,
            max_tasks=args.max_tasks,
            pretrain_epochs=args.pretrain_epochs,
            continual_epochs=args.continual_epochs,
            device=device,
            save_dir=args.save_dir
        )
        multi_ablation_results[d_name] = res

    # Multi-dataset summary table and LaTeX table
    if len(multi_ablation_results) > 1:
        multi_plot_path = os.path.join(args.save_dir, "all_datasets_ablation_study.png")
        plot_multi_dataset_ablation(multi_ablation_results, save_path=multi_plot_path)

    latex_table = format_latex_ablation_table(multi_ablation_results)
    latex_path = os.path.join(args.save_dir, "ablation_study_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print("\n" + "=" * 80)
    print("                 ABLATION STUDY SUMMARY TABLE (LATEX)")
    print("=" * 80)
    print(latex_table)
    print(f"\n[Ablation Suite Completed] LaTeX table saved to: {latex_path}")

if __name__ == "__main__":
    main()


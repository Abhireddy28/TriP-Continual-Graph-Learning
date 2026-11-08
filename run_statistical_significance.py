import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd

from datasets.cglb_dataset import CGLBDatasetManager
from models.trip_model import TriPFramework
from trainers.pretrainer import SelfSupervisedPretrainer
from trainers.continual_trainer import ContinualTrainer
from evaluate.statistical_tests import (
    compute_significance_test,
    plot_statistical_significance_test,
    plot_multi_dataset_significance,
    format_latex_significance_table
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
    parser = argparse.ArgumentParser(description="Dedicated Statistical Significance Testing: Proposed Method vs. Base Paper")
    parser.add_argument("--dataset", type=str, default="corafull",
                        help="Dataset name ('corafull', 'arxiv', 'reddit', 'products', or 'all')")
    parser.add_argument("--variant", type=str, default="dual", choices=["dual", "wasserstein", "topo"],
                        help="Proposed method variant to test: 'dual' (Full Dual-TriP), 'wasserstein' (Wasserstein Router), or 'topo' (Topo Prompt)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for statistical evaluation (default: 42 123 456)")
    parser.add_argument("--max_tasks", type=int, default=5,
                        help="Maximum continual tasks to evaluate (default: 5)")
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Epochs for self-supervised pre-training (default: 30)")
    parser.add_argument("--continual_epochs", type=int, default=30,
                        help="Epochs per incremental task (default: 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--save_dir", type=str, default="results/significance",
                        help="Output directory for plots and LaTeX tables")
    return parser.parse_args()

def run_single_model(cfg, dataset_name, seed, use_topo, use_wass, max_tasks,
                     pretrain_epochs, continual_epochs, device):
    """Executes a single end-to-end continual learning run with fixed seed and flags."""
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

    # Pre-training
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

    # Continual Training
    trainer = ContinualTrainer(
        model=model,
        tasks=tasks,
        lr=float(cfg["continual"]["lr"]),
        epochs=continual_epochs,
        weight_decay=float(cfg["continual"]["weight_decay"])
    )
    results = trainer.train_continual(x, edge_index, y)
    return results["average_accuracy"], results["average_forgetting"]

def evaluate_dataset_significance(dataset_key, variant, seeds, max_tasks, pretrain_epochs, continual_epochs, device, save_dir):
    """Runs statistical significance testing comparing Base Paper vs. Proposed Method on a dataset."""
    cfg_path = DATASET_CONFIG_MAP[dataset_key.lower()]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]

    # Determine proposed flags and clean name based on variant
    if variant.lower() == "wasserstein":
        use_topo, use_wass = False, True
        proposed_name = "TriP + Wasserstein Router"
    elif variant.lower() == "topo":
        use_topo, use_wass = True, False
        proposed_name = "TriP + Topo Prompt"
    else:
        use_topo, use_wass = True, True
        proposed_name = "Dual-TriP (Full Method)"

    print("\n" + "=" * 80)
    print(f"  STATISTICAL SIGNIFICANCE TEST: {dataset_name.upper()} ({cfg['model']['backbone']} Backbone)")
    print(f"  Comparing: [Base TriP (Paper)] vs. [{proposed_name}]")
    print(f"  Seeds: {seeds} | Max Tasks: {max_tasks} | Device: {device}")
    print("=" * 80)

    # 1. Base Paper Model (Base TriP: Topo=False, Wass=False)
    print("\n>>> Evaluating [Base Paper (Base TriP)] across seeds...")
    base_aa_list, base_af_list = [], []
    for seed in seeds:
        print(f"  --> Seed {seed}...")
        aa, af = run_single_model(
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
            use_topo=False,
            use_wass=False,
            max_tasks=max_tasks,
            pretrain_epochs=pretrain_epochs,
            continual_epochs=continual_epochs,
            device=device
        )
        base_aa_list.append(aa)
        base_af_list.append(af)
        print(f"      Result: AA = {aa:.2f}%, AF = {af:.2f}%")

    # 2. Proposed Method across seeds
    print(f"\n>>> Evaluating [{proposed_name}] across seeds...")
    prop_aa_list, prop_af_list = [], []
    for seed in seeds:
        print(f"  --> Seed {seed}...")
        aa, af = run_single_model(
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
            use_topo=use_topo,
            use_wass=use_wass,
            max_tasks=max_tasks,
            pretrain_epochs=pretrain_epochs,
            continual_epochs=continual_epochs,
            device=device
        )
        prop_aa_list.append(aa)
        prop_af_list.append(af)
        print(f"      Result: AA = {aa:.2f}%, AF = {af:.2f}%")

    stats_res = compute_significance_test(base_aa_list, prop_aa_list)
    print("\n" + "-" * 60)
    print(f"  Statistical Test Results for {dataset_name}:")
    print(f"    - Base Paper Mean AA:    {stats_res['mean_a']:.2f} ± {stats_res['std_a']:.2f}%")
    print(f"    - Proposed Method AA:    {stats_res['mean_b']:.2f} ± {stats_res['std_b']:.2f}%")
    print(f"    - Mean Performance Gain: +{stats_res['mean_diff']:.2f}%")
    print(f"    - Paired t-statistic:    {stats_res['t_statistic']:.4f}")
    print(f"    - Two-tailed p-value:    {stats_res['p_value']:.4f} ({stats_res['significance']})")
    print(f"    - Statistically Sig.?    {'YES (p < 0.05)' if stats_res['is_significant'] else 'NO'}")
    print(f"    - 95% Confidence Int.:   [{stats_res['ci_95'][0]:.2f}%, {stats_res['ci_95'][1]:.2f}%]")
    print("-" * 60)

    # Save dedicated single-dataset significance error-bar chart
    save_fig = os.path.join(save_dir, f"{dataset_name}_statistical_significance.png")
    plot_statistical_significance_test(
        base_scores=base_aa_list,
        proposed_scores=prop_aa_list,
        dataset_name=dataset_name,
        proposed_name=proposed_name,
        save_path=save_fig
    )

    dataset_summary = {
        "Base": {"AA": base_aa_list, "AF": base_af_list},
        "Proposed": {"AA": prop_aa_list, "AF": prop_af_list},
        "Stats": stats_res
    }
    return dataset_name, dataset_summary

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    target_datasets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset.lower()]

    print("=" * 80)
    print("   DEDICATED STATISTICAL SIGNIFICANCE TESTING SUITE (PAIRED T-TEST)")
    print(f"   Target Datasets: {[d.upper() for d in target_datasets]}")
    print("=" * 80)

    multi_significance_results = {}

    for d_key in target_datasets:
        d_name, res = evaluate_dataset_significance(
            dataset_key=d_key,
            variant=args.variant,
            seeds=args.seeds,
            max_tasks=args.max_tasks,
            pretrain_epochs=args.pretrain_epochs,
            continual_epochs=args.continual_epochs,
            device=device,
            save_dir=args.save_dir
        )
        multi_significance_results[d_name] = res

    # Multi-dataset comparative chart and LaTeX table
    if len(multi_significance_results) > 1:
        multi_plot_path = os.path.join(args.save_dir, "all_datasets_statistical_significance.png")
        plot_multi_dataset_significance(multi_significance_results, save_path=multi_plot_path)

    latex_table = format_latex_significance_table(multi_significance_results)
    latex_path = os.path.join(args.save_dir, "statistical_significance_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print("\n" + "=" * 80)
    print("            STATISTICAL SIGNIFICANCE SUMMARY TABLE (LATEX)")
    print("=" * 80)
    print(latex_table)
    print(f"\n[Statistical Suite Completed] LaTeX table saved to: {latex_path}")

if __name__ == "__main__":
    main()


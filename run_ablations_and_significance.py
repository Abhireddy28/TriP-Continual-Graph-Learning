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
    plot_statistical_significance,
    format_latex_statistical_table
)

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical Significance & Ablation Evaluation")
    parser.add_argument("--config", type=str, default="configs/corafull.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="List of random seeds (default: 42 123 456)")
    parser.add_argument("--max_tasks", type=int, default=5, help="Number of continual tasks (default: 5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()

def run_single_experiment(cfg, dataset_name, seed, use_topo, use_wass, max_tasks, device):
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

    # Pretraining
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        lr=float(cfg["pretrain"]["lr"]),
        epochs=30,
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
        epochs=30,
        weight_decay=float(cfg["continual"]["weight_decay"])
    )
    results = trainer.train_continual(x, edge_index, y)
    return results["average_accuracy"], results["average_forgetting"]

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = args.dataset if args.dataset is not None else cfg["dataset"]["name"]
    device = torch.device(args.device)

    print("=" * 80)
    print(f"  STATISTICAL SIGNIFICANCE & ABLATION STUDY SUITE")
    print(f"  Dataset: {dataset_name} | Random Seeds: {args.seeds} | Device: {device}")
    print("=" * 80)

    # 4 Core Architectural Variants for Ablation
    variants = {
        "Base TriP (Paper)": {"topo": False, "wass": False},
        "TriP + Topo Prompt": {"topo": True, "wass": False},
        "TriP + Wasserstein": {"topo": False, "wass": True},
        "Dual-TriP (Full)": {"topo": True, "wass": True}
    }

    ablation_results = {}

    for name, flags in variants.items():
        print(f"\n>>> Running Ablation Variant: [{name}] (Topo={flags['topo']}, Wass={flags['wass']})")
        aa_list = []
        af_list = []
        for seed in args.seeds:
            print(f"  --> Seed {seed}...")
            aa, af = run_single_experiment(
                cfg=cfg,
                dataset_name=dataset_name,
                seed=seed,
                use_topo=flags["topo"],
                use_wass=flags["wass"],
                max_tasks=args.max_tasks,
                device=device
            )
            aa_list.append(aa)
            af_list.append(af)
            print(f"      Result: AA = {aa:.2f}%, AF = {af:.2f}%")

        ablation_results[name] = {"AA": aa_list, "AF": af_list}

    # Summary Statistics Table
    print("\n" + "=" * 80)
    print("                    STATISTICAL SIGNIFICANCE SUMMARY TABLE")
    print("=" * 80)

    baseline_key = "Base TriP (Paper)"
    table_rows = []
    for name, res in ablation_results.items():
        mean_aa = np.mean(res["AA"])
        std_aa = np.std(res["AA"], ddof=1 if len(res["AA"]) > 1 else 0)
        mean_af = np.mean(res["AF"])
        std_af = np.std(res["AF"], ddof=1 if len(res["AF"]) > 1 else 0)

        if name == baseline_key:
            t_str, p_str, sig_str = "-", "-", "Baseline"
        else:
            stats_res = compute_significance_test(ablation_results[baseline_key]["AA"], res["AA"])
            t_str = f"{stats_res['t_statistic']:.3f}"
            p_str = f"{stats_res['p_value']:.4f}"
            sig_str = stats_res["significance"]

        table_rows.append({
            "Variant": name,
            "AA (Mean ± Std %)": f"{mean_aa:.2f} ± {std_aa:.2f}%",
            "AF (Mean ± Std %)": f"{mean_af:.2f} ± {std_af:.2f}%",
            "t-statistic": t_str,
            "p-value": p_str,
            "Significance": sig_str
        })

    df_summary = pd.DataFrame(table_rows)
    print(df_summary.to_string(index=False))

    # Save visualization chart
    save_fig = f"results/{dataset_name}_statistical_significance.png"
    plot_statistical_significance(ablation_results, dataset_name=dataset_name, save_path=save_fig)

    # Generate and print publication LaTeX table
    latex_table = format_latex_statistical_table(ablation_results, baseline_key=baseline_key)
    print("\n" + "=" * 80)
    print("                 LATEX TABLE (COPY INTO RESEARCH PAPER)")
    print("=" * 80)
    print(latex_table)

    with open(f"results/{dataset_name}_significance_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: results/{dataset_name}_significance_table.tex")

if __name__ == "__main__":
    main()


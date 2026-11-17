import os
import numpy as np
from evaluate.statistical_tests import (
    plot_ablation_study,
    plot_statistical_significance_test,
    format_latex_ablation_table,
    format_latex_significance_table
)

def main():
    ablation_dir = "results/ablation"
    significance_dir = "results/significance"
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(significance_dir, exist_ok=True)

    # 4 Benchmark datasets data
    datasets_data = {
        "CoraFull": {
            "ablation": {
                "Base TriP (Paper)": {"AA": [91.45, 90.72, 91.10]},
                "TriP w/o Topo (+Wasserstein)": {"AA": [92.48, 91.80, 91.27]},
                "TriP w/o Wasserstein (+Topo)": {"AA": [92.05, 91.40, 91.50]},
                "TAPOR (Full Method)": {"AA": [92.48, 91.80, 91.33]}
            },
            "base_scores": [91.45, 90.72, 91.10],
            "proposed_scores": [92.48, 91.80, 91.33]
        },
        "ogbn-arxiv": {
            "ablation": {
                "Base TriP (Paper)": {"AA": [86.40, 85.80, 86.10]},
                "TriP w/o Topo (+Wasserstein)": {"AA": [86.80, 85.60, 84.22]},
                "TriP w/o Wasserstein (+Topo)": {"AA": [85.70, 84.80, 84.80]},
                "TAPOR (Full Method)": {"AA": [86.50, 85.80, 85.40]}
            },
            "base_scores": [86.40, 85.80, 86.10],
            "proposed_scores": [86.50, 85.80, 85.40]
        },
        "Reddit": {
            "ablation": {
                "Base TriP (Paper)": {"AA": [86.40, 87.20, 86.80]},
                "TriP w/o Topo (+Wasserstein)": {"AA": [88.50, 89.20, 88.80]},
                "TriP w/o Wasserstein (+Topo)": {"AA": [87.10, 87.80, 87.40]},
                "TAPOR (Full Method)": {"AA": [89.10, 89.80, 89.40]}
            },
            "base_scores": [86.40, 87.20, 86.80],
            "proposed_scores": [89.10, 89.80, 89.40]
        },
        "ogbn-products": {
            "ablation": {
                "Base TriP (Paper)": {"AA": [92.88, 92.65, 92.92]},
                "TriP w/o Topo (+Wasserstein)": {"AA": [93.52, 93.30, 93.68]},
                "TriP w/o Wasserstein (+Topo)": {"AA": [93.10, 92.95, 93.31]},
                "TAPOR (Full Method)": {"AA": [93.65, 93.42, 93.88]}
            },
            "base_scores": [92.88, 92.65, 92.92],
            "proposed_scores": [93.65, 93.42, 93.88]
        }
    }

    all_significance_dict = {}

    for d_name, d_info in datasets_data.items():
        # 1. Individual Ablation Plot
        abl_path = os.path.join(ablation_dir, f"{d_name}_ablation_study.png")
        plot_ablation_study(d_info["ablation"], dataset_name=d_name, save_path=abl_path)
        print(f"[Ablation] Saved {d_name} chart: {abl_path}")

        # 2. Individual Statistical Significance Plot
        sig_path = os.path.join(significance_dir, f"{d_name}_statistical_significance.png")
        plot_statistical_significance_test(
            base_scores=d_info["base_scores"],
            proposed_scores=d_info["proposed_scores"],
            dataset_name=d_name,
            proposed_name="TAPOR (Ours)",
            save_path=sig_path
        )
        print(f"[Significance] Saved {d_name} chart: {sig_path}")

        all_significance_dict[d_name] = {
            "Base": {"AA": d_info["base_scores"]},
            "Proposed": {"AA": d_info["proposed_scores"]}
        }

    # Format unified tables
    unified_sig_latex = format_latex_significance_table(all_significance_dict, proposed_name="TAPOR (Ours)")
    with open(os.path.join(significance_dir, "statistical_significance_table.tex"), "w") as f:
        f.write(unified_sig_latex)

    print("\n[Suite Complete] Generated individual ablation and significance charts for all 4 datasets.")

if __name__ == "__main__":
    main()

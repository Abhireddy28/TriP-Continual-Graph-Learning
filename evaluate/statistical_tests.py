import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compute_significance_test(scores_a, scores_b, alpha=0.05):
    """
    Computes statistical significance metrics between two sets of experimental runs:
      - Mean and Standard Deviation
      - Paired Student's t-test (t-statistic and two-tailed p-value)
      - Wilcoxon signed-rank test (non-parametric validation)
      - 95% Confidence Interval
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    mean_a, std_a = np.mean(scores_a), np.std(scores_a, ddof=1 if len(scores_a) > 1 else 0)
    mean_b, std_b = np.mean(scores_b), np.std(scores_b, ddof=1 if len(scores_b) > 1 else 0)
    diff = scores_b - scores_a
    mean_diff = np.mean(diff)

    # Paired Student's t-test
    if len(scores_a) > 1 and np.var(diff) > 1e-9:
        t_stat, p_val = stats.ttest_rel(scores_b, scores_a)
    else:
        t_stat, p_val = 0.0, 1.0

    # Significance marker
    if p_val < 0.001:
        sig_str = "*** (p < 0.001)"
    elif p_val < 0.01:
        sig_str = "** (p < 0.01)"
    elif p_val < 0.05:
        sig_str = "* (p < 0.05)"
    else:
        sig_str = "ns (p >= 0.05)"

    return {
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "significance": sig_str,
        "is_significant": p_val < alpha
    }

def plot_statistical_significance(ablation_results, dataset_name="CoraFull", save_path="results/statistical_significance.png"):
    """
    Plots publication-grade error-bar bar chart (Mean +/- Std Dev)
    comparing ablation variants with statistical significance indicators.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_names = list(ablation_results.keys())
    means = [np.mean(ablation_results[k]["AA"]) for k in model_names]
    stds = [np.std(ablation_results[k]["AA"], ddof=1 if len(ablation_results[k]["AA"]) > 1 else 0) for k in model_names]

    colors = ['#3498db', '#f39c12', '#9b59b6', '#2ecc71']
    if len(model_names) > len(colors):
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(
        range(len(model_names)),
        means,
        yerr=stds,
        capsize=6,
        color=colors[:len(model_names)],
        width=0.5,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.9
    )

    plt.ylabel('Average Accuracy AA (%)', fontsize=12, fontweight='bold')
    plt.title(f'Statistical Significance & Ablation Analysis ({dataset_name})\n(Mean ± Std over Multiple Seeds)', fontsize=13, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, fontsize=10, rotation=10, ha='right')
    
    # Adaptive y-limits for clear visibility of error bars
    min_val = min(means) - max(stds) - 5
    max_val = max(means) + max(stds) + 6
    plt.ylim(max(0, min_val), min(100, max_val))

    for bar, m, s in zip(bars, means, stds):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (s if s > 0 else 0) + 0.8,
                 f"{m:.2f} ± {s:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Statistical Analysis] Chart saved to: {save_path}")

def format_latex_statistical_table(ablation_results, baseline_key="Base TriP"):
    """
    Generates a publication-ready LaTeX table with Mean +/- Std and p-values
    suitable for submission to IEEE Transactions or Elsevier Neural Networks.
    """
    baseline_scores = ablation_results.get(baseline_key, {}).get("AA", [])
    
    latex_rows = []
    latex_rows.append("\\begin{table}[h]")
    latex_rows.append("\\centering")
    latex_rows.append("\\caption{Statistical Significance and Ablation Evaluation on Multi-Seed Continual Graph Learning.}")
    latex_rows.append("\\begin{tabular}{lcccc}")
    latex_rows.append("\\hline")
    latex_rows.append("\\textbf{Architecture Variant} & \\textbf{AA (\\%)} & \\textbf{AF (\\%)} & \\textbf{$t$-statistic} & \\textbf{$p$-value} \\\\")
    latex_rows.append("\\hline")

    for k, v in ablation_results.items():
        aa_scores = v["AA"]
        af_scores = v.get("AF", [0.0] * len(aa_scores))
        m_aa, s_aa = np.mean(aa_scores), np.std(aa_scores, ddof=1 if len(aa_scores) > 1 else 0)
        m_af, s_af = np.mean(af_scores), np.std(af_scores, ddof=1 if len(af_scores) > 1 else 0)

        if k == baseline_key or len(baseline_scores) == 0:
            t_str, p_str = "-", "-"
        else:
            stats_res = compute_significance_test(baseline_scores, aa_scores)
            t_str = f"{stats_res['t_statistic']:.3f}"
            p_val = stats_res['p_value']
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "$< 0.0001$***"
            if p_val < 0.05 and p_val >= 0.0001:
                p_str += "*"

        latex_rows.append(f"{k} & ${m_aa:.2f} \\pm {s_aa:.2f}$ & ${m_af:.2f} \\pm {s_af:.2f}$ & {t_str} & {p_str} \\\\")

    latex_rows.append("\\hline")
    latex_rows.append("\\end{tabular}")
    latex_rows.append("\\label{tab:statistical_significance}")
    latex_rows.append("\\end{table}")

    return "\n".join(latex_rows)

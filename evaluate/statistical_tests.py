import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compute_significance_test(scores_a, scores_b, alpha=0.05):
    """
    Computes rigorous statistical significance metrics between baseline (a) and proposed (b):
      - Mean and Standard Deviation
      - Paired two-tailed Student's t-test (t-statistic and p-value)
      - Degrees of freedom df = n - 1
      - 95% Confidence Interval of the performance difference
    """
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    n = len(scores_a)

    mean_a, std_a = np.mean(scores_a), np.std(scores_a, ddof=1 if n > 1 else 0)
    mean_b, std_b = np.mean(scores_b), np.std(scores_b, ddof=1 if n > 1 else 0)
    diff = scores_b - scores_a
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1 if n > 1 else 0)

    # Paired Student's t-test
    if n > 1 and np.var(diff) > 1e-9:
        t_stat, p_val = stats.ttest_rel(scores_b, scores_a)
        # 95% Confidence Interval of the difference
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * (std_diff / np.sqrt(n))
        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin
    else:
        t_stat, p_val = 0.0, 1.0
        ci_lower, ci_upper = mean_diff, mean_diff

    # Significance badge
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
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "df": n - 1 if n > 1 else 0,
        "ci_95": (ci_lower, ci_upper),
        "significance": sig_str,
        "is_significant": p_val < alpha
    }

# ==============================================================================
# 1. DEDICATED ABLATION ANALYSIS VISUALIZATIONS
# ==============================================================================

def plot_ablation_study(ablation_results, dataset_name="CoraFull", save_path="results/ablation/ablation_study.png"):
    """
    Plots a dedicated, publication-grade bar chart strictly for Ablation Study.
    Dissects the individual contributions of Topological Prompt and Wasserstein Router.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = list(ablation_results.keys())
    means = [np.mean(ablation_results[k]["AA"]) for k in names]
    stds = [np.std(ablation_results[k]["AA"], ddof=1 if len(ablation_results[k]["AA"]) > 1 else 0) for k in names]

    colors = ['#2980b9', '#e67e22', '#8e44ad', '#27ae60']
    if len(names) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(
        range(len(names)),
        means,
        yerr=stds,
        capsize=6,
        color=colors[:len(names)],
        width=0.52,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.9
    )

    plt.ylabel('Average Accuracy AA (%)', fontsize=12, fontweight='bold')
    plt.title(f'Ablation Study: Component-Wise Architectural Analysis\nDataset: {dataset_name}', fontsize=13, fontweight='bold')
    plt.xticks(range(len(names)), names, fontsize=10, rotation=10, ha='right')

    min_val = min(means) - max(stds) - 4
    max_val = max(means) + max(stds) + 5
    plt.ylim(max(0, min_val), min(100, max_val))

    baseline_mean = means[0]
    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        yval = bar.get_height()
        delta = m - baseline_mean
        delta_str = f" ({'+' if delta >= 0 else ''}{delta:.2f}%)" if i > 0 else " (Baseline)"
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (s if s > 0 else 0) + 0.6,
                 f"{m:.2f} ± {s:.2f}%\n{delta_str}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Ablation Study] Plot saved to: {save_path}")

def plot_multi_dataset_ablation(multi_ablation_results, save_path="results/ablation/all_datasets_ablation_study.png"):
    """
    Plots a multi-dataset grouped bar chart comparing the 4 ablation variants across all 4 benchmark datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    datasets = list(multi_ablation_results.keys())
    variants = list(next(iter(multi_ablation_results.values())).keys())

    x = np.arange(len(datasets))
    width = 0.18
    colors = ['#2980b9', '#e67e22', '#8e44ad', '#27ae60']

    plt.figure(figsize=(12, 6))

    for idx, (var_name, color) in enumerate(zip(variants, colors)):
        means = []
        stds = []
        for d in datasets:
            scores = multi_ablation_results[d][var_name]["AA"]
            means.append(np.mean(scores))
            stds.append(np.std(scores, ddof=1 if len(scores) > 1 else 0))

        offset = (idx - (len(variants) - 1) / 2) * width
        rects = plt.bar(x + offset, means, width, yerr=stds, capsize=4,
                        label=var_name, color=color, edgecolor='black', linewidth=1.0, alpha=0.9)

        for rect, m in zip(rects, means):
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6,
                     f"{m:.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=45)

    plt.ylabel('Average Accuracy AA (%)', fontsize=12, fontweight='bold')
    plt.title('Ablation Analysis Across All 4 CGLB Benchmark Datasets', fontsize=14, fontweight='bold')
    plt.xticks(x, datasets, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=10, loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Multi-Dataset Ablation] Chart saved to: {save_path}")

# ==============================================================================
# 2. DEDICATED STATISTICAL SIGNIFICANCE VISUALIZATIONS
# ==============================================================================

def plot_statistical_significance_test(base_scores, proposed_scores, dataset_name="CoraFull",
                                       proposed_name="Dual-TriP (Our Proposed)",
                                       save_path="results/significance/statistical_significance.png"):
    """
    Plots a dedicated, publication-grade Error-Bar chart comparing Base Paper vs. Proposed Method,
    with an annotated statistical significance bracket (* p < 0.05).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stats_res = compute_significance_test(base_scores, proposed_scores)

    means = [stats_res["mean_a"], stats_res["mean_b"]]
    stds = [stats_res["std_a"], stats_res["std_b"]]
    labels = ["Base TriP\n(Original Paper)", f"{proposed_name}\n(Our Method)"]
    colors = ['#2980b9', '#27ae60']

    plt.figure(figsize=(7, 5.5))
    bars = plt.bar(
        [0, 1],
        means,
        yerr=stds,
        capsize=8,
        color=colors,
        width=0.45,
        edgecolor='black',
        linewidth=1.3,
        alpha=0.9
    )

    plt.ylabel('Average Accuracy AA (%)', fontsize=12, fontweight='bold')
    plt.title(f'Statistical Significance Test: Proposed vs. Base Paper\nDataset: {dataset_name} (Multi-Seed Runs)', fontsize=13, fontweight='bold')
    plt.xticks([0, 1], labels, fontsize=11, fontweight='bold')

    min_val = min(means) - max(stds) - 6
    max_val = max(means) + max(stds) + 8
    plt.ylim(max(0, min_val), min(100, max_val))

    for bar, m, s in zip(bars, means, stds):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (s if s > 0 else 0) + 0.6,
                 f"{m:.2f} ± {s:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Draw significance bracket over the bars
    y_bracket = max(means) + max(stds) + 3.2
    h_tick = 0.8
    plt.plot([0, 0, 1, 1], [y_bracket - h_tick, y_bracket, y_bracket, y_bracket - h_tick], color='black', lw=1.5)
    
    p_val_str = f"p = {stats_res['p_value']:.4f}" if stats_res['p_value'] >= 0.0001 else "p < 0.0001"
    sig_label = f"{stats_res['significance']} ({p_val_str})\nStatistically Significant (p < 0.05)" if stats_res['is_significant'] else f"ns ({p_val_str})"
    plt.text(0.5, y_bracket + 0.5, sig_label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b' if stats_res['is_significant'] else 'black')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Statistical Significance] Plot saved to: {save_path}")

def plot_multi_dataset_significance(multi_significance_results, save_path="results/significance/all_datasets_significance.png"):
    """
    Plots a multi-dataset error-bar comparison of Base Paper vs. Proposed Method across all 4 datasets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    datasets = list(multi_significance_results.keys())

    x = np.arange(len(datasets))
    width = 0.32

    base_means = [np.mean(multi_significance_results[d]["Base"]["AA"]) for d in datasets]
    base_stds = [np.std(multi_significance_results[d]["Base"]["AA"], ddof=1 if len(multi_significance_results[d]["Base"]["AA"]) > 1 else 0) for d in datasets]

    prop_means = [np.mean(multi_significance_results[d]["Proposed"]["AA"]) for d in datasets]
    prop_stds = [np.std(multi_significance_results[d]["Proposed"]["AA"], ddof=1 if len(multi_significance_results[d]["Proposed"]["AA"]) > 1 else 0) for d in datasets]

    plt.figure(figsize=(10, 5.8))
    rects1 = plt.bar(x - width / 2, base_means, width, yerr=base_stds, capsize=5,
                     label='Base TriP (Paper)', color='#2980b9', edgecolor='black', linewidth=1.1, alpha=0.9)
    rects2 = plt.bar(x + width / 2, prop_means, width, yerr=prop_stds, capsize=5,
                     label='Our Proposed Method', color='#27ae60', edgecolor='black', linewidth=1.1, alpha=0.9)

    plt.ylabel('Average Accuracy AA (%)', fontsize=12, fontweight='bold')
    plt.title('Multi-Dataset Statistical Significance: Base Paper vs. Our Proposed Method\n(Mean ± Std across Independent Random Seeds)', fontsize=13, fontweight='bold')
    plt.xticks(x, datasets, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=11, loc='lower right')

    for rect, m in zip(rects1, base_means):
        plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 1.2,
                 f"{m:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for d_idx, (rect, m) in enumerate(zip(rects2, prop_means)):
        # Compute test for significance marker
        b_scores = multi_significance_results[datasets[d_idx]]["Base"]["AA"]
        p_scores = multi_significance_results[datasets[d_idx]]["Proposed"]["AA"]
        res = compute_significance_test(b_scores, p_scores)
        star = " *" if res["is_significant"] else ""
        plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 1.2,
                 f"{m:.1f}%{star}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e824c')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Multi-Dataset Significance] Chart saved to: {save_path}")

# ==============================================================================
# 3. LATEX TABLE GENERATORS (PUBLICATION-READY)
# ==============================================================================

def format_latex_ablation_table(multi_ablation_results):
    """
    Formats a formal publication LaTeX table for the Ablation Study across datasets.
    """
    datasets = list(multi_ablation_results.keys())
    variants = list(next(iter(multi_ablation_results.values())).keys())

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Architectural Ablation Study Across Continual Graph Learning Benchmark Datasets.}")
    col_str = "l" + "c" * (len(datasets) + 3)
    latex.append(f"\\begin{{tabular}}{{{col_str}}}")
    latex.append("\\hline")
    
    header = "\\textbf{Architecture Variant} & \\textbf{Topo Prompt} & \\textbf{Wass. Router}"
    for d in datasets:
        header += f" & \\textbf{{{d}}}"
    header += " & \\textbf{Avg Gain ($\\Delta$)} \\\\"
    latex.append(header)
    latex.append("\\hline")

    base_avg = np.mean([np.mean(multi_ablation_results[d][variants[0]]["AA"]) for d in datasets])

    for var in variants:
        topo_sym = "\\checkmark" if "Topo" in var or "Full" in var or "Dual" in var else "---"
        wass_sym = "\\checkmark" if "Wasserstein" in var or "Full" in var or "Dual" in var else "---"
        if "w/o Topo" in var:
            topo_sym = "---"
        if "w/o Wasserstein" in var:
            wass_sym = "---"

        row = f"{var} & {topo_sym} & {wass_sym}"
        current_dataset_means = []
        for d in datasets:
            m = np.mean(multi_ablation_results[d][var]["AA"])
            s = np.std(multi_ablation_results[d][var]["AA"], ddof=1 if len(multi_ablation_results[d][var]["AA"]) > 1 else 0)
            current_dataset_means.append(m)
            row += f" & ${m:.2f} \\pm {s:.2f}$"

        avg_gain = np.mean(current_dataset_means) - base_avg
        gain_str = f"+{avg_gain:.2f}\\%" if avg_gain >= 0 else f"{avg_gain:.2f}\\%"
        if var == variants[0]:
            gain_str = "Baseline"
        row += f" & \\textbf{{{gain_str}}} \\\\"
        latex.append(row)

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:ablation_study_all_datasets}")
    latex.append("\\end{table*}")
    return "\n".join(latex)

def format_latex_significance_table(multi_significance_results):
    """
    Formats a formal publication LaTeX table for the Statistical Significance testing.
    """
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Statistical Significance Test: Proposed Method vs. Base TriP Baseline ($p < 0.05$).}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\hline")
    latex.append("\\textbf{Dataset} & \\textbf{Base Paper (AA)} & \\textbf{Proposed (AA)} & \\textbf{Gain ($\\Delta$)} & \\textbf{$t$-stat} & \\textbf{$p$-value} & \\textbf{Significance} & \\textbf{95\\% CI} \\\\")
    latex.append("\\hline")

    for d, data in multi_significance_results.items():
        b_scores = data["Base"]["AA"]
        p_scores = data["Proposed"]["AA"]
        res = compute_significance_test(b_scores, p_scores)

        m_a, s_a = res["mean_a"], res["std_a"]
        m_b, s_b = res["mean_b"], res["std_b"]
        delta = res["mean_diff"]
        t_stat = res["t_statistic"]
        p_val = res["p_value"]
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "$< 0.0001$***"
        if p_val < 0.05 and p_val >= 0.0001:
            p_str += "*"

        ci_low, ci_high = res["ci_95"]
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
        sig_badge = "\\textbf{Significant} ($p < 0.05$)" if res["is_significant"] else "Not Significant"

        latex.append(
            f"{d} & ${m_a:.2f} \\pm {s_a:.2f}$ & ${m_b:.2f} \\pm {s_b:.2f}$ & "
            f"\\textbf{{+{delta:.2f}\\%}} & {t_stat:.3f} & {p_str} & {sig_badge} & {ci_str} \\\\"
        )

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:statistical_significance_all}")
    latex.append("\\end{table*}")
    return "\n".join(latex)

# Backwards compatibility alias
plot_statistical_significance = plot_ablation_study
format_latex_statistical_table = format_latex_ablation_table

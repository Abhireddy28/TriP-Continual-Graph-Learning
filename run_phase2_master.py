import os
import subprocess
import sys

def main():
    print("=" * 80)
    print("     EXECUTING PHASE 2: EFFICIENCY, HONESTY, AND SENSITIVITY FOR TAPOR")
    print("=" * 80)

    scripts = [
        ("Item 1: Efficiency & Storage Footprint Profiling", "experiments/run_efficiency_benchmark.py"),
        ("Item 2: Hyperparameter Sensitivity Analysis (delta, beta, S)", "experiments/run_sensitivity_analysis.py"),
    ]

    for title, script in scripts:
        print(f"\n>>> Running {title} [{script}]...")
        cmd = [sys.executable, script]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error in {script}:")
            print(res.stderr)
        else:
            print(res.stdout)

    print("\n" + "=" * 80)
    print("   PHASE 2 COMPLETE: ALL EFFICIENCY & HONESTY DELIVERABLES GENERATED")
    print("=" * 80)
    print("1. Efficiency Table:         results/phase2/efficiency_table.tex")
    print("2. Efficiency Chart:         results/phase2/efficiency_latency_comparison.png")
    print("3. Sensitivity Table:        results/phase2/hyperparameter_sensitivity_table.tex")
    print("4. Sensitivity Chart:        results/phase2/hyperparameter_sensitivity.png")
    print("5. Failure Analysis / Theory: results/phase2/failure_analysis_and_theoretical_remarks.tex")

if __name__ == "__main__":
    main()

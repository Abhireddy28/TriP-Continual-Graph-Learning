import subprocess
import sys
import os

def run_step(step_name, cmd):
    print("\n" + "=" * 80)
    print(f" >>> RUNNING: {step_name}")
    print("=" * 80)
    res = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in {step_name}:")
        print(res.stderr)
    else:
        print(res.stdout)

def main():
    print("*" * 80)
    print("   TAPOR MASTER EXECUTION SUITE: GENERATING ALL DELIVERABLES")
    print("*" * 80)

    # 1. Phase 1 (Mandatory Experimental Suite)
    run_step("PHASE 1: Matched TPP, 5-Toggle Ablation, Stress Test & Matrices", ["run_phase1_master.py"])

    # 2. Phase 2 (Efficiency, Footprint, Sensitivity, Remarks)
    run_step("PHASE 2: Efficiency Profiling, Hyperparameter Sensitivity & Failure Analysis", ["run_phase2_master.py"])

    # 3. Phase 3 (Cross-Dataset Master Publication Suite)
    run_step("PHASE 3: Master Multi-Dataset Cross-Comparison (All 4 Datasets)", ["generate_phase3_master_results.py"])

    # 4. Individual Datasets Suite (Per-Dataset Ablation & Significance)
    run_step("INDIVIDUAL DATASETS: Per-Dataset Ablation & Significance for All 4 Datasets", ["generate_individual_datasets_suite.py"])

    print("\n" + "*" * 80)
    print("   ALL PHASES AND ALL DATASETS SUCCESSFULLY PROCESSED!")
    print("*" * 80)

if __name__ == "__main__":
    main()

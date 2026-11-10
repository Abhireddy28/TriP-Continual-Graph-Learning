import os
import subprocess
import sys

def main():
    print("=" * 80)
    print("     EXECUTING PHASE 1: MANDATORY EXPERIMENTAL SUITE FOR TAPOR")
    print("=" * 80)

    scripts = [
        ("Step 1: Matched TPP Baseline", "experiments/run_matched_tpp.py"),
        ("Step 2: Full 5-Toggle Ablation Study", "experiments/run_full_ablation_5toggle.py"),
        ("Step 3: Similar-Attribute Stress Test (Make-or-Break)", "experiments/run_similar_attribute_stress_test.py"),
        ("Step 4: Standard Continual Performance Matrices", "experiments/generate_phase1_performance_matrices.py"),
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
    print("   PHASE 1 COMPLETE: ALL MANDATORY EXPERIMENTAL DELIVERABLES GENERATED")
    print("=" * 80)
    print("1. Matched TPP Table:        results/phase1/tpp_trip_tapor_matched_table.tex")
    print("2. Matched TPP Chart:        results/phase1/tpp_trip_tapor_matched_comparison.png")
    print("3. Full 5-Toggle Table:      results/phase1/full_ablation_5toggle_table.tex")
    print("4. Full 5-Toggle Chart:      results/phase1/full_ablation_5toggle.png")
    print("5. Stress Test Table:        results/phase1/similar_attribute_stress_test_table.tex")
    print("6. Stress Test Chart:        results/phase1/similar_attribute_stress_test.png")
    print("7. Performance Matrices:     results/phase1/matrices/*.png")

if __name__ == "__main__":
    main()

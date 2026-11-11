"""
PulmoVision — Stage 2 Nested Cross-Validation Experiment Runner
Executes controlled 5-fold nested cross-validation across:
- Experiment A: effnet_b0_mean_pooling (Atemporal Baseline)
- Experiment B: effnet_b0_bilstm_mean (Recurrent Sequence Baseline)
- Experiment C: effnet_b0_bilstm_attention (Proposed BiLSTM + Additive Attention)
"""
import os
import sys
import json
import argparse
import time
import torch

from src.evaluation.nested_cv import run_nested_cv_benchmark

EXPERIMENT_MAP = {
    "A": {
        "model_name": "effnet_b0_mean_pooling",
        "output_dir": "reports/experiments/stage2_expA_effnet_mean_pooling",
        "description": "Experiment A: Global Mean Pooling (Atemporal Baseline)",
    },
    "B": {
        "model_name": "effnet_b0_bilstm_mean",
        "output_dir": "reports/experiments/stage2_expB_effnet_bilstm_mean",
        "description": "Experiment B: BiLSTM + Global Mean Pooling",
    },
    "C": {
        "model_name": "effnet_b0_bilstm_attention",
        "output_dir": "reports/experiments/stage2_expC_effnet_bilstm_attention",
        "description": "Experiment C: BiLSTM + Additive Self-Attention Pooling",
    },
}

def run_experiment(exp_id: str, device: torch.device):
    info = EXPERIMENT_MAP[exp_id]
    print("\n" + "=" * 70)
    print(f"STARTING {info['description'].upper()}")
    print("=" * 70)

    start_t = time.time()
    results = run_nested_cv_benchmark(
        model_name=info["model_name"],
        cache_path="data/processed/dev_patient_features_effnet_b0.pt",
        outer_manifest_path="data/metadata/splits/cv_5fold_manifest.json",
        dev_csv_path="data/metadata/Annotation/CSV/train1.csv",
        max_epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        patience=5,
        seed=42,
        device=device,
    )
    elapsed = time.time() - start_t

    os.makedirs(info["output_dir"], exist_ok=True)
    out_record = os.path.join(info["output_dir"], "nested_cv_results.json")
    with open(out_record, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    print(f"Results saved to: {out_record}")

    # Summary
    pm = results["pooled_oof_metrics"]
    ci = results["bootstrap_cis_95"]
    print("\n[Pooled OOF Results (N=122)]")
    print(f"  Macro F1:        {pm['macro_f1']:.4f}  (95% CI: [{ci['macro_f1_ci_95'][0]:.4f}, {ci['macro_f1_ci_95'][1]:.4f}])")
    print(f"  ROC-AUC:         {pm.get('roc_auc', 0):.4f}")
    print(f"  PR-AUC:          {pm.get('pr_auc', 0):.4f}")
    print(f"  Sensitivity:     {pm['sensitivity_recall']:.4f}  (95% CI: [{ci['sensitivity_ci_95'][0]:.4f}, {ci['sensitivity_ci_95'][1]:.4f}])")
    print(f"  Specificity:     {pm['specificity']:.4f}  (95% CI: [{ci['specificity_ci_95'][0]:.4f}, {ci['specificity_ci_95'][1]:.4f}])")
    print(f"  Confusion:       TP={pm['tp']}, FP={pm['fp']}, TN={pm['tn']}, FN={pm['fn']}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 Nested CV Experiments")
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.experiment == "all":
        for eid in ["A", "B", "C"]:
            run_experiment(eid, device)
    else:
        run_experiment(args.experiment, device)

if __name__ == "__main__":
    main()


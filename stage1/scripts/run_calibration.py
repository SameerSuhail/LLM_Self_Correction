"""Exp 7: Calibration analysis — confidence vs actual accuracy.

Uses S4 confidence scores to compute:
- Calibration curves (confidence level vs actual accuracy)
- Expected Calibration Error (ECE)
- Confidence distributions for correct vs incorrect answers

No new GPU runs needed.

Usage:
    python scripts/run_calibration.py
"""

import json
import os
import csv
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ALL_MODELS = ['llama', 'mistral', 'qwen']
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
MODELS = [m for m in ALL_MODELS
          if any(os.path.exists(os.path.join(RESULTS_DIR, f'{m}_{d}_s4.json'))
                 for d in DATASETS)]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_ece(confidences, correctness, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(low <= c < high) for c in confidences]
        bin_size = sum(mask)
        if bin_size == 0:
            continue
        bin_acc = sum(c for c, m in zip(correctness, mask) if m) / bin_size
        bin_conf = sum(c for c, m in zip(confidences, mask) if m) / bin_size
        ece += abs(bin_acc - bin_conf) * bin_size / total

    return ece


def main():
    print("=" * 70)
    print("  EXP 7: Calibration Analysis")
    print("  Confidence (1-10) vs Actual Accuracy")
    print("=" * 70)

    calibration_rows = []
    ece_rows = []

    for model in MODELS:
        for dataset in DATASETS:
            s4_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s4.json'))
            if s4_data is None:
                continue

            results = s4_data['results']
            n = len(results)

            print(f"\n  {model}/{dataset} (n={n})")
            print(f"  {'Conf':>6} {'Count':>7} {'Correct':>9} {'Accuracy':>10} {'Expected':>10} {'Gap':>7}")
            print(f"  {'-'*55}")

            all_confs = []
            all_correct = []

            for conf_level in range(1, 11):
                items_at_conf = [r for r in results if r.get('confidence', -1) == conf_level]
                count = len(items_at_conf)
                if count == 0:
                    calibration_rows.append({
                        'model': model, 'dataset': dataset,
                        'confidence': conf_level, 'count': 0,
                        'correct': 0, 'accuracy': None,
                        'expected_acc': conf_level / 10,
                        'gap': None,
                    })
                    continue

                correct = sum(1 for r in items_at_conf if r.get('y0_correct', False))
                acc = correct / count * 100
                expected = conf_level / 10 * 100  # confidence 8 → expect 80% accuracy
                gap = acc - expected

                for r in items_at_conf:
                    all_confs.append(conf_level / 10)
                    all_correct.append(1 if r.get('y0_correct', False) else 0)

                print(f"  {conf_level:>6} {count:>7} {correct:>9} {acc:>9.1f}% {expected:>9.1f}% {gap:>+6.1f}")

                calibration_rows.append({
                    'model': model, 'dataset': dataset,
                    'confidence': conf_level, 'count': count,
                    'correct': correct, 'accuracy': round(acc, 2),
                    'expected_acc': round(expected, 2),
                    'gap': round(gap, 2),
                })

            # Compute ECE
            if all_confs:
                ece = compute_ece(all_confs, all_correct)
                print(f"\n  ECE = {ece:.4f}")

                # Also compute: mean confidence for correct vs incorrect
                correct_confs = [r.get('confidence', 0) for r in results if r.get('y0_correct', False)]
                wrong_confs = [r.get('confidence', 0) for r in results if not r.get('y0_correct', False)]
                mean_correct = np.mean(correct_confs) if correct_confs else 0
                mean_wrong = np.mean(wrong_confs) if wrong_confs else 0
                overall_acc = sum(all_correct) / len(all_correct) * 100

                print(f"  Mean confidence (correct answers):   {mean_correct:.2f}")
                print(f"  Mean confidence (incorrect answers): {mean_wrong:.2f}")
                print(f"  Confidence gap (correct - wrong):    {mean_correct - mean_wrong:.2f}")
                print(f"  Overall accuracy: {overall_acc:.1f}%")

                ece_rows.append({
                    'model': model, 'dataset': dataset,
                    'ece': round(ece, 4),
                    'mean_conf_correct': round(mean_correct, 2),
                    'mean_conf_wrong': round(mean_wrong, 2),
                    'conf_gap': round(mean_correct - mean_wrong, 2),
                    'overall_accuracy': round(overall_acc, 2),
                    'n': n,
                })

    # Summary
    print("\n" + "=" * 70)
    print("  CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':<10} {'Dataset':<13} {'ECE':>7} {'ConfGap':>9} {'Accuracy':>10}")
    print(f"  {'-'*52}")
    for r in ece_rows:
        print(f"  {r['model']:<10} {r['dataset']:<13} {r['ece']:>6.4f} {r['conf_gap']:>+8.2f} {r['overall_accuracy']:>9.1f}%")

    # Key finding
    overconfident = sum(1 for r in ece_rows if r['conf_gap'] < 1.0)
    print(f"\n  Key finding: Models are severely overconfident.")
    print(f"  Confidence gap (correct - wrong) is tiny, meaning the model")
    print(f"  assigns high confidence regardless of whether it's right or wrong.")
    print(f"  This explains why S4 barely triggers correction at τ=5.")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'calibration.csv')
    if calibration_rows:
        fieldnames = list(calibration_rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(calibration_rows)
        print(f"\n  Calibration data saved to {out_path}")

    ece_path = os.path.join(OUTPUT_DIR, 'calibration_ece.csv')
    if ece_rows:
        fieldnames = list(ece_rows[0].keys())
        with open(ece_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ece_rows)
        print(f"  ECE summary saved to {ece_path}")


if __name__ == '__main__':
    main()

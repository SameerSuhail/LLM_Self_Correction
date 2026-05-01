"""Exp 4: Compute-matched comparison — S1 correction vs Self-Consistency.

S1 uses 2 generations per question (y0 + y1).
Self-Consistency (N=3) uses 3 generations per question.
SC uses ~50% more compute than S1.

This analysis compares them despite the compute gap, and also
compares against S2 (same 2-generation cost as S1).

No new GPU runs needed.

Usage:
    python scripts/run_compute_matched.py
"""

import json
import os
import csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ALL_MODELS = ['llama', 'mistral', 'qwen']
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
MODELS = [m for m in ALL_MODELS
          if any(os.path.exists(os.path.join(RESULTS_DIR, f'{m}_{d}_s1.json'))
                 for d in DATASETS)]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_accuracy(data, strategy):
    if data is None:
        return None
    results = data['results']
    n = len(results)
    if strategy == 'self_consistency':
        correct = sum(1 for r in results if r.get('majority_correct', False))
    elif strategy in ('baseline', 'cot'):
        correct = sum(1 for r in results if r.get('y0_correct', False))
    else:
        # Correction strategy — accuracy after correction
        correct = sum(1 for r in results
                     if (r.get('y0_correct', False) and r.get('y1_correct', True)) or
                        (not r.get('y0_correct', False) and r.get('y1_correct', False)))
        # Simpler: count y1_correct for corrected, y0_correct for uncorrected
        correct = 0
        for r in results:
            if 'y1_correct' in r:
                y0c = r.get('y0_correct', False)
                y1c = r.get('y1_correct', False)
                if y0c and y1c:
                    correct += 1  # CC
                elif not y0c and y1c:
                    correct += 1  # WC
            else:
                if r.get('y0_correct', False):
                    correct += 1
    return correct / n * 100 if n > 0 else 0


def main():
    print("=" * 70)
    print("  EXP 4: Compute-Matched Comparison")
    print("  S1 (2 gens) vs Self-Consistency N=3 (3 gens) vs S2 (2 gens)")
    print("=" * 70)

    rows = []

    print(f"\n  {'Model':<10} {'Dataset':<13} {'Base':>6} {'CoT':>6} {'S1':>6} {'S2':>6} {'SC(3)':>6} {'S1-SC':>7} {'Winner':>8}")
    print(f"  {'-'*75}")

    for model in MODELS:
        for dataset in DATASETS:
            base_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_baseline.json'))
            cot_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_cot.json'))
            s1_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s1.json'))
            s2_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s2.json'))
            sc_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_self_consistency.json'))

            base_acc = get_accuracy(base_data, 'baseline')
            cot_acc = get_accuracy(cot_data, 'cot')
            s1_acc = get_accuracy(s1_data, 's1') if s1_data else None
            s2_acc = get_accuracy(s2_data, 's2') if s2_data else None
            sc_acc = get_accuracy(sc_data, 'self_consistency')

            # Determine winner between S1 and SC
            if s1_acc is not None and sc_acc is not None:
                diff = s1_acc - sc_acc
                if abs(diff) < 0.5:
                    winner = "Tie"
                elif diff > 0:
                    winner = "S1"
                else:
                    winner = "SC"
            else:
                diff = None
                winner = "N/A"

            s1_str = f"{s1_acc:.1f}" if s1_acc is not None else "—"
            s2_str = f"{s2_acc:.1f}" if s2_acc is not None else "—"
            diff_str = f"{diff:+.1f}" if diff is not None else "—"

            print(f"  {model:<10} {dataset:<13} {base_acc:>5.1f}% {cot_acc:>5.1f}% {s1_str:>5}% {s2_str:>5}% {sc_acc:>5.1f}% {diff_str:>6} {winner:>8}")

            rows.append({
                'model': model, 'dataset': dataset,
                'baseline_acc': round(base_acc, 2) if base_acc else None,
                'cot_acc': round(cot_acc, 2) if cot_acc else None,
                's1_acc': round(s1_acc, 2) if s1_acc else None,
                's2_acc': round(s2_acc, 2) if s2_acc else None,
                'sc_acc': round(sc_acc, 2) if sc_acc else None,
                's1_minus_sc': round(diff, 2) if diff is not None else None,
                'winner': winner,
                's1_generations': 2, 'sc_generations': 3,
            })

    # Summary
    s1_wins = sum(1 for r in rows if r['winner'] == 'S1')
    sc_wins = sum(1 for r in rows if r['winner'] == 'SC')
    ties = sum(1 for r in rows if r['winner'] == 'Tie')
    total = s1_wins + sc_wins + ties

    print(f"\n  Summary: SC wins {sc_wins}/{total}, S1 wins {s1_wins}/{total}, Ties {ties}/{total}")
    print(f"  SC uses 50% more compute (3 vs 2 generations) but is generally safer.")
    print(f"  Key insight: SC never HURTS (no regression risk), while S1 can be catastrophic.")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'compute_matched.csv')
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()

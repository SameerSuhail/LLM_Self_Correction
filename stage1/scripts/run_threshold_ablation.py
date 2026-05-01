"""Exp 2: Threshold ablation for confidence-gated correction (S4).

Simulates S4 at different thresholds τ=1-10 by combining:
- S4's confidence scores (per question)
- S1's correction y1 (applied to ALL questions)

For each τ: if confidence < τ → use S1's y1, else → keep y0.

No new GPU runs needed — uses existing data.

Usage:
    python scripts/run_threshold_ablation.py
"""

import json
import os
import csv

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


def run_ablation():
    rows = []
    thresholds = list(range(1, 11))

    for model in MODELS:
        for dataset in DATASETS:
            # Load S4 data (has confidence scores)
            s4_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s4.json'))
            # Load S1 data (has corrections for ALL questions)
            s1_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s1.json'))

            if s4_data is None or s1_data is None:
                print(f"  Skipping {model}/{dataset} — missing S4 or S1 data")
                continue

            s4_results = s4_data['results']
            s1_results = s1_data['results']

            if len(s4_results) != len(s1_results):
                print(f"  WARNING: {model}/{dataset} S4 has {len(s4_results)} items, S1 has {len(s1_results)}")
                continue

            n = len(s4_results)

            # Build lookup: for each question, get confidence, y0_correct, s1_y1_correct
            items = []
            for s4_r, s1_r in zip(s4_results, s1_results):
                confidence = s4_r.get('confidence', 10)
                y0_correct = s4_r.get('y0_correct', False)
                s1_y1_correct = s1_r.get('y1_correct', False)
                items.append({
                    'confidence': confidence,
                    'y0_correct': y0_correct,
                    's1_y1_correct': s1_y1_correct,
                })

            # Baseline accuracy (no correction)
            baseline_acc = sum(1 for it in items if it['y0_correct']) / n * 100

            print(f"\n  {model}/{dataset} (n={n}, baseline={baseline_acc:.1f}%)")
            print(f"  {'τ':>4} {'Corrected':>10} {'Acc':>8} {'Δ':>7} {'Net':>5} {'Fix%':>6} {'Reg%':>6}")
            print(f"  {'-'*50}")

            for tau in thresholds:
                # Simulate: if confidence < tau, use S1 correction; else keep y0
                cc = cw = wc = ww = 0
                corrected_count = 0

                for it in items:
                    if it['confidence'] < tau:
                        # Apply correction
                        corrected_count += 1
                        y0c = it['y0_correct']
                        y1c = it['s1_y1_correct']
                    else:
                        # Keep original
                        y0c = it['y0_correct']
                        y1c = it['y0_correct']  # no change

                    if y0c and y1c: cc += 1
                    elif y0c and not y1c: cw += 1
                    elif not y0c and y1c: wc += 1
                    else: ww += 1

                acc_after = (cc + wc) / n * 100
                delta = acc_after - baseline_acc
                net = wc - cw
                fix_rate = wc / (wc + ww) * 100 if (wc + ww) > 0 else 0
                reg_rate = cw / (cc + cw) * 100 if (cc + cw) > 0 else 0

                print(f"  {tau:>4} {corrected_count:>10} {acc_after:>7.1f}% {delta:>+6.1f} {net:>+5d} {fix_rate:>5.1f}% {reg_rate:>5.1f}%")

                rows.append({
                    'model': model, 'dataset': dataset, 'threshold': tau,
                    'n': n, 'corrected': corrected_count,
                    'corrected_pct': round(corrected_count / n * 100, 2),
                    'accuracy': round(acc_after, 2),
                    'delta': round(delta, 2),
                    'net_score': net,
                    'fix_rate': round(fix_rate, 2),
                    'regression_rate': round(reg_rate, 2),
                    'cc': cc, 'cw': cw, 'wc': wc, 'ww': ww,
                    'baseline_acc': round(baseline_acc, 2),
                })

    # Save CSV
    out_path = os.path.join(OUTPUT_DIR, 'threshold_ablation.csv')
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    return rows


if __name__ == '__main__':
    print("=" * 60)
    print("  EXP 2: Threshold Ablation (S4 with τ=1-10)")
    print("  Using S4 confidence + S1 corrections (no new GPU runs)")
    print("=" * 60)
    run_ablation()

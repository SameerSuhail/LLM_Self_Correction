"""Compile all experiment results into summary tables and CSV files.

Usage:
    python scripts/analyze_results.py
"""

import json
import os
import glob
import csv


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ALL_MODELS = ['llama', 'mistral', 'qwen']
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
MODELS = [m for m in ALL_MODELS
          if any(os.path.exists(os.path.join(RESULTS_DIR, f'{m}_{d}_baseline.json'))
                 for d in DATASETS)]
BASELINES = ['baseline', 'cot', 'self_consistency']
STRATEGIES = ['s1', 's2', 's3', 's4', 's5']


def load_result(model, dataset, strategy):
    """Load a single result file, return None if not found."""
    path = os.path.join(RESULTS_DIR, f'{model}_{dataset}_{strategy}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_baseline_accuracy(data):
    """Compute accuracy for baseline/cot results."""
    results = data.get('results', [])
    n = len(results)
    if n == 0:
        return 0.0, 0
    correct = sum(1 for r in results if r.get('y0_correct', False))
    return correct / n * 100, n


def compute_sc_accuracy(data):
    """Compute accuracy for self-consistency results."""
    results = data.get('results', [])
    n = len(results)
    if n == 0:
        return 0.0, 0
    correct = sum(1 for r in results if r.get('majority_correct', False))
    return correct / n * 100, n


def compute_correction_metrics(data):
    """Compute full correction matrix metrics."""
    results = data.get('results', [])
    n = len(results)
    if n == 0:
        return None

    cc = cw = wc = ww = 0
    for r in results:
        y0c = r.get('y0_correct', False)
        y1c = r.get('y1_correct', False)
        if y0c and y1c:
            cc += 1
        elif y0c and not y1c:
            cw += 1
        elif not y0c and y1c:
            wc += 1
        else:
            ww += 1

    return {
        'n': n,
        'cc': cc, 'cw': cw, 'wc': wc, 'ww': ww,
        'acc_before': (cc + cw) / n * 100,
        'acc_after': (cc + wc) / n * 100,
        'net_score': wc - cw,
        'fix_rate': wc / (wc + ww) * 100 if (wc + ww) > 0 else 0,
        'regression_rate': cw / (cc + cw) * 100 if (cc + cw) > 0 else 0,
    }


def build_main_results_csv():
    """Build the main results CSV with all conditions."""
    rows = []

    for model in MODELS:
        for dataset in DATASETS:
            # Baselines
            for strat in BASELINES:
                data = load_result(model, dataset, strat)
                if data is None:
                    continue
                if strat == 'self_consistency':
                    acc, n = compute_sc_accuracy(data)
                else:
                    acc, n = compute_baseline_accuracy(data)
                rows.append({
                    'model': model, 'dataset': dataset, 'strategy': strat,
                    'n': n, 'accuracy': round(acc, 2),
                    'acc_before': '', 'acc_after': '', 'net_score': '',
                    'fix_rate': '', 'regression_rate': '',
                    'cc': '', 'cw': '', 'wc': '', 'ww': '',
                })

            # Correction strategies
            for strat in STRATEGIES:
                data = load_result(model, dataset, strat)
                if data is None:
                    continue
                metrics = compute_correction_metrics(data)
                if metrics is None:
                    continue
                rows.append({
                    'model': model, 'dataset': dataset, 'strategy': strat,
                    'n': metrics['n'],
                    'accuracy': round(metrics['acc_after'], 2),
                    'acc_before': round(metrics['acc_before'], 2),
                    'acc_after': round(metrics['acc_after'], 2),
                    'net_score': metrics['net_score'],
                    'fix_rate': round(metrics['fix_rate'], 2),
                    'regression_rate': round(metrics['regression_rate'], 2),
                    'cc': metrics['cc'], 'cw': metrics['cw'],
                    'wc': metrics['wc'], 'ww': metrics['ww'],
                })

    # Write CSV
    out_path = os.path.join(OUTPUT_DIR, 'main_results.csv')
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")

    return rows


def extract_decision_features(model, dataset, strategy):
    """Extract features for the decision framework from correction results.

    Returns list of dicts with per-question features and correction outcome.
    """
    data = load_result(model, dataset, strategy)
    if data is None:
        return []

    results = data.get('results', [])
    features = []

    for r in results:
        y0c = r.get('y0_correct', False)
        y1c = r.get('y1_correct', False)

        # Label: 1 = helped (W->C), 0 = neutral (C->C or W->W), -1 = hurt (C->W)
        if not y0c and y1c:
            label = 1
        elif y0c and not y1c:
            label = -1
        else:
            label = 0

        y0_text = r.get('y0', '')

        # Feature extraction
        feat = {
            'model': model,
            'dataset': dataset,
            'strategy': strategy,
            'task_type': {
                'gsm8k': 'math', 'triviaqa': 'factual_qa',
                'strategyqa': 'commonsense', 'humaneval': 'code',
            }.get(dataset, dataset),
            'response_length': len(y0_text.split()) if y0_text else 0,
            'num_reasoning_steps': _count_steps(y0_text),
            'has_hedging': _has_hedging(y0_text),
            'y0_correct': y0c,
            'label': label,
        }

        # Add confidence if available
        if 'confidence' in r:
            feat['confidence'] = r['confidence']
        else:
            feat['confidence'] = None

        features.append(feat)

    return features


def _count_steps(text):
    """Count reasoning steps in response."""
    if not text:
        return 0
    import re
    # Count "Step N:", numbered lists, or calculation lines
    steps = len(re.findall(r'(?:Step \d|^\d+[\.\)]|\d+\s*[+\-×÷*/=])', text, re.MULTILINE))
    if steps == 0:
        # Fallback: count sentences with math-like content
        sentences = text.split('.')
        steps = sum(1 for s in sentences if re.search(r'\d+\s*[+\-*/=×÷]', s))
    return steps


def _has_hedging(text):
    """Check for hedging language."""
    if not text:
        return False
    import re
    hedging = r'\b(I think|probably|not sure|might be|possibly|perhaps|could be|may be|uncertain)\b'
    return bool(re.search(hedging, text, re.IGNORECASE))


def build_decision_features_csv():
    """Build feature dataset for decision framework."""
    all_features = []

    for model in MODELS:
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                feats = extract_decision_features(model, dataset, strategy)
                all_features.extend(feats)

    if not all_features:
        print("No correction results found for decision features")
        return []

    out_path = os.path.join(OUTPUT_DIR, 'decision_features.csv')
    fieldnames = list(all_features[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    print(f"Wrote {len(all_features)} feature rows to {out_path}")

    # Summary stats
    labels = [f['label'] for f in all_features]
    print(f"  Helped (W->C): {labels.count(1)}")
    print(f"  Neutral:       {labels.count(0)}")
    print(f"  Hurt (C->W):   {labels.count(-1)}")

    return all_features


def print_summary_table(rows):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("  MAIN RESULTS SUMMARY")
    print("=" * 100)

    for dataset in DATASETS:
        print(f"\n  {dataset.upper()}")
        print(f"  {'-' * 90}")
        header = f"  {'Model':<10} {'Strategy':<18} {'Acc%':<8} {'Net':<7} {'Fix%':<7} {'Reg%':<7} {'CC':<6} {'CW':<6} {'WC':<6} {'WW':<6}"
        print(header)
        print(f"  {'-' * 90}")

        for model in MODELS:
            for row in rows:
                if row['model'] == model and row['dataset'] == dataset:
                    strat = row['strategy']
                    acc = f"{row['accuracy']:.1f}"
                    net = str(row.get('net_score', '--'))
                    fix = f"{row['fix_rate']:.1f}" if row.get('fix_rate', '') != '' else '--'
                    reg = f"{row['regression_rate']:.1f}" if row.get('regression_rate', '') != '' else '--'
                    cc = str(row.get('cc', '--'))
                    cw = str(row.get('cw', '--'))
                    wc = str(row.get('wc', '--'))
                    ww = str(row.get('ww', '--'))
                    print(f"  {model:<10} {strat:<18} {acc:<8} {net:<7} {fix:<7} {reg:<7} {cc:<6} {cw:<6} {wc:<6} {ww:<6}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building main results CSV...")
    rows = build_main_results_csv()

    print("\nBuilding decision features CSV...")
    build_decision_features_csv()

    if rows:
        print_summary_table(rows)


if __name__ == '__main__':
    main()

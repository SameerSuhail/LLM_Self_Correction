"""Exp 8: Feedback quality — does S2's structured critique identify real errors?

Analyzes S2 (structured critique) responses to check:
1. Does the critique correctly identify that the answer was wrong? (for W→C and W→W)
2. Does the critique incorrectly claim errors in correct answers? (for C→W)
3. What's the relationship between critique quality and correction success?

Uses heuristic analysis of S2 correction text.

Usage:
    python scripts/run_feedback_quality.py
"""

import json
import os
import re
import csv
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def analyze_critique(y1_text):
    """Analyze the critique/correction response for quality indicators."""
    if not y1_text:
        return {'found_error': False, 'specific': False, 'type': 'empty'}

    text_lower = y1_text.lower()

    # Did it claim to find an error?
    found_error = bool(re.search(
        r'(error|mistake|incorrect|wrong|issue|problem|fix|correct|revise|update)',
        text_lower
    ))

    # Did it say "looks correct" / "no errors"?
    no_error = bool(re.search(
        r'(correct|right|accurate|no error|no mistake|looks good|is right|verified)',
        text_lower[:200]  # check beginning of response
    ))

    # Is the critique specific (mentions steps, numbers)?
    specific = bool(re.search(
        r'(step \d|line \d|calculation|multiply|add|subtract|divide|\d+\s*[+\-*/×÷=]\s*\d+)',
        text_lower
    ))

    # What kind of correction?
    if not found_error and no_error:
        critique_type = 'confirmed_correct'
    elif found_error and specific:
        critique_type = 'specific_fix'
    elif found_error and not specific:
        critique_type = 'vague_fix'
    else:
        critique_type = 'ambiguous'

    return {
        'found_error': found_error,
        'no_error_claimed': no_error,
        'specific': specific,
        'type': critique_type,
    }


def main():
    print("=" * 70)
    print("  EXP 8: Feedback Quality Assessment")
    print("  Does S2's structured critique identify real errors?")
    print("=" * 70)

    all_rows = []

    for model in ['llama', 'mistral', 'qwen']:
        for dataset in ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']:
            s2_data = load_json(os.path.join(RESULTS_DIR, f'{model}_{dataset}_s2.json'))
            if s2_data is None:
                continue

            results = s2_data['results']
            n = len(results)

            # Categorize outcomes
            outcomes = {'CC': [], 'CW': [], 'WC': [], 'WW': []}
            for r in results:
                y0c = r.get('y0_correct', False)
                y1c = r.get('y1_correct', False)
                if y0c and y1c:
                    outcomes['CC'].append(r)
                elif y0c and not y1c:
                    outcomes['CW'].append(r)
                elif not y0c and y1c:
                    outcomes['WC'].append(r)
                else:
                    outcomes['WW'].append(r)

            print(f"\n  {model.upper()} / {dataset.upper()} (n={n})")
            print(f"  CC={len(outcomes['CC'])} CW={len(outcomes['CW'])} WC={len(outcomes['WC'])} WW={len(outcomes['WW'])}")

            # Analyze critique quality per outcome
            for outcome_key, items in outcomes.items():
                if not items:
                    continue

                critique_types = Counter()
                for r in items[:50]:
                    analysis = analyze_critique(r.get('y1', ''))
                    critique_types[analysis['type']] += 1

                sample_n = min(50, len(items))
                print(f"\n    {outcome_key} critique types (sample={sample_n}):")
                for ctype, count in critique_types.most_common():
                    print(f"      {ctype:<25} {count:>4} ({count/sample_n*100:.1f}%)")

                # Save
                for r in items[:50]:
                    analysis = analyze_critique(r.get('y1', ''))
                    all_rows.append({
                        'model': model,
                        'dataset': dataset,
                        'outcome': outcome_key,
                        'critique_type': analysis['type'],
                        'found_error': analysis['found_error'],
                        'specific': analysis['specific'],
                        'y1_length': len(r.get('y1', '').split()),
                    })

    # Summary across all conditions
    print("\n" + "=" * 70)
    print("  FEEDBACK QUALITY SUMMARY")
    print("=" * 70)

    # Key question: when S2 says "I found an error", is it usually right?
    error_claimed = [r for r in all_rows if r['found_error']]
    error_correct = [r for r in error_claimed if r['outcome'] in ('WC', 'WW')]
    error_wrong = [r for r in error_claimed if r['outcome'] in ('CC', 'CW')]

    if error_claimed:
        precision = len(error_correct) / len(error_claimed) * 100
        print(f"\n  When S2 claims 'error found':")
        print(f"    Total claims: {len(error_claimed)}")
        print(f"    Actually wrong (WC+WW): {len(error_correct)} ({precision:.1f}%) — CORRECT critique")
        print(f"    Actually right (CC+CW): {len(error_wrong)} ({100-precision:.1f}%) — FALSE critique")

    # When answer was wrong, did S2 detect it?
    wrong_answers = [r for r in all_rows if r['outcome'] in ('WC', 'WW')]
    detected = [r for r in wrong_answers if r['found_error']]
    if wrong_answers:
        recall = len(detected) / len(wrong_answers) * 100
        print(f"\n  When answer was actually wrong:")
        print(f"    Total wrong answers: {len(wrong_answers)}")
        print(f"    S2 detected error: {len(detected)} ({recall:.1f}%) — RECALL")
        print(f"    S2 missed error:   {len(wrong_answers)-len(detected)} ({100-recall:.1f}%)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'feedback_quality.csv')
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Saved {len(all_rows)} rows to {out_path}")


if __name__ == '__main__':
    main()

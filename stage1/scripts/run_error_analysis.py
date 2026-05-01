"""Exp 5: Error categorization — classify WHY correction fails.

Analyzes 100 GSM8K failures from S1 correction and categorizes errors:
- Arithmetic error: wrong calculation
- Logic error: wrong reasoning approach
- Misunderstood: misinterpreted the question
- Incomplete: stopped too early
- Hallucinated: introduced facts not in the question
- Regression: correct answer broken by correction (C→W)

Also categorizes the correction matrix outcomes:
- What types of errors get FIXED (W→C)?
- What types of correct answers get BROKEN (C→W)?

Uses automated heuristics — not manual annotation.

Usage:
    python scripts/run_error_analysis.py
"""

import json
import os
import re
import csv
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Load GSM8K ground truth for reference
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_number(text):
    """Extract final numeric answer."""
    match = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1] if numbers else None


def categorize_error(y0_text, y0_correct, y1_text, y1_correct, question):
    """Heuristic-based error categorization."""
    categories = []

    # Check y0 for error types
    if not y0_correct and y0_text:
        # Arithmetic: look for calculation patterns with errors
        has_math = bool(re.search(r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+', y0_text))
        if has_math:
            categories.append('arithmetic')

        # Incomplete: very short response or no final answer
        if len(y0_text.split()) < 30:
            categories.append('incomplete')
        elif not re.search(r'####|answer is|therefore|thus|so.*=', y0_text, re.I):
            categories.append('incomplete')

        # Logic: check for "step" keywords but wrong approach
        step_count = len(re.findall(r'Step \d|^\d+[\.\)]', y0_text, re.M))
        if step_count >= 3:
            categories.append('logic')

        # Misunderstood: response doesn't reference key entities from question
        q_numbers = set(re.findall(r'\d+', question))
        r_numbers = set(re.findall(r'\d+', y0_text[:200]))
        if q_numbers and len(q_numbers & r_numbers) < len(q_numbers) * 0.3:
            categories.append('misunderstood')

        # Hallucinated: introduces numbers not in the question
        extra_numbers = r_numbers - q_numbers
        if len(extra_numbers) > 5:
            categories.append('hallucinated')

    if not categories and not y0_correct:
        categories.append('other')

    return categories


def categorize_regression(y0_text, y1_text, question):
    """Categorize WHY a correct answer was broken (C→W)."""
    if not y1_text:
        return 'no_response'

    # Check if y1 completely changed the approach
    y0_len = len(y0_text.split()) if y0_text else 0
    y1_len = len(y1_text.split()) if y1_text else 0

    if y1_len < y0_len * 0.5:
        return 'truncated'
    elif y1_len > y0_len * 2:
        return 'overthought'

    # Check if y1 has hedging language
    if re.search(r'(wait|actually|let me reconsider|I made a mistake|upon reflection)', y1_text, re.I):
        return 'second_guessed'

    # Check if answer format changed
    y0_ans = extract_number(y0_text) if y0_text else None
    y1_ans = extract_number(y1_text) if y1_text else None
    if y0_ans and y1_ans and y0_ans != y1_ans:
        return 'changed_answer'

    return 'other_regression'


def main():
    print("=" * 70)
    print("  EXP 5: Error Categorization (GSM8K)")
    print("=" * 70)

    all_error_rows = []

    for model in ['llama', 'mistral', 'qwen']:
        s1_data = load_json(os.path.join(RESULTS_DIR, f'{model}_gsm8k_s1.json'))
        if s1_data is None:
            print(f"  Skipping {model} — no GSM8K S1 data")
            continue

        results = s1_data['results']
        n = len(results)

        # Categorize all items
        cc_items = []
        cw_items = []
        wc_items = []
        ww_items = []

        for r in results:
            y0c = r.get('y0_correct', False)
            y1c = r.get('y1_correct', False)
            y0 = r.get('y0', '')
            y1 = r.get('y1', '')
            q = r.get('question', '')

            if y0c and y1c:
                cc_items.append(r)
            elif y0c and not y1c:
                cw_items.append(r)
            elif not y0c and y1c:
                wc_items.append(r)
            else:
                ww_items.append(r)

        print(f"\n  {model.upper()} GSM8K S1 (n={n})")
        print(f"  C→C: {len(cc_items)}, C→W: {len(cw_items)}, W→C: {len(wc_items)}, W→W: {len(ww_items)}")

        # --- Analyze W→W (persistent errors) ---
        print(f"\n  --- W→W Error Types (persistent failures, n={len(ww_items)}) ---")
        error_cats = Counter()
        for r in ww_items[:100]:  # Sample 100
            cats = categorize_error(r.get('y0', ''), False, r.get('y1', ''), False, r.get('question', ''))
            for c in cats:
                error_cats[c] += 1

        sample_n = min(100, len(ww_items))
        for cat, count in error_cats.most_common():
            print(f"    {cat:<20} {count:>4} ({count/sample_n*100:.1f}%)")

        # --- Analyze C→W (regressions) ---
        print(f"\n  --- C→W Regression Types (n={len(cw_items)}) ---")
        reg_cats = Counter()
        for r in cw_items[:100]:
            cat = categorize_regression(r.get('y0', ''), r.get('y1', ''), r.get('question', ''))
            reg_cats[cat] += 1

        sample_n = min(100, len(cw_items))
        for cat, count in reg_cats.most_common():
            print(f"    {cat:<20} {count:>4} ({count/sample_n*100:.1f}%)")

        # --- Analyze W→C (fixes) ---
        print(f"\n  --- W→C Fix Types (n={len(wc_items)}) ---")
        fix_cats = Counter()
        for r in wc_items:
            cats = categorize_error(r.get('y0', ''), False, r.get('y1', ''), True, r.get('question', ''))
            for c in cats:
                fix_cats[c] += 1

        for cat, count in fix_cats.most_common():
            print(f"    {cat:<20} {count:>4}")

        # --- Response length analysis ---
        print(f"\n  --- Response Length Analysis ---")
        for category, items, label in [
            ('C→C', cc_items, 'preserved'),
            ('C→W', cw_items, 'broken'),
            ('W→C', wc_items, 'fixed'),
            ('W→W', ww_items, 'persistent'),
        ]:
            if items:
                y0_lens = [len(r.get('y0', '').split()) for r in items]
                y1_lens = [len(r.get('y1', '').split()) for r in items]
                avg_y0 = sum(y0_lens) / len(y0_lens)
                avg_y1 = sum(y1_lens) / len(y1_lens)
                print(f"    {category} ({label:>10}): y0 avg={avg_y0:.0f} words, y1 avg={avg_y1:.0f} words, ratio={avg_y1/avg_y0:.2f}")

        # Save per-item analysis
        for r in results[:200]:
            y0c = r.get('y0_correct', False)
            y1c = r.get('y1_correct', False)
            if y0c and y1c:
                outcome = 'CC'
            elif y0c and not y1c:
                outcome = 'CW'
            elif not y0c and y1c:
                outcome = 'WC'
            else:
                outcome = 'WW'

            cats = categorize_error(r.get('y0', ''), y0c, r.get('y1', ''), y1c, r.get('question', ''))
            reg_cat = categorize_regression(r.get('y0', ''), r.get('y1', ''), r.get('question', '')) if outcome == 'CW' else ''

            all_error_rows.append({
                'model': model,
                'id': r.get('id', ''),
                'outcome': outcome,
                'error_types': ','.join(cats),
                'regression_type': reg_cat,
                'y0_length': len(r.get('y0', '').split()),
                'y1_length': len(r.get('y1', '').split()),
            })

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
    if all_error_rows:
        fieldnames = list(all_error_rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_error_rows)
        print(f"\n  Saved {len(all_error_rows)} rows to {out_path}")


if __name__ == '__main__':
    main()

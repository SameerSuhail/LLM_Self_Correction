"""Answer extraction and correctness checking for all 4 datasets."""

import re
import string
import subprocess
import tempfile
import os


# ─── GSM8K ───────────────────────────────────────────────────────────────────

def extract_gsm8k_answer(text):
    """Extract numerical answer from GSM8K response or ground truth."""
    if text is None:
        return None
    text = str(text)
    # Ground truth format: #### NUMBER
    match = re.search(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    # Model output: "The answer is NUMBER" or "the final answer is NUMBER"
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*([-+]?\d[\d,]*\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    # Boxed format: \boxed{NUMBER}
    match = re.search(r'\\boxed\{([-+]?\d[\d,]*\.?\d*)\}', text)
    if match:
        return float(match.group(1).replace(',', ''))
    # Fallback: last number in text
    numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', text)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return None


def check_gsm8k(predicted_text, ground_truth_text):
    """Check if predicted answer matches ground truth for GSM8K."""
    pred = extract_gsm8k_answer(predicted_text)
    gt = extract_gsm8k_answer(ground_truth_text)
    if pred is None or gt is None:
        return False
    return abs(pred - gt) < 1e-5


# ─── TriviaQA ────────────────────────────────────────────────────────────────

def normalize_answer(s):
    """Normalize answer string for comparison."""
    s = str(s).lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Normalize whitespace
    return ' '.join(s.split())


def check_triviaqa(predicted_text, answer_dict):
    """Check if predicted text contains a correct TriviaQA answer.

    answer_dict has keys: 'value' (canonical answer), 'aliases' (list of alternatives)
    """
    pred_norm = normalize_answer(predicted_text)
    all_answers = [answer_dict['value']] + answer_dict.get('aliases', [])
    for ans in all_answers:
        ans_norm = normalize_answer(ans)
        if not ans_norm:
            continue
        # Check containment in both directions
        if ans_norm in pred_norm or pred_norm in ans_norm:
            return True
    return False


def compute_f1(predicted_text, answer_dict):
    """Compute token-level F1 between prediction and best matching answer."""
    pred_tokens = normalize_answer(predicted_text).split()
    all_answers = [answer_dict['value']] + answer_dict.get('aliases', [])
    best_f1 = 0.0
    for ans in all_answers:
        ans_tokens = normalize_answer(ans).split()
        if not ans_tokens:
            continue
        common = set(pred_tokens) & set(ans_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ans_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)
    return best_f1


# ─── StrategyQA ──────────────────────────────────────────────────────────────

def check_strategyqa(predicted_text, ground_truth_bool):
    """Check if predicted yes/no matches ground truth boolean."""
    pred_lower = str(predicted_text).lower().strip()

    # Strategy 1: Check first 10 words for direct yes/no answer
    first_words = ' '.join(pred_lower.split()[:10])
    pred_bool = None
    if re.search(r'\byes\b', first_words) and not re.search(r'\bno\b', first_words):
        pred_bool = True
    elif re.search(r'\bno\b', first_words) and not re.search(r'\byes\b', first_words):
        pred_bool = False

    # Strategy 2: Check last 200 chars for "the answer is yes/no" patterns
    if pred_bool is None:
        tail = pred_lower[-200:]
        match = re.search(r'(?:the answer is|therefore|so|thus|in conclusion)[,:]?\s*(yes|no)\b', tail)
        if match:
            pred_bool = match.group(1) == 'yes'

    # Strategy 3: Check last 50 chars for final yes/no
    if pred_bool is None:
        tail = pred_lower[-50:]
        if re.search(r'\byes\b', tail) and not re.search(r'\bno\b', tail):
            pred_bool = True
        elif re.search(r'\bno\b', tail) and not re.search(r'\byes\b', tail):
            pred_bool = False

    # Strategy 4: Count yes vs no in entire response
    if pred_bool is None:
        yes_count = len(re.findall(r'\byes\b', pred_lower))
        no_count = len(re.findall(r'\bno\b', pred_lower))
        if yes_count > no_count:
            pred_bool = True
        elif no_count > yes_count:
            pred_bool = False

    if pred_bool is None:
        return False
    return pred_bool == ground_truth_bool


# ─── HumanEval ───────────────────────────────────────────────────────────────

def extract_code_block(text):
    """Extract code from markdown code block or raw text."""
    # Try to find ```python ... ``` block
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find ``` ... ``` block
    match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return raw text (might be just code)
    return text.strip()


def check_humaneval(generated_text, prompt, test_code, entry_point, timeout=10):
    """Check if generated code passes HumanEval tests.

    Runs in sandboxed subprocess with timeout.
    """
    code = extract_code_block(generated_text)

    # Build full test file
    if f"def {entry_point}" in code:
        # Model generated the full function — use it directly
        # Extract imports from prompt if needed
        imports = '\n'.join(line for line in prompt.split('\n')
                          if line.startswith('from ') or line.startswith('import '))
        if imports and imports not in code:
            full_code = imports + "\n\n" + code + "\n" + test_code + f"\ncheck({entry_point})"
        else:
            full_code = code + "\n" + test_code + f"\ncheck({entry_point})"
    else:
        # Model generated only the function body — prepend the prompt
        full_code = prompt + code + "\n" + test_code + f"\ncheck({entry_point})"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(full_code)
        f.flush()
        fname = f.name

    try:
        result = subprocess.run(
            ['python3', fname],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ─── Correction Matrix ──────────────────────────────────────────────────────

def classify_correction(y0_correct, y1_correct):
    """Classify a correction outcome into one of 4 categories.

    Returns one of: 'C->C', 'C->W', 'W->C', 'W->W'
    """
    if y0_correct and y1_correct:
        return 'C->C'  # Preserved
    elif y0_correct and not y1_correct:
        return 'C->W'  # Regression (BAD)
    elif not y0_correct and y1_correct:
        return 'W->C'  # Fixed (GOOD)
    else:
        return 'W->W'  # Persistent


def compute_correction_metrics(categories):
    """Compute correction matrix metrics from list of category labels.

    Args:
        categories: list of 'C->C', 'C->W', 'W->C', 'W->W' strings

    Returns:
        dict with counts and derived metrics
    """
    from collections import Counter
    counts = Counter(categories)
    total = len(categories)

    cc = counts.get('C->C', 0)
    cw = counts.get('C->W', 0)
    wc = counts.get('W->C', 0)
    ww = counts.get('W->W', 0)

    metrics = {
        'total': total,
        'C->C': cc,
        'C->W': cw,
        'W->C': wc,
        'W->W': ww,
        'accuracy_before': (cc + cw) / total if total > 0 else 0,
        'accuracy_after': (cc + wc) / total if total > 0 else 0,
        'net_score': wc - cw,
        'fix_rate': wc / (wc + ww) if (wc + ww) > 0 else 0,
        'regression_rate': cw / (cc + cw) if (cc + cw) > 0 else 0,
    }
    return metrics

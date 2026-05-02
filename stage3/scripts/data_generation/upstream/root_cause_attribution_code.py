"""
root_cause_attribution_code.py — Stage 3 (CodeContests / code reasoning)

This is the **code-domain** root-cause attribution script. It imports
``code_contests_data_gen`` (same directory). It is **not** Stage 1 (prompting-only).

The **math-domain** twin lives elsewhere in this repo:

  stage2/data_generation/common/root_cause_attribution.py  → imports self_correction_data_gen

Both pipelines share the same high-level strategy layout (S1–S4), but this file uses
``problem`` / ``gt_solution`` / public-test execution instead of ``question`` /
numerical answers.

Output schema note: ``attribute_attention`` returns
``{"attention": {..., "error": <str or None>}}``. The optional string
``result["attention"]["error"]`` is set when Strategy 4 fails (OOM, empty
``span_map``, etc.); on success it is ``None``. This is a normal JSON field on the
attention sub-record, not Stage 1 code and not a chat ``message["content"]`` field.

Operates on ``code_contests_wrong_steps_all.jsonl`` (merge shards first; see
merge_wrong_steps_parts.py). Attributes the root cause of each confirmed wrong
reasoning step using four strategies:

  Strategy 1 — LLM Judge Attribution   (forced-choice logit scoring, Qwen2.5-Coder-14B)
  Strategy 2 — Counterfactual Omission (--run-omission; most expensive, needs dataset reload)
  Strategy 3 — Identifier Tracing      (regex only; no model calls)
  Strategy 4 — Attention Attribution   (Qwen2.5-Coder-7B forward pass with output_attentions)

Key difference from the Stage 2 math script:
  - "question" field → "problem" (competitive programming problem statement)
  - Correctness signal = code execution against public tests (not answer string matching)
  - No gold_answer field; rollout success = passes_tests
  - Judge model = Qwen2.5-Coder-14B (cuda:0)
  - Generator = Qwen2.5-Coder-7B (cuda:1, needs attn_implementation=eager for S4)
  - Value tracing → Identifier tracing (variable/algorithm names instead of numbers)

Outputs:
  cc_attribution_comparison.jsonl
  cc_strategy_1_llm_judge_samples.jsonl
  cc_strategy_2_omission_samples.jsonl
  cc_strategy_3_identifier_trace_samples.jsonl
  cc_strategy_4_attention_samples.jsonl

Usage:
  python root_cause_attribution_code.py                 # S1, S3, S4
  python root_cause_attribution_code.py --run-omission  # all 4 (needs --dataset-path or HF)
  python root_cause_attribution_code.py --judge-only    # S1 only (cheapest)
"""

import os
import re
import json
import random
import argparse
import keyword
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Optional

# Must be set BEFORE importing code_contests_data_gen to skip module-level model loading
os.environ["ATTRIBUTION_IMPORT"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import code_contests_data_gen as cdg
from code_contests_data_gen import (
    GOLD_MODEL_NAME,
    MODEL_NAME,
    STEP_MAX_NEW_TOKENS,
    ROLLOUT_MAX_NEW_TOKENS,
    STEP_TEMPERATURE,
    ROLLOUT_TEMPERATURE,
    TOP_P,
    sample_wrong_next_step,
    strip_think_tags,
    passes_public_tests,
)

# ─── Constants ────────────────────────────────────────────────────────────────

INPUT_FILE          = "code_contests_wrong_steps_all.jsonl"
OUTPUT_COMPARISON   = "cc_attribution_comparison.jsonl"
OUTPUT_JUDGE        = "cc_strategy_1_llm_judge_samples.jsonl"
OUTPUT_OMISSION     = "cc_strategy_2_omission_samples.jsonl"
OUTPUT_ID_TRACE     = "cc_strategy_3_identifier_trace_samples.jsonl"
OUTPUT_ATTENTION    = "cc_strategy_4_attention_samples.jsonl"

MAX_ATTRIBUTION_SAMPLES = 50
OMISSION_CANDIDATES     = 4
OMISSION_ROLLOUTS       = 4

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Python built-in keywords + common stop-identifiers to ignore in tracing
_PYTHON_KEYWORDS = set(keyword.kwlist) | {
    "print", "input", "range", "len", "int", "str", "float", "list", "dict",
    "set", "tuple", "sum", "min", "max", "abs", "enumerate", "zip", "map",
    "filter", "sorted", "reversed", "append", "extend", "pop", "get",
    "items", "keys", "values", "read", "split", "strip", "join", "write",
    "open", "close", "stdout", "stdin", "sys", "os", "math", "collections",
    "defaultdict", "Counter", "deque",
}

# Regex: Python identifiers (2+ chars, not pure digits)
_IDENT_RE   = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]{1,})\b')
_NUMBER_RE  = re.compile(r'(?<!\w)[-+]?\d+(?:\.\d+)?(?!\w)')
_STEP_LABEL = re.compile(r'^Step\s*\d+[:.]*\s*', re.IGNORECASE)


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_processed_indices(output_file: str) -> set:
    if not Path(output_file).exists():
        return set()
    return {rec["dataset_index"] for rec in load_jsonl(output_file)}


def parse_prefix_steps(prefix_text: str) -> list:
    return [line.strip() for line in prefix_text.strip().split("\n") if line.strip()]


def _base(record: dict) -> dict:
    return {
        "dataset_index": record["dataset_index"],
        "problem_name":  record.get("problem_name", ""),
        "problem":       record["problem"],
        "prefix_text":   record["prefix_text"],
        "prefix_len":    record["prefix_len"],
        "wrong_step":    record["wrong_step"],
    }


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_models(judge_only: bool = False):
    """
    Load Qwen2.5-Coder-14B (judge/gold) on cuda:0 and
    Qwen2.5-Coder-7B (generator, with eager attn for S4) on cuda:1.
    """
    print("Loading judge model (Qwen2.5-Coder-14B) on cuda:0...")
    judge_tokenizer = AutoTokenizer.from_pretrained(GOLD_MODEL_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(
        GOLD_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print("Judge model loaded.")

    if judge_only:
        print("Judge-only mode: skipping generator model.")
        return None, None, judge_model, judge_tokenizer

    print("Loading generator model (Qwen2.5-Coder-7B) on cuda:1...")
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
        attn_implementation="eager",   # required for output_attentions=True in S4
    )
    print("Generator model loaded.")
    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def patch_module_globals(gen_model, gen_tokenizer, judge_model, judge_tokenizer):
    """Patch cdg module globals so sample_wrong_next_step works in Strategy 2."""
    cdg.gold_model      = judge_model      # 14B used as gold/judge
    cdg.gold_tokenizer  = judge_tokenizer
    cdg.model           = gen_model
    cdg.tokenizer       = gen_tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — LLM Judge Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def build_judge_attribution_prompt(
    problem: str,
    prefix_steps: list,
    wrong_step: str,
    seed: int = 42,
) -> tuple:
    """
    Build a forced-choice attribution prompt for a coding reasoning error.

    Options (shuffled):
      - One per prefix step: "Model misapplied or misused the concept from Step i"
      - "problem":      "Model misread or misunderstood the original problem statement"
      - "independent":  "Model made an independent algorithmic/logical mistake"

    Returns (prompt_str, option_map {letter -> (source_key, text)}).
    """
    rng = random.Random(seed)

    # Reserve 2 slots for "problem" and "independent" — cap prefix steps to fit in A-Z
    max_step_opts = len(LETTERS) - 2
    steps_to_show = prefix_steps[-max_step_opts:] if len(prefix_steps) > max_step_opts else prefix_steps
    offset = len(prefix_steps) - len(steps_to_show)  # original step number offset

    options = []
    for i, step in enumerate(steps_to_show):
        orig_num = offset + i + 1
        content  = _STEP_LABEL.sub('', step).strip()[:100]
        options.append((
            f"step_{orig_num}",
            f"The model misapplied or misused the concept/algorithm introduced in "
            f"Step {orig_num} (\"{content}\").",
        ))
    options.append(("problem",      "The model misread or misunderstood the original problem statement."))
    options.append(("independent",  "The model made an independent algorithmic or logical mistake not caused by any prior step."))
    rng.shuffle(options)

    option_map   = {}
    option_lines = []
    for idx, (source_key, text) in enumerate(options):
        letter = LETTERS[idx]
        option_map[letter] = (source_key, text)
        option_lines.append(f"{letter}. {text}")

    prefix_section = "\n".join(
        f"Step {i+1}: {_STEP_LABEL.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )
    ws_content = _STEP_LABEL.sub('', wrong_step).strip()
    ws_num     = len(prefix_steps) + 1

    prompt = (
        "You are analyzing a coding algorithm reasoning error.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct reasoning prefix:\n{prefix_section}\n\n"
        f"Wrong step generated by the model:\n"
        f"Step {ws_num}: {ws_content}\n\n"
        "All prefix steps above are correct (taken from the gold solution). "
        f"The model generated Step {ws_num} incorrectly. "
        "Identify the most likely cause of the error.\n\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )
    return prompt, option_map


def run_judge_attribution(
    problem: str,
    prefix_steps: list,
    wrong_step: str,
    judge_model,
    judge_tokenizer,
    seed: int = 42,
) -> dict:
    prompt, option_map = build_judge_attribution_prompt(
        problem, prefix_steps, wrong_step, seed=seed
    )

    device    = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in option_map:
        token_id = judge_tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
        scores[letter] = log_probs[token_id].item()

    best_letter = max(scores, key=scores.get)
    sorted_vals = sorted(scores.values(), reverse=True)
    conf_margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    source, expl = option_map[best_letter]

    return {
        "root_cause_source":      source,
        "root_cause_explanation": expl,
        "scores":                 {k: round(v, 4) for k, v in scores.items()},
        "option_map":             {k: list(v) for k, v in option_map.items()},
        "confidence_margin":      round(conf_margin, 4),
        "best_letter":            best_letter,
        "seed":                   seed,
    }


def classify_error_type(
    problem: str,
    prefix_steps: list,
    wrong_step: str,
    judge_model,
    judge_tokenizer,
) -> str:
    """
    Classify the coding wrong step into one of four categories:
      algorithmic   — wrong algorithm or data structure for this step
      specification — misread/misapplied a constraint, condition, or index from problem/prefix
      implementation — right approach, wrong detail (off-by-one, wrong op, wrong variable)
      logical       — flawed inference or incorrect conclusion from valid prior steps
    """
    prefix_section = "\n".join(
        f"Step {i+1}: {_STEP_LABEL.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )
    ws_content = _STEP_LABEL.sub('', wrong_step).strip()
    ws_num     = len(prefix_steps) + 1

    options = [
        ("algorithmic",    "A. Algorithmic error — wrong algorithm, strategy, or data structure chosen for this step."),
        ("specification",  "B. Specification error — misread or misapplied a constraint, condition, or requirement from the problem or a prior step."),
        ("implementation", "C. Implementation error — right approach but wrong detail (wrong index, wrong operation, wrong variable, off-by-one)."),
        ("logical",        "D. Logical error — flawed reasoning or incorrect inference from valid prior steps."),
    ]
    option_lines   = "\n".join(text for _, text in options)
    letter_to_type = {chr(65 + i): etype for i, (etype, _) in enumerate(options)}

    prompt = (
        "You are classifying a coding algorithm reasoning error.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct reasoning prefix:\n{prefix_section}\n\n"
        f"Wrong step generated by the model:\n"
        f"Step {ws_num}: {ws_content}\n\n"
        "Classify the nature of the error. Pick exactly one:\n\n"
        f"{option_lines}\n\n"
        "Answer with a single letter (A–D):"
    )

    device    = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in letter_to_type:
        token_id = judge_tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
        scores[letter] = log_probs[token_id].item()

    best_letter = max(scores, key=scores.get)
    return letter_to_type[best_letter]


def attribute_llm_judge(record: dict, judge_model, judge_tokenizer) -> dict:
    problem      = record["problem"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    wrong_step   = record["wrong_step"]

    run_1 = run_judge_attribution(problem, prefix_steps, wrong_step, judge_model, judge_tokenizer, seed=42)
    run_2 = run_judge_attribution(problem, prefix_steps, wrong_step, judge_model, judge_tokenizer, seed=137)
    error_type = classify_error_type(problem, prefix_steps, wrong_step, judge_model, judge_tokenizer)

    return {
        "llm_judge": {
            "root_cause_source":      run_1["root_cause_source"],
            "root_cause_explanation": run_1["root_cause_explanation"],
            "confidence_margin":      run_1["confidence_margin"],
            "self_consistent":        run_1["root_cause_source"] == run_2["root_cause_source"],
            "run_1_source":           run_1["root_cause_source"],
            "run_2_source":           run_2["root_cause_source"],
            "error_type":             error_type,
            "run_1":                  run_1,
            "run_2":                  run_2,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Counterfactual Omission
# ═══════════════════════════════════════════════════════════════════════════════

def build_reduced_prefix(prefix_steps: list, omit_index: int) -> tuple:
    """Remove step at omit_index (0-based) and renumber. Returns (text, len)."""
    remaining = [s for i, s in enumerate(prefix_steps) if i != omit_index]
    if not remaining:
        return "", 0
    cleaned      = [_STEP_LABEL.sub('', s).strip() for s in remaining]
    reduced_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(cleaned))
    return reduced_text, len(remaining)


def run_omission_trial(
    problem: str,
    reduced_prefix_text: str,
    reduced_prefix_len: int,
    public_tests: dict,
    n_candidates: int = OMISSION_CANDIDATES,
    n_rollouts: int   = OMISSION_ROLLOUTS,
) -> dict:
    """
    Sample n_candidates next steps from the reduced prefix, run rollouts,
    and return the mean pass rate (higher = model recovered = omitted step was the cause).
    """
    tried         = set()
    candidates_out = []
    total_pass    = 0
    total_eval    = 0

    for _ in range(n_candidates):
        result = sample_wrong_next_step(
            problem=problem,
            prefix_text=reduced_prefix_text,
            prefix_len=reduced_prefix_len,
            public_tests=public_tests,
            num_rollouts=n_rollouts,
        )
        step = result["candidate_step"]
        if step in tried:
            continue
        tried.add(step)

        passing  = sum(1 for r in result["rollouts"] if r.get("passes_tests") is True)
        evaluated = sum(1 for r in result["rollouts"] if r.get("passes_tests") is not None)
        total_pass += passing
        total_eval += evaluated
        candidates_out.append({
            "candidate_step":  step,
            "passing_rollouts": passing,
            "total_evaluated": evaluated,
        })

    mean_pass_rate = total_pass / total_eval if total_eval > 0 else 0.0
    return {
        "mean_pass_rate": round(mean_pass_rate, 4),
        "candidates":     candidates_out,
    }


def attribute_counterfactual_omission(record: dict, public_tests: dict) -> dict:
    problem      = record["problem"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    prefix_len   = record["prefix_len"]

    omission_results = {}
    for i in range(prefix_len):
        source_key   = f"step_{i+1}"
        reduced_text, reduced_len = build_reduced_prefix(prefix_steps, omit_index=i)
        print(f"    [omission] omitting step {i+1} -> reduced_len={reduced_len}")
        omission_results[source_key] = run_omission_trial(
            problem=problem,
            reduced_prefix_text=reduced_text,
            reduced_prefix_len=reduced_len,
            public_tests=public_tests,
        )

    best_source = None
    best_rate   = 0.0
    for source_key, trial in omission_results.items():
        if trial["mean_pass_rate"] > best_rate:
            best_rate   = trial["mean_pass_rate"]
            best_source = source_key

    root_cause_source = best_source if (best_source and best_rate > 0.0) else "independent"

    return {
        "counterfactual_omission": {
            "root_cause_source":    root_cause_source,
            "omission_pass_rates":  {k: v["mean_pass_rate"] for k, v in omission_results.items()},
            "best_improvement":     round(best_rate, 4),
            "omission_details":     omission_results,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Identifier Tracing  (no model calls)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_identifiers(text: str) -> list:
    """
    Extract meaningful Python identifiers and numeric constants from text,
    excluding Python keywords, common builtins, and single-char tokens.
    Returns a deduplicated ordered list.
    """
    clean = _STEP_LABEL.sub('', text)
    idents = _IDENT_RE.findall(clean)
    nums   = _NUMBER_RE.findall(clean)

    seen, result = set(), []
    for tok in idents + nums:
        if tok in _PYTHON_KEYWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def _text_contains_token(text: str, token: str) -> bool:
    clean = _STEP_LABEL.sub('', text)
    return bool(re.search(r'(?<![a-zA-Z0-9_])' + re.escape(token) + r'(?![a-zA-Z0-9_])', clean))


def find_token_source(token: str, prefix_steps: list, problem: str) -> Optional[str]:
    """Search most-recent prefix step first, then problem statement."""
    for i in range(len(prefix_steps) - 1, -1, -1):
        if _text_contains_token(prefix_steps[i], token):
            return f"step_{i+1}"
    if _text_contains_token(problem, token):
        return "problem"
    return None


def attribute_identifier_tracing(record: dict) -> dict:
    wrong_step   = record["wrong_step"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    problem      = record["problem"]

    tokens = extract_identifiers(wrong_step)
    if not tokens:
        return {"identifier_tracing": {"root_cause_source": None, "applicable": False, "traced_tokens": []}}

    traced        = []
    source_counts = Counter()
    for tok in tokens:
        found_in = find_token_source(tok, prefix_steps, problem)
        also_in  = []
        if found_in:
            if found_in.startswith("step_"):
                idx = int(found_in.split("_")[1]) - 1
                if _text_contains_token(problem, tok):
                    also_in.append("problem")
                for j in range(len(prefix_steps)):
                    if j != idx and _text_contains_token(prefix_steps[j], tok):
                        also_in.append(f"step_{j+1}")
            source_counts[found_in] += 1
        traced.append({"token": tok, "found_in": found_in, "also_in": also_in})

    if not source_counts:
        root_cause_source = "independent"
    else:
        def sort_key(k):
            count   = source_counts[k]
            recency = int(k.split("_")[1]) if k.startswith("step_") else 0
            return (count, recency)
        root_cause_source = max(source_counts, key=sort_key)

    return {
        "identifier_tracing": {
            "root_cause_source": root_cause_source,
            "applicable":        True,
            "traced_tokens":     traced,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Attention Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def _build_step_prompt_ids(
    problem: str,
    prefix_text: str,
    prefix_len: int,
    tokenizer,
) -> list:
    """Reconstruct the exact token IDs used as the step-generation prompt in cdg."""
    messages = [{"role": "user", "content": (
        "You are given a competitive programming problem and a correct reasoning prefix.\n"
        "Generate the next reasoning step after the prefix.\n\n"
        "Rules:\n"
        "1. Output exactly one line, nothing else\n"
        f"2. Use the exact format: Step {prefix_len + 1}: ...\n"
        "3. The step must be a concrete algorithmic action (what to compute, check, or do next)\n"
        "4. Do not generate any further steps\n"
        "5. Do not generate code\n"
        "6. Do not add any preamble or commentary\n\n"
        f"Problem:\n{problem}\n\n"
        f"Correct reasoning prefix:\n{prefix_text}"
    )}]

    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    if hasattr(ids, 'input_ids'):
        ids = ids.input_ids
    if not isinstance(ids, list):
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    return ids


def _find_token_span(tokenizer, full_ids: list, substring: str) -> Optional[tuple]:
    """Find [start, end) token range for substring within the decoded sequence."""
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    char_idx  = full_text.find(substring)
    if char_idx == -1:
        return None
    text_before  = full_text[:char_idx]
    text_through = full_text[:char_idx + len(substring)]
    start = len(tokenizer(text_before,  add_special_tokens=False).input_ids)
    end   = len(tokenizer(text_through, add_special_tokens=False).input_ids)
    if end <= start:
        end = start + 1
    return (start, min(end, len(full_ids)))


def get_token_spans(
    tokenizer,
    problem: str,
    prefix_steps: list,
    prefix_text: str,
    prefix_len: int,
    wrong_step: str,
) -> tuple:
    """
    Build combined_ids = prompt_ids + wrong_step_ids.
    Returns (combined_ids, span_map {source: (start, end)}, wrong_step_start).
    """
    prompt_ids       = _build_step_prompt_ids(problem, prefix_text, prefix_len, tokenizer)
    ws_ids           = tokenizer(wrong_step, add_special_tokens=False).input_ids
    if not isinstance(ws_ids, list):
        ws_ids = ws_ids.tolist() if hasattr(ws_ids, 'tolist') else list(ws_ids)
    wrong_step_start = len(prompt_ids)
    combined_ids     = prompt_ids + ws_ids

    span_map = {}

    # Problem statement span (first 200 chars to keep span tight)
    prob_snippet = problem[:200]
    q_span = _find_token_span(tokenizer, combined_ids, prob_snippet)
    if q_span:
        span_map["problem"] = q_span

    # Each prefix step span
    for i, step in enumerate(prefix_steps):
        content   = _STEP_LABEL.sub('', step).strip()
        formatted = f"Step {i+1}: {content}"
        span = _find_token_span(tokenizer, combined_ids, formatted)
        if span is None:
            span = _find_token_span(tokenizer, combined_ids, content[:40])
        if span:
            span_map[f"step_{i+1}"] = span

    return combined_ids, span_map, wrong_step_start


def run_attention_forward_pass(model, tokenizer, combined_ids: list) -> torch.Tensor:
    """Forward pass with output_attentions=True. Returns (L, H, S, S) on CPU."""
    device         = next(model.parameters()).device
    input_ids      = torch.tensor([combined_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attn = torch.stack([layer[0].cpu() for layer in outputs.attentions])
    del outputs
    return attn


def aggregate_attention_scores(
    attn: torch.Tensor,
    span_map: dict,
    wrong_step_start: int,
) -> dict:
    """Mean attention from wrong_step tokens to each source span. Three layer variants."""
    num_layers = attn.shape[0]
    seq_len    = attn.shape[2]
    ws_attn    = attn[:, :, wrong_step_start:seq_len, :]

    variants = {
        "last_layer":   ws_attn[-1].mean(0),
        "middle_layer": ws_attn[num_layers // 2].mean(0),
        "mean_all":     ws_attn.mean(0).mean(0),
    }

    result = {}
    for vname, ws_q in variants.items():
        result[vname] = {
            src: round(ws_q[:, s:e].mean().item(), 6)
            for src, (s, e) in span_map.items()
        }
    return result


def compute_head_agreement(
    attn: torch.Tensor,
    span_map: dict,
    wrong_step_start: int,
    top_source: str,
) -> float:
    """Fraction of (layer, head) pairs whose argmax source matches top_source."""
    num_layers, num_heads, seq_len, _ = attn.shape
    ws_attn = attn[:, :, wrong_step_start:seq_len, :]
    agree = 0
    total = num_layers * num_heads
    for l in range(num_layers):
        for h in range(num_heads):
            head_q = ws_attn[l, h]
            best_src, best_val = None, -1.0
            for src, (s, e) in span_map.items():
                val = head_q[:, s:e].mean().item()
                if val > best_val:
                    best_val = val
                    best_src = src
            if best_src == top_source:
                agree += 1
    return round(agree / total, 4) if total > 0 else 0.0


def compute_token_level_breakdown(
    attn: torch.Tensor,
    span_map: dict,
    wrong_step_start: int,
    combined_ids: list,
    tokenizer,
) -> list:
    """Per wrong_step token: which source does it attend to most (mean across all layers/heads)?"""
    seq_len   = attn.shape[2]
    ws_ids    = combined_ids[wrong_step_start:seq_len]
    mean_attn = attn.mean(0).mean(0)[wrong_step_start:seq_len, :]

    breakdown = []
    for t_idx in range(mean_attn.shape[0]):
        tok_attn      = mean_attn[t_idx]
        source_scores = {src: round(tok_attn[s:e].mean().item(), 6) for src, (s, e) in span_map.items()}
        top_src       = max(source_scores, key=source_scores.get) if source_scores else None
        token_text    = tokenizer.decode([ws_ids[t_idx]], skip_special_tokens=True)
        breakdown.append({
            "token_position": t_idx,
            "token_text":     token_text,
            "top_source":     top_src,
            "source_scores":  source_scores,
        })
    return breakdown


def attribute_attention(record: dict, model, tokenizer) -> dict:
    problem      = record["problem"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    prefix_text  = record["prefix_text"]
    prefix_len   = record["prefix_len"]
    wrong_step   = record["wrong_step"]

    try:
        combined_ids, span_map, ws_start = get_token_spans(
            tokenizer, problem, prefix_steps, prefix_text, prefix_len, wrong_step
        )
        if not span_map:
            return {"attention": {"root_cause_source": None, "error": "span_map_empty"}}

        attn              = run_attention_forward_pass(model, tokenizer, combined_ids)
        scores_by_variant = aggregate_attention_scores(attn, span_map, ws_start)
        root_cause_source = max(scores_by_variant["mean_all"], key=scores_by_variant["mean_all"].get)

        layer_sensitivity = {
            "last_layer_top":   max(scores_by_variant["last_layer"],   key=scores_by_variant["last_layer"].get),
            "middle_layer_top": max(scores_by_variant["middle_layer"], key=scores_by_variant["middle_layer"].get),
            "mean_all_top":     root_cause_source,
        }
        layer_sensitivity["all_layers_agree"] = (
            layer_sensitivity["last_layer_top"] ==
            layer_sensitivity["middle_layer_top"] ==
            layer_sensitivity["mean_all_top"]
        )

        head_agree      = compute_head_agreement(attn, span_map, ws_start, root_cause_source)
        token_breakdown = compute_token_level_breakdown(attn, span_map, ws_start, combined_ids, tokenizer)

        del attn
        torch.cuda.empty_cache()

        return {
            "attention": {
                "root_cause_source":          root_cause_source,
                "attention_scores_by_source": scores_by_variant,
                "mean_all_scores":            scores_by_variant["mean_all"],
                "layer_sensitivity":          layer_sensitivity,
                "head_agreement":             head_agree,
                "token_level_breakdown":      token_breakdown,
                "all_layers_agree":           layer_sensitivity["all_layers_agree"],
                "error":                      None,
            }
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"attention": {"root_cause_source": None, "error": "OOM"}}
    except Exception as exc:
        return {"attention": {"root_cause_source": None, "error": str(exc)}}


# ═══════════════════════════════════════════════════════════════════════════════
# Majority Vote
# ═══════════════════════════════════════════════════════════════════════════════

def compute_majority_vote(attributions: dict) -> dict:
    """Majority vote across available strategies (threshold: >50% of active votes)."""
    votes = {}
    for strategy, attr in attributions.items():
        if attr is None:
            continue
        # identifier_tracing abstains when applicable=False
        if strategy == "identifier_tracing" and not attr.get("applicable", True):
            continue
        source = attr.get("root_cause_source")
        if source is not None:
            votes[strategy] = source

    if not votes:
        return {"source": None, "agreement_count": 0, "total_votes": 0,
                "is_unanimous": False, "is_ambiguous": True, "vote_breakdown": {}}

    counts    = Counter(votes.values())
    best, cnt = counts.most_common(1)[0]
    total     = len(votes)
    return {
        "source":          best if cnt > total / 2 else None,
        "agreement_count": cnt,
        "total_votes":     total,
        "is_unanimous":    cnt == total,
        "is_ambiguous":    cnt <= total / 2,
        "vote_breakdown":  dict(counts),
    }


# ─── Output record builders ───────────────────────────────────────────────────

def build_comparison_record(record: dict, attributions: dict, majority: dict) -> dict:
    j   = attributions.get("llm_judge") or {}
    co  = attributions.get("counterfactual_omission")
    it  = attributions.get("identifier_tracing") or {}
    att = attributions.get("attention") or {}
    return {
        **_base(record),
        "gold_reasoning": record.get("gold_reasoning", ""),
        "attributions": {
            "llm_judge": {
                "root_cause_source":      j.get("root_cause_source"),
                "confidence_margin":      j.get("confidence_margin"),
                "self_consistent":        j.get("self_consistent"),
                "run_1_source":           j.get("run_1_source"),
                "run_2_source":           j.get("run_2_source"),
                "error_type":             j.get("error_type"),
            } if j else None,
            "counterfactual_omission": {
                "root_cause_source":   co.get("root_cause_source") if co else None,
                "omission_pass_rates": co.get("omission_pass_rates") if co else None,
                "best_improvement":    co.get("best_improvement") if co else None,
            } if co else None,
            "identifier_tracing": {
                "root_cause_source": it.get("root_cause_source"),
                "applicable":        it.get("applicable"),
                "traced_tokens":     it.get("traced_tokens"),
            } if it else None,
            "attention": {
                "root_cause_source": att.get("root_cause_source"),
                "mean_all_scores":   att.get("mean_all_scores"),
                "head_agreement":    att.get("head_agreement"),
                "all_layers_agree":  att.get("all_layers_agree"),
                "error":             att.get("error"),
            } if att else None,
        },
        "majority_vote": majority,
        "processed_at":  datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset reload helper (for Strategy 2 — needs public_tests)
# ═══════════════════════════════════════════════════════════════════════════════

_filtered_dataset = None

def get_filtered_dataset():
    """Load and filter CodeContests once; cache in module-level variable."""
    global _filtered_dataset
    if _filtered_dataset is not None:
        return _filtered_dataset
    from datasets import load_dataset
    print("Loading CodeContests dataset for omission strategy...")
    ds = load_dataset("deepmind/code_contests", split="train")
    POOL_SIZE = min(cdg.NUM_DATASET_SAMPLES * 3, len(ds))
    ds = ds.select(range(POOL_SIZE))
    EASY_DIFFICULTIES = {7, 8}
    ds = ds.filter(lambda x: (
        x["difficulty"] in EASY_DIFFICULTIES and
        len(x["public_tests"]["input"]) > 0 and
        len(x["solutions"]["solution"]) > 0
    ))
    print(f"Filtered dataset size: {len(ds)}")
    _filtered_dataset = ds
    return ds


def load_public_tests(dataset_index: int) -> Optional[dict]:
    """Retrieve public_tests for a record by its dataset_index."""
    try:
        ds = get_filtered_dataset()
        if dataset_index >= len(ds):
            return None
        return ds[dataset_index]["public_tests"]
    except Exception as e:
        print(f"    [omission] Could not load public tests for idx={dataset_index}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Root cause attribution for code reasoning steps")
    parser.add_argument("--run-omission", action="store_true",
                        help="Also run Strategy 2 (counterfactual omission; expensive)")
    parser.add_argument("--judge-only",   action="store_true",
                        help="Run only Strategy 1 (LLM Judge). Fastest — single GPU.")
    parser.add_argument("--input",  default=INPUT_FILE, help="Input JSONL path")
    parser.add_argument("--max",    type=int, default=MAX_ATTRIBUTION_SAMPLES,
                        help="Max samples to process (default: 50)")
    parser.add_argument("--output-comparison", default=OUTPUT_COMPARISON)
    parser.add_argument("--output-judge",      default=OUTPUT_JUDGE)
    args = parser.parse_args()

    gen_model, gen_tokenizer, judge_model, judge_tokenizer = load_models(
        judge_only=args.judge_only
    )
    patch_module_globals(gen_model, gen_tokenizer, judge_model, judge_tokenizer)

    all_records   = load_jsonl(args.input)
    wrong_records = [r for r in all_records if r.get("wrong_step_found") and r.get("wrong_step")]
    print(f"Found {len(wrong_records)} confirmed wrong-step records in {args.input}")

    processed  = load_processed_indices(args.output_comparison)
    to_process = [r for r in wrong_records if r["dataset_index"] not in processed]
    to_process = to_process[:max(0, args.max - len(processed))]
    print(f"Already processed: {len(processed)}  |  Will process now: {len(to_process)}")

    for i, record in enumerate(to_process):
        idx = record["dataset_index"]
        print(f"\n[{i+1}/{len(to_process)}] idx={idx}  problem={record.get('problem_name','')}  "
              f"prefix_len={record['prefix_len']}")

        attributions = {}

        # ── Strategy 1: LLM Judge ────────────────────────────────────────────
        print("  S1 LLM Judge...")
        j = attribute_llm_judge(record, judge_model, judge_tokenizer)
        attributions.update(j)
        jj = j["llm_judge"]
        print(f"    => source={jj['root_cause_source']}  consistent={jj['self_consistent']}  "
              f"conf={jj['confidence_margin']:.3f}  error_type={jj['error_type']}")

        # ── Strategy 3: Identifier Tracing ───────────────────────────────────
        if not args.judge_only:
            it = attribute_identifier_tracing(record)
            attributions.update(it)
            print(f"  S3 Identifier Trace => {it['identifier_tracing']['root_cause_source']}  "
                  f"(applicable={it['identifier_tracing']['applicable']})")
        else:
            attributions["identifier_tracing"] = None

        # ── Strategy 4: Attention ────────────────────────────────────────────
        if not args.judge_only:
            print("  S4 Attention...")
            att = attribute_attention(record, gen_model, gen_tokenizer)
            attributions.update(att)
            # S4 diagnostic: optional string, see module docstring (not Stage 1 / not API "content")
            err = att["attention"].get("error")
            if err:
                print(f"    => error: {err}")
            else:
                print(f"    => source={att['attention']['root_cause_source']}  "
                      f"head_agree={att['attention'].get('head_agreement')}  "
                      f"layers_agree={att['attention'].get('all_layers_agree')}")
        else:
            attributions["attention"] = None

        # ── Strategy 2: Counterfactual Omission (optional) ───────────────────
        if args.run_omission:
            print("  S2 Counterfactual Omission...")
            public_tests = load_public_tests(idx)
            if public_tests:
                om = attribute_counterfactual_omission(record, public_tests)
                attributions.update(om)
                print(f"    => {om['counterfactual_omission']['root_cause_source']}  "
                      f"best_improvement={om['counterfactual_omission']['best_improvement']:.3f}")
            else:
                print("    => skipped (public tests unavailable)")
                attributions["counterfactual_omission"] = None
        else:
            attributions["counterfactual_omission"] = None

        majority = compute_majority_vote(attributions)
        print(f"  Majority => {majority['source']}  "
              f"({majority['agreement_count']}/{majority['total_votes']}  "
              f"ambiguous={majority['is_ambiguous']})")

        # ── Write outputs ─────────────────────────────────────────────────────
        comp_rec = build_comparison_record(record, attributions, majority)
        append_jsonl(args.output_comparison, comp_rec)

        append_jsonl(args.output_judge, {
            **_base(record),
            "llm_judge":          attributions.get("llm_judge"),
            "attribution_method": "llm_judge",
        })
        if not args.judge_only:
            append_jsonl(OUTPUT_ID_TRACE, {
                **_base(record),
                "identifier_tracing": attributions.get("identifier_tracing"),
                "attribution_method": "identifier_tracing",
            })
            append_jsonl(OUTPUT_ATTENTION, {
                **_base(record),
                "attention":          attributions.get("attention"),
                "attribution_method": "attention",
            })
        if args.run_omission and attributions.get("counterfactual_omission"):
            append_jsonl(OUTPUT_OMISSION, {
                **_base(record),
                "counterfactual_omission": attributions["counterfactual_omission"],
                "attribution_method":      "counterfactual_omission",
            })

    print(f"\nDone. Processed {len(to_process)} records.")
    print(f"  Comparison:         {args.output_comparison}")
    print(f"  Judge:              {args.output_judge}")
    if not args.judge_only:
        print(f"  Identifier Trace:   {OUTPUT_ID_TRACE}")
        print(f"  Attention:          {OUTPUT_ATTENTION}")
    if args.run_omission:
        print(f"  Omission:           {OUTPUT_OMISSION}")


if __name__ == "__main__":
    main()

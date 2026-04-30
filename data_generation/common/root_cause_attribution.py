import os
import re
import json
import random
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Optional

                                                                                
os.environ["ATTRIBUTION_IMPORT"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import self_correction_data_gen as sdg
from self_correction_data_gen import (
    normalize_answer_string,
    score_continuation,
    batched_generate,
    sample_wrong_next_step,
    GENERATOR_MODEL_NAME,
    JUDGE_MODEL_NAME,
    SYSTEM_PROMPT,
    STEP_MAX_NEW_TOKENS,
    ROLLOUT_MAX_NEW_TOKENS,
    STEP_TEMPERATURE,
    ROLLOUT_TEMPERATURE,
    TOP_P,
)

                                                                                

INPUT_FILE               = "wrong_steps_data_prev.jsonl"
OUTPUT_COMPARISON        = "attribution_comparison.jsonl"
OUTPUT_JUDGE             = "strategy_1_llm_judge_samples.jsonl"
OUTPUT_OMISSION          = "strategy_2_omission_samples.jsonl"
OUTPUT_VALUE_TRACE       = "strategy_3_value_trace_samples.jsonl"
OUTPUT_ATTENTION         = "strategy_4_attention_samples.jsonl"

MAX_ATTRIBUTION_SAMPLES  = 50
OMISSION_CANDIDATES      = 4                                              
OMISSION_ROLLOUTS        = 4                                     

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

                                                                    
_NUMBER_RE      = re.compile(r'(?<!\w)[-+]?\d+(?:\.\d+)?(?:/\d+)?(?!\w)')
_STEP_LABEL_RE  = re.compile(r'^Step\s*\d+[:.]\s*', re.IGNORECASE)

                                                                                

def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r") as f:
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
    """Split prefix_text into individual non-empty step strings."""
    return [line.strip() for line in prefix_text.strip().split("\n") if line.strip()]

                                                                                

def load_models(judge_only=False):
    """Load generator on cuda:0 and judge on cuda:1 (or auto if single GPU).
    If judge_only=True, skip loading the generator model (saves memory/time).
    """
    if judge_only:
        print("Judge-only mode: skipping generator model load.")
        model, tokenizer = None, None
    else:
        print("Loading generator model...")
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            attn_implementation="eager",                                                     
        )
        print(f"Generator on: {next(model.parameters()).device}")

    print("Loading judge model...")
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    if not judge_only and torch.cuda.device_count() >= 2:
        judge_device_map = "cuda:1"
        print("Judge on: cuda:1")
    else:
        judge_device_map = "auto"
        print(f"Judge using device_map='auto'")

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=judge_device_map,
    )
    print("Models loaded.")
    return model, tokenizer, judge_model, judge_tokenizer

def patch_module_globals(model, tokenizer, judge_model, judge_tokenizer):
    """
    Monkey-patch self_correction_data_gen module globals so that
    sample_wrong_next_step (used in Strategy 2) works correctly.
    """
    sdg.model          = model
    sdg.tokenizer      = tokenizer
    sdg.judge_model    = judge_model
    sdg.judge_tokenizer = judge_tokenizer

                                                                                 
                                    
                                                                                 

def build_judge_attribution_prompt(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    seed: int = 42,
) -> tuple:
    """
    Build a forced-choice attribution prompt.

    Options (shuffled with `seed`):
      - One per prefix step: "Model misused a value or result from Step i"
      - "question":     "Model misread or misinterpreted the original question"
      - "independent":  "Model made an independent arithmetic/algebraic mistake"

    Returns:
      prompt_str  (str)
      option_map  (dict): letter -> (source_key, option_text)
    """
    rng = random.Random(seed)

    step_label_re_local = re.compile(r'^Step\s*\d+[:.]*\s*')
    options = []
    for i in range(len(prefix_steps)):
        content = step_label_re_local.sub('', prefix_steps[i]).strip()[:100]
        options.append((
            f"step_{i+1}",
            f"The model misused a value or result from Step {i+1} (\"{content}\").",
        ))
    options.append(("question",     "The model misread or misinterpreted the original question."))
    options.append(("independent",  "The model made an independent arithmetic or algebraic mistake not caused by any prior step."))
    rng.shuffle(options)

    option_map  = {}
    option_lines = []
    for idx, (source_key, text) in enumerate(options):
        letter = LETTERS[idx]
        option_map[letter] = (source_key, text)
        option_lines.append(f"{letter}. {text}")

    step_label_re = re.compile(r'^Step\s*\d+[:.]*\s*')
    prefix_section = "\n".join(
        f"Step {i+1}: {step_label_re.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )
    ws_content = re.sub(r'^Step\s*\d+[:.]*\s*', '', wrong_step).strip()
    ws_num     = len(prefix_steps) + 1

    prompt = (
        "You are analyzing a math reasoning error.\n\n"
        f"Problem:\n{question}\n\n"
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
    question: str,
    prefix_steps: list,
    wrong_step: str,
    judge_model,
    judge_tokenizer,
    seed: int = 42,
) -> dict:
    prompt, option_map = build_judge_attribution_prompt(
        question, prefix_steps, wrong_step, seed=seed
    )

                                                                              
    device = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in option_map:
        token_id = judge_tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
        scores[letter] = log_probs[token_id].item()

    best_letter    = max(scores, key=scores.get)
    sorted_vals    = sorted(scores.values(), reverse=True)
    conf_margin    = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    source, expl   = option_map[best_letter]

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
    question: str,
    prefix_steps: list,
    wrong_step: str,
    judge_model,
    judge_tokenizer,
) -> str:
    """
    Classify the wrong step into one of four broad categories based on
    how the error relates to the surrounding context.

    Categories:
      mechanical   — pure computation error (arithmetic, algebra, sign) with no
                     misreading of context; the model identified the right things
                     to compute but computed them wrongly
      setup        — wrong equation, formula, or problem structure; the model
                     misunderstood what computation to perform
      substitution — right approach but wrong value or variable used; the model
                     pulled the wrong input from somewhere
      reasoning    — wrong logical inference or interpretation; the model's
                     chain of reasoning is flawed regardless of the numbers
    """
    step_label_re = re.compile(r'^Step\s*\d+[:.]*\s*')
    prefix_section = "\n".join(
        f"Step {i+1}: {step_label_re.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )
    ws_content = step_label_re.sub('', wrong_step).strip()
    ws_num     = len(prefix_steps) + 1

    options = [
        ("mechanical",   "A. Mechanical error — the model identified the right things to compute but made a pure arithmetic, algebraic, or sign mistake."),
        ("setup",        "B. Setup error — the model used the wrong equation, formula, or problem structure for this step."),
        ("substitution", "C. Substitution error — the model used the right approach but plugged in the wrong value, variable, or quantity."),
        ("reasoning",    "D. Reasoning error — the model made a flawed logical inference or misinterpreted what the step should conclude."),
    ]

    option_lines = "\n".join(text for _, text in options)
    letter_to_type = {chr(65 + i): etype for i, (etype, _) in enumerate(options)}

    prompt = (
        "You are classifying a math reasoning error.\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct reasoning prefix:\n{prefix_section}\n\n"
        f"Wrong step generated by the model:\n"
        f"Step {ws_num}: {ws_content}\n\n"
        "Classify the nature of the error. Pick exactly one:\n\n"
        f"{option_lines}\n\n"
        "Answer with a single letter (A–D):"
    )

    device = next(judge_model.parameters()).device
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
    question     = record["question"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    wrong_step   = record["wrong_step"]

    run_1 = run_judge_attribution(question, prefix_steps, wrong_step, judge_model, judge_tokenizer, seed=42)
    run_2 = run_judge_attribution(question, prefix_steps, wrong_step, judge_model, judge_tokenizer, seed=137)

    return {
        "llm_judge": {
            "root_cause_source":      run_1["root_cause_source"],
            "root_cause_explanation": run_1["root_cause_explanation"],
            "confidence_margin":      run_1["confidence_margin"],
            "self_consistent":        run_1["root_cause_source"] == run_2["root_cause_source"],
            "run_1_source":           run_1["root_cause_source"],
            "run_2_source":           run_2["root_cause_source"],
            "run_1":                  run_1,
            "run_2":                  run_2,
        }
    }

                                                                                 
                                      
                                                                                 

def build_reduced_prefix(prefix_steps: list, omit_index: int) -> tuple:
    """
    Remove step at `omit_index` (0-based) and renumber remaining steps.
    Returns (reduced_prefix_text, reduced_prefix_len).
    Edge case: if prefix has only one step, returns ("", 0).
    """
    remaining = [s for i, s in enumerate(prefix_steps) if i != omit_index]
    if not remaining:
        return "", 0
    cleaned      = [re.sub(r'^Step\s*\d+[:.]*\s*', '', s).strip() for s in remaining]
    reduced_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(cleaned))
    return reduced_text, len(remaining)

def run_omission_trial(
    question: str,
    reduced_prefix_text: str,
    reduced_prefix_len: int,
    gold_text: str,
    gold_answer: str,
    n_candidates: int = OMISSION_CANDIDATES,
    n_rollouts: int   = OMISSION_ROLLOUTS,
) -> dict:
    """
    Sample `n_candidates` next steps from the reduced prefix.
    For each, run `n_rollouts` to check correctness.
    Returns mean_correct_rate and condensed candidate details.
    """
    tried          = set()
    candidates_out = []
    total_correct  = 0
    total_eval     = 0

    for _ in range(n_candidates):
        result = sample_wrong_next_step(
            question=question,
            prefix_text=reduced_prefix_text,
            prefix_len=reduced_prefix_len,
            gold_text=gold_text,
            gold_answer=gold_answer,
            num_rollouts=n_rollouts,
        )
        step = result["candidate_step"]
        if step in tried:
            continue
        tried.add(step)

        correct   = sum(1 for r in result["rollouts"] if r["is_correct"] is True)
        evaluated = sum(1 for r in result["rollouts"] if r["is_correct"] is not None)
        total_correct += correct
        total_eval    += evaluated
        candidates_out.append({
            "candidate_step":   step,
            "correct_rollouts": correct,
            "total_evaluated":  evaluated,
        })

    mean_correct_rate = total_correct / total_eval if total_eval > 0 else 0.0
    return {
        "mean_correct_rate": round(mean_correct_rate, 4),
        "candidates":        candidates_out,
    }

def attribute_counterfactual_omission(record: dict) -> dict:
    question     = record["question"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    prefix_len   = record["prefix_len"]
    gold_text    = record["gold_solution"]
    gold_answer  = record["gold_answer"]

    omission_results = {}
    for i in range(prefix_len):
        source_key    = f"step_{i+1}"
        reduced_text, reduced_len = build_reduced_prefix(prefix_steps, omit_index=i)
        print(f"    [omission] omitting step {i+1} -> reduced_len={reduced_len}")
        omission_results[source_key] = run_omission_trial(
            question=question,
            reduced_prefix_text=reduced_text,
            reduced_prefix_len=reduced_len,
            gold_text=gold_text,
            gold_answer=gold_answer,
        )

                                                                                       
    best_source = None
    best_rate   = 0.0
    for source_key, trial in omission_results.items():
        if trial["mean_correct_rate"] > best_rate:
            best_rate   = trial["mean_correct_rate"]
            best_source = source_key

    root_cause_source = best_source if (best_source and best_rate > 0.0) else "independent"

    return {
        "counterfactual_omission": {
            "root_cause_source":      root_cause_source,
            "omission_correct_rates": {k: v["mean_correct_rate"] for k, v in omission_results.items()},
            "best_improvement":       round(best_rate, 4),
            "omission_details":       omission_results,
        }
    }

                                                                                 
                                              
                                                                                 

def extract_numbers_from_text(text: str) -> list:
    """
    Extract numeric tokens from `text`, excluding the step-label number.
    Returns a deduplicated list of string values.
    """
    clean = _STEP_LABEL_RE.sub('', text)
    clean = normalize_answer_string(clean)
    found = _NUMBER_RE.findall(clean)
    seen, result = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result

def _text_contains_number(text: str, value: str) -> bool:
    norm = normalize_answer_string(_STEP_LABEL_RE.sub('', text))
    return bool(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', norm))

def find_value_source(value: str, prefix_steps: list, question: str) -> Optional[str]:
    """
    Search most-recent prefix step first, then question.
    Returns 'step_i' (1-indexed), 'question', or None.
    """
    for i in range(len(prefix_steps) - 1, -1, -1):
        if _text_contains_number(prefix_steps[i], value):
            return f"step_{i+1}"
    if _text_contains_number(question, value):
        return "question"
    return None

def attribute_value_tracing(record: dict) -> dict:
    wrong_step   = record["wrong_step"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    question     = record["question"]

    numbers = extract_numbers_from_text(wrong_step)
    if not numbers:
        return {"value_tracing": {"root_cause_source": None, "applicable": False, "traced_values": []}}

    traced        = []
    source_counts = Counter()
    for value in numbers:
        found_in = find_value_source(value, prefix_steps, question)
        also_in  = []
        if found_in:
            if found_in.startswith("step_"):
                idx = int(found_in.split("_")[1]) - 1
                if _text_contains_number(question, value):
                    also_in.append("question")
                for j in range(len(prefix_steps)):
                    if j != idx and _text_contains_number(prefix_steps[j], value):
                        also_in.append(f"step_{j+1}")
            source_counts[found_in] += 1
        traced.append({"value": value, "found_in": found_in, "also_in": also_in})

    if not source_counts:
        root_cause_source = "independent"
    else:
                                                                         
        def sort_key(k):
            count = source_counts[k]
            recency = int(k.split("_")[1]) if k.startswith("step_") else 0
            return (count, recency)
        root_cause_source = max(source_counts, key=sort_key)

    return {
        "value_tracing": {
            "root_cause_source": root_cause_source,
            "applicable":        True,
            "traced_values":     traced,
        }
    }

                                                                                 
                                    
                                                                                 

def _build_step_prompt_ids(
    question: str,
    prefix_text: str,
    prefix_len: int,
    tokenizer,
) -> list:
    """
    Reconstruct the exact token ids used as the step-generation prompt
    (same template as sample_wrong_next_step).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "You are given a math problem and a correct reasoning prefix extracted from a gold solution.\n\n"
            "Your task is to generate the next reasoning step after the prefix.\n\n"
            "Rules:\n"
            "1. Output exactly one line.\n"
            f"2. Use the exact format:\nStep {prefix_len + 1}: ...\n"
            "3. Do not generate any further steps.\n"
            "4. Do not generate the final answer.\n"
            "5. Do not generate any extra text.\n\n"
            f"Problem:\n{question}\n\n"
            f"Correct reasoning prefix:\n{prefix_text}"
        )},
    ]
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
    """
    Find the token index range [start, end) for `substring` within the
    decoded token sequence. Uses decoded character positions and re-tokenization.
    Returns (start_token_idx, end_token_idx) or None if not found.
    """
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    char_idx  = full_text.find(substring)
    if char_idx == -1:
        return None

    text_before = full_text[:char_idx]
    text_through = full_text[:char_idx + len(substring)]
    start = len(tokenizer(text_before, add_special_tokens=False).input_ids)
    end   = len(tokenizer(text_through, add_special_tokens=False).input_ids)
    if end <= start:
        end = start + 1
    return (start, min(end, len(full_ids)))

def get_token_spans(
    tokenizer,
    question: str,
    prefix_steps: list,
    prefix_text: str,
    prefix_len: int,
    wrong_step: str,
) -> tuple:
    """
    Build combined_ids = prompt_ids + wrong_step_ids.
    Return:
      combined_ids       (list of int)
      span_map           ({source_key: (start, end)})
      wrong_step_start   (int)
    """
    prompt_ids       = _build_step_prompt_ids(question, prefix_text, prefix_len, tokenizer)
    ws_ids           = tokenizer(wrong_step, add_special_tokens=False).input_ids
    if not isinstance(ws_ids, list):
        ws_ids = ws_ids.tolist() if hasattr(ws_ids, 'tolist') else list(ws_ids)
    wrong_step_start = len(prompt_ids)
    combined_ids     = prompt_ids + ws_ids

    span_map = {}

                   
    q_span = _find_token_span(tokenizer, combined_ids, question)
    if q_span:
        span_map["question"] = q_span

                           
    for i, step in enumerate(prefix_steps):
        content   = re.sub(r'^Step\s*\d+[:.]*\s*', '', step).strip()
        formatted = f"Step {i+1}: {content}"
        span = _find_token_span(tokenizer, combined_ids, formatted)
        if span is None:
                                                                  
            span = _find_token_span(tokenizer, combined_ids, content[:40])
        if span:
            span_map[f"step_{i+1}"] = span

    return combined_ids, span_map, wrong_step_start

def run_attention_forward_pass(model, tokenizer, combined_ids: list) -> torch.Tensor:
    """
    Forward pass with output_attentions=True.
    Returns stacked attention: (num_layers, num_heads, seq_len, seq_len) on CPU.
    """
    device = next(model.parameters()).device
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
    """
    Mean attention from wrong_step tokens (as queries) to each source span (keys).
    Three variants: last_layer, middle_layer, mean_all_layers.
    Returns {variant: {source_key: float}}.
    """
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
    """
    Fraction of (layer, head) pairs whose argmax source matches `top_source`.
    """
    num_layers, num_heads, seq_len, _ = attn.shape
    ws_attn = attn[:, :, wrong_step_start:seq_len, :]                     
    agree = 0
    total = num_layers * num_heads

    for l_idx in range(num_layers):
        for h_idx in range(num_heads):
            head_q = ws_attn[l_idx, h_idx]               
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
    """
    For each wrong_step token, which source does it attend to most?
    Uses mean over all layers and heads.
    """
    seq_len  = attn.shape[2]
    ws_ids   = combined_ids[wrong_step_start:seq_len]
    mean_attn = attn.mean(0).mean(0)[wrong_step_start:seq_len, :]               

    breakdown = []
    for t_idx in range(mean_attn.shape[0]):
        tok_attn     = mean_attn[t_idx]
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
    question     = record["question"]
    prefix_steps = parse_prefix_steps(record["prefix_text"])
    prefix_text  = record["prefix_text"]
    prefix_len   = record["prefix_len"]
    wrong_step   = record["wrong_step"]

    try:
        combined_ids, span_map, ws_start = get_token_spans(
            tokenizer, question, prefix_steps, prefix_text, prefix_len, wrong_step
        )
        if not span_map:
            return {"attention": {"root_cause_source": None, "error": "span_map_empty"}}

        attn               = run_attention_forward_pass(model, tokenizer, combined_ids)
        scores_by_variant  = aggregate_attention_scores(attn, span_map, ws_start)
        root_cause_source  = max(scores_by_variant["mean_all"], key=scores_by_variant["mean_all"].get)

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

        head_agree       = compute_head_agreement(attn, span_map, ws_start, root_cause_source)
        token_breakdown  = compute_token_level_breakdown(attn, span_map, ws_start, combined_ids, tokenizer)

        del attn
        torch.cuda.empty_cache()

        return {
            "attention": {
                "root_cause_source":       root_cause_source,
                "attention_scores_by_source": scores_by_variant,
                "mean_all_scores":         scores_by_variant["mean_all"],
                "layer_sensitivity":       layer_sensitivity,
                "head_agreement":          head_agree,
                "token_level_breakdown":   token_breakdown,
                "all_layers_agree":        layer_sensitivity["all_layers_agree"],
                "error":                   None,
            }
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"attention": {"root_cause_source": None, "error": "OOM"}}
    except Exception as exc:
        return {"attention": {"root_cause_source": None, "error": str(exc)}}

                                                                                 
               
                                                                                 

def compute_majority_vote(attributions: dict) -> dict:
    """
    Majority vote across all four strategies (threshold: ≥3 agree).
    Value tracing with applicable=False is excluded (abstain).
    """
    votes = {}
    for strategy, attr in attributions.items():
        if attr is None:
            continue
        if strategy == "value_tracing" and not attr.get("applicable", True):
            continue
        source = attr.get("root_cause_source")
        if source is not None:
            votes[strategy] = source

    if not votes:
        return {"source": None, "agreement_count": 0, "total_votes": 0,
                "is_unanimous": False, "is_ambiguous": True, "vote_breakdown": {}}

    counts      = Counter(votes.values())
    best, cnt   = counts.most_common(1)[0]
    total       = len(votes)
                                                                     
    majority_threshold = total / 2
    return {
        "source":           best if cnt > majority_threshold else None,
        "agreement_count":  cnt,
        "total_votes":      total,
        "is_unanimous":     cnt == total,
        "is_ambiguous":     cnt <= majority_threshold,
        "vote_breakdown":   dict(counts),
    }

                                                                                

def _base(record: dict) -> dict:
    return {
        "dataset_index": record["dataset_index"],
        "question":      record["question"],
        "prefix_text":   record["prefix_text"],
        "prefix_len":    record["prefix_len"],
        "wrong_step":    record["wrong_step"],
    }

def build_comparison_record(record: dict, attributions: dict, majority: dict) -> dict:
    j   = attributions.get("llm_judge") or {}
    co  = attributions.get("counterfactual_omission")
    vt  = attributions.get("value_tracing") or {}
    att = attributions.get("attention") or {}
    return {
        **_base(record),
        "gold_solution": record["gold_solution"],
        "gold_answer":   record["gold_answer"],
        "attributions": {
            "llm_judge": {
                "root_cause_source":      j.get("root_cause_source"),
                "confidence_margin":      j.get("confidence_margin"),
                "self_consistent":        j.get("self_consistent"),
                "run_1_source":           j.get("run_1_source"),
                "run_2_source":           j.get("run_2_source"),
            },
            "counterfactual_omission": {
                "root_cause_source":      co.get("root_cause_source") if co else None,
                "omission_correct_rates": co.get("omission_correct_rates") if co else None,
                "best_improvement":       co.get("best_improvement") if co else None,
            } if co else None,
            "value_tracing": {
                "root_cause_source": vt.get("root_cause_source"),
                "applicable":        vt.get("applicable"),
                "traced_values":     vt.get("traced_values"),
            } if vt else None,
            "attention": {
                "root_cause_source": att.get("root_cause_source"),
                "mean_all_scores":   att.get("mean_all_scores"),
                "head_agreement":    att.get("head_agreement"),
                "all_layers_agree":  att.get("all_layers_agree"),
                "error":             att.get("error"),
            } if att else None,
        },
        "majority_vote":  majority,
        "processed_at":   datetime.utcnow().isoformat(),
    }

                                                                                 
      
                                                                                 

def main():
    parser = argparse.ArgumentParser(description="Root cause attribution for wrong reasoning steps")
    parser.add_argument("--run-omission", action="store_true",
                        help="Also run the (expensive) counterfactual omission strategy")
    parser.add_argument("--judge-only", action="store_true",
                        help="Run only Strategy 1 (LLM Judge). Skips generator load, strategies 3 & 4.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSONL path")
    parser.add_argument("--max",   type=int, default=MAX_ATTRIBUTION_SAMPLES,
                        help="Max samples to process (default: 50)")
    parser.add_argument("--output-judge", default=OUTPUT_JUDGE, help="Output path for judge results")
    parser.add_argument("--output-comparison", default=OUTPUT_COMPARISON, help="Output path for comparison file (used for resume tracking)")
    args = parser.parse_args()

    model, tokenizer, judge_model, judge_tokenizer = load_models(judge_only=args.judge_only)
    patch_module_globals(model, tokenizer, judge_model, judge_tokenizer)

    all_records   = load_jsonl(args.input)
    wrong_records = [r for r in all_records if r.get("wrong_step_found") and r.get("wrong_step")]
    print(f"Found {len(wrong_records)} confirmed wrong-step records in {args.input}")

    processed  = load_processed_indices(args.output_comparison)
    to_process = [r for r in wrong_records if r["dataset_index"] not in processed]
    to_process = to_process[:max(0, args.max - len(processed))]
    print(f"Already processed: {len(processed)}  |  Will process now: {len(to_process)}")

    for i, record in enumerate(to_process):
        idx = record["dataset_index"]
        print(f"\n[{i+1}/{len(to_process)}] dataset_index={idx}  prefix_len={record['prefix_len']}")

        attributions = {}

                                
        print("  S1 LLM Judge...")
        j = attribute_llm_judge(record, judge_model, judge_tokenizer)
        attributions.update(j)
        jj = j['llm_judge']
        print(f"    => source={jj['root_cause_source']}  "
              f"consistent={jj['self_consistent']}  conf={jj['confidence_margin']:.3f}")

                                                     
        if not args.judge_only:
            vt = attribute_value_tracing(record)
            attributions.update(vt)
            print(f"  S3 Value Trace => {vt['value_tracing']['root_cause_source']}  "
                  f"(applicable={vt['value_tracing']['applicable']})")
        else:
            attributions["value_tracing"] = None

                                
        if not args.judge_only:
            print("  S4 Attention...")
            att = attribute_attention(record, model, tokenizer)
            attributions.update(att)
            err = att["attention"].get("error")
            if err:
                print(f"    => error: {err}")
            else:
                print(f"    => {att['attention']['root_cause_source']}  "
                      f"(head_agree={att['attention'].get('head_agreement')}  "
                      f"layers_agree={att['attention'].get('all_layers_agree')})")
        else:
            attributions["attention"] = None

                                                         
        if args.run_omission:
            print("  S2 Counterfactual Omission...")
            om = attribute_counterfactual_omission(record)
            attributions.update(om)
            print(f"    => {om['counterfactual_omission']['root_cause_source']}  "
                  f"(best_improvement={om['counterfactual_omission']['best_improvement']:.3f})")
        else:
            attributions["counterfactual_omission"] = None

        majority = compute_majority_vote(attributions)
        print(f"  Majority => {majority['source']}  "
              f"(count={majority['agreement_count']}/{majority['total_votes']}  "
              f"ambiguous={majority['is_ambiguous']})")

                                
        comp_rec = build_comparison_record(record, attributions, majority)
        append_jsonl(args.output_comparison, comp_rec)

        append_jsonl(args.output_judge, {
            **_base(record), "llm_judge": attributions["llm_judge"],
            "attribution_method": "llm_judge",
        })
        if not args.judge_only:
            append_jsonl(OUTPUT_VALUE_TRACE, {
                **_base(record), "value_tracing": attributions["value_tracing"],
                "attribution_method": "value_tracing",
            })
            append_jsonl(OUTPUT_ATTENTION, {
                **_base(record), "attention": attributions["attention"],
                "attribution_method": "attention",
            })
        if args.run_omission and attributions.get("counterfactual_omission"):
            append_jsonl(OUTPUT_OMISSION, {
                **_base(record),
                "counterfactual_omission": attributions["counterfactual_omission"],
                "attribution_method": "counterfactual_omission",
            })

    print(f"\nDone. Processed {len(to_process)} records.")
    print(f"  Comparison:   {args.output_comparison}")
    print(f"  Judge:        {OUTPUT_JUDGE}")
    print(f"  Value Trace:  {OUTPUT_VALUE_TRACE}")
    print(f"  Attention:    {OUTPUT_ATTENTION}")
    if args.run_omission:
        print(f"  Omission:     {OUTPUT_OMISSION}")

if __name__ == "__main__":
    main()

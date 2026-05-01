import os
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

                                                                                

JUDGE_MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
INPUT_FILE        = "wrong_steps_50k.jsonl"
OUTPUT_FILE       = "error_trace.jsonl"
MAX_SAMPLES       = 50
NUM_ROLLOUTS      = 3                                              

LETTERS           = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_STEP_LABEL_RE    = re.compile(r'^Step\s*\d+[:.]\s*', re.IGNORECASE)
_STEP_NUM_RE      = re.compile(r'^Step\s*(\d+)[:.]\s*', re.IGNORECASE)

LABEL_MAP = {
    "propagated": (
        "Propagated — this step makes no new mistake. It correctly uses the wrong value AND "
        "correctly accounts for all results from prior steps (gold prefix steps, the wrong step, "
        "and any previous downstream steps)."
    ),
    "new_error": (
        "New error — this step introduces a fresh mistake. This includes: "
        "(1) arithmetic or logical errors; "
        "(2) omitting or forgetting a value computed in any prior step "
        "(gold prefix steps, the wrong step, or previous downstream steps); "
        "(3) misusing a value from any prior step "
        "(gold prefix steps, the wrong step, or previous downstream steps)."
    ),
}

                                                                                

def load_judge():
    print(f"Loading judge: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Judge loaded.\n")
    return model, tokenizer

                                                                                

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_processed_indices(output_file: str) -> set:
    if not Path(output_file).exists():
        return set()
    return {rec["dataset_index"] for rec in load_jsonl(output_file)}

def parse_prefix_steps(prefix_text: str) -> list:
    return [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]

def parse_rollout_steps(full_reasoning: str) -> list:
    """
    Parse full_reasoning into a list of step strings (including Final Answer line).
    Each element is the full line, e.g. "Step 3: ..."
    """
    lines = [l.strip() for l in full_reasoning.strip().split("\n") if l.strip()]
    steps = []
    for line in lines:
        if _STEP_NUM_RE.match(line) or line.lower().startswith("final answer"):
            steps.append(line)
    return steps

def get_wrong_step_num(prefix_len: int) -> int:
    """Wrong step is always the step immediately after the prefix."""
    return prefix_len + 1

def downstream_steps(rollout_steps: list, wrong_step_num: int) -> list:
    """
    Return only the steps that come after the wrong step in the rollout.
    Filters by step number > wrong_step_num, plus Final Answer.
    """
    result = []
    for step in rollout_steps:
        m = _STEP_NUM_RE.match(step)
        if m:
            num = int(m.group(1))
            if num > wrong_step_num:
                result.append(step)
        elif step.lower().startswith("final answer"):
            result.append(step)
    return result

                                                                                

def build_step_classification_prompt(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    attribution_source: str,
    prior_downstream_steps: list = None,
    seed: int = 42,
) -> tuple:
    """
    Build a forced-choice prompt to classify one downstream step.

    prior_downstream_steps: list of step strings already annotated before this one.

    Returns:
        prompt_str  (str)
        option_map  (dict): letter -> (label_key, label_text)
    """
    rng = random.Random(seed)

    options = list(LABEL_MAP.items())
    rng.shuffle(options)

    option_map   = {}
    option_lines = []
    for idx, (label_key, text) in enumerate(options):
        letter = LETTERS[idx]
        option_map[letter] = (label_key, text)
        option_lines.append(f"{letter}. {text}")

    step_re = re.compile(r'^Step\s*\d+[:.]*\s*')
    prefix_section = "\n".join(
        f"Step {i+1}: {step_re.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )

    ws_content = step_re.sub('', wrong_step).strip()
    ds_content = downstream_step.strip()

                         
    if attribution_source.startswith("step_"):
        attr_note = f"Root cause of wrong step: model misused a value from {attribution_source.replace('_', ' ')}."
    elif attribution_source == "question":
        attr_note = "Root cause of wrong step: model misread the original question."
    else:
        attr_note = "Root cause of wrong step: independent arithmetic/algebraic mistake."

                                    
    prior_section = ""
    if prior_downstream_steps:
        prior_section = (
            "Previous downstream steps (already executed after the wrong step):\n"
            + "\n".join(prior_downstream_steps)
            + "\n\n"
        )

    prompt = (
        "You are analyzing a math reasoning chain that contains an error.\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct reasoning prefix (all gold steps):\n{prefix_section}\n\n"
        f"Wrong step (confirmed error — all rollouts from here lead to wrong answers):\n"
        f"Step {wrong_step_num}: {ws_content}\n"
        f"({attr_note})\n\n"
        + prior_section +
        f"Downstream step to classify:\n{ds_content}\n\n"
        "This step already operates on wrong context inherited from the wrong step. "
        "Before answering, carefully verify that this step correctly accounts for ALL values "
        "and results established in prior steps, regardless of whether those prior steps are "
        "correct or wrong. If anything is missing or inconsistently used, classify as new_error. "
        "Classify whether this step introduces a fresh mistake of its own. Pick exactly one:\n\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )
    return prompt, option_map

                                                                               

def build_new_error_attribution_prompt(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    seed: int = 42,
) -> tuple:
    """
    Build a forced-choice attribution prompt for a downstream step labeled new_error.

    Options:
      - One per prefix step: "step_i" — misused a value from gold Step i
      - "step_{wrong_step_num}": misused the flawed value from the wrong step
      - "question":         "misread the original question"
      - "independent":      "independent arithmetic/algebraic mistake"

    Returns:
        prompt_str  (str)
        option_map  (dict): letter -> (source_key, option_text)
    """
    rng = random.Random(seed)
    step_re = re.compile(r'^Step\s*\d+[:.]*\s*')

    options = []
    for i, s in enumerate(prefix_steps):
        content = step_re.sub('', s).strip()[:100]
        options.append((
            f"step_{i+1}",
            f"The model misused a value or result from gold Step {i+1} (\"{content}\").",
        ))
    ws_content = step_re.sub('', wrong_step).strip()[:100]
    options.append((
        f"step_{wrong_step_num}",
        f"The model misused or misread the flawed value introduced by the wrong Step {wrong_step_num} (\"{ws_content}\").",
    ))
    options.append(("question",    "The model misread or misinterpreted the original question."))
    options.append(("independent", "The model made an independent arithmetic or algebraic mistake not caused by any prior step."))
    rng.shuffle(options)

    option_map   = {}
    option_lines = []
    for idx, (source_key, text) in enumerate(options):
        letter = LETTERS[idx]
        option_map[letter] = (source_key, text)
        option_lines.append(f"{letter}. {text}")

    prefix_section = "\n".join(
        f"Step {i+1}: {step_re.sub('', s).strip()}"
        for i, s in enumerate(prefix_steps)
    )
    ds_content = downstream_step.strip()

    prompt = (
        "You are analyzing a fresh mistake in a math reasoning chain.\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct reasoning prefix (all gold steps):\n{prefix_section}\n\n"
        f"Wrong step (confirmed error):\n"
        f"Step {wrong_step_num}: {ws_content}\n\n"
        f"Downstream step that introduces a new mistake:\n{ds_content}\n\n"
        "This downstream step makes a fresh error of its own (on top of inheriting wrong context). "
        "Identify the most likely cause of this new mistake:\n\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )
    return prompt, option_map

def run_new_error_attribution(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    judge_model,
    judge_tokenizer,
    seed: int = 42,
) -> dict:
    prompt, option_map = build_new_error_attribution_prompt(
        question, prefix_steps, wrong_step, wrong_step_num, downstream_step, seed=seed,
    )

    device    = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)

    last_logits = outputs.logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in option_map:
        token_id       = judge_tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
        scores[letter] = log_probs[token_id].item()

    best_letter = max(scores, key=scores.get)
    sorted_vals = sorted(scores.values(), reverse=True)
    conf_margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    source, text = option_map[best_letter]

    return {
        "source":            source,
        "explanation":       text,
        "confidence_margin": round(conf_margin, 4),
        "best_letter":       best_letter,
        "scores":            {k: round(v, 4) for k, v in scores.items()},
        "option_map":        {k: list(v) for k, v in option_map.items()},
        "seed":              seed,
    }

def attribute_new_error(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    judge_model,
    judge_tokenizer,
) -> dict:
    """Two-run self-consistency for new_error attribution."""
    run_1 = run_new_error_attribution(
        question, prefix_steps, wrong_step, wrong_step_num,
        downstream_step, judge_model, judge_tokenizer, seed=42,
    )
    run_2 = run_new_error_attribution(
        question, prefix_steps, wrong_step, wrong_step_num,
        downstream_step, judge_model, judge_tokenizer, seed=137,
    )
    return {
        "source":            run_1["source"],
        "explanation":       run_1["explanation"],
        "confidence_margin": run_1["confidence_margin"],
        "self_consistent":   run_1["source"] == run_2["source"],
        "run_1_source":      run_1["source"],
        "run_2_source":      run_2["source"],
        "run_1":             run_1,
        "run_2":             run_2,
    }

                                                                                

def run_step_classification(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    attribution_source: str,
    judge_model,
    judge_tokenizer,
    prior_downstream_steps: list = None,
    seed: int = 42,
) -> dict:
    prompt, option_map = build_step_classification_prompt(
        question, prefix_steps, wrong_step, wrong_step_num,
        downstream_step, attribution_source,
        prior_downstream_steps=prior_downstream_steps, seed=seed,
    )

    device    = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)

    last_logits = outputs.logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in option_map:
        token_id        = judge_tokenizer(f" {letter}", add_special_tokens=False).input_ids[0]
        scores[letter]  = log_probs[token_id].item()

    best_letter  = max(scores, key=scores.get)
    sorted_vals  = sorted(scores.values(), reverse=True)
    conf_margin  = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    label, text  = option_map[best_letter]

    return {
        "label":            label,
        "label_text":       text,
        "confidence_margin": round(conf_margin, 4),
        "best_letter":      best_letter,
        "scores":           {k: round(v, 4) for k, v in scores.items()},
        "option_map":       {k: list(v) for k, v in option_map.items()},
        "seed":             seed,
    }

def classify_step(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    downstream_step: str,
    attribution_source: str,
    judge_model,
    judge_tokenizer,
    prior_downstream_steps: list = None,
) -> dict:
    """
    Run two judge passes for self-consistency.
    If the step is labeled new_error, also run attribution to identify the cause.
    """
    run_1 = run_step_classification(
        question, prefix_steps, wrong_step, wrong_step_num,
        downstream_step, attribution_source, judge_model, judge_tokenizer,
        prior_downstream_steps=prior_downstream_steps, seed=42,
    )
    run_2 = run_step_classification(
        question, prefix_steps, wrong_step, wrong_step_num,
        downstream_step, attribution_source, judge_model, judge_tokenizer,
        prior_downstream_steps=prior_downstream_steps, seed=137,
    )
    label = run_1["label"]

                                                                                     
    new_error_attribution = None
    if label == "new_error":
        new_error_attribution = attribute_new_error(
            question, prefix_steps, wrong_step, wrong_step_num,
            downstream_step, judge_model, judge_tokenizer,
        )

    return {
        "step":                  downstream_step,
        "label":                 label,
        "self_consistent":       run_1["label"] == run_2["label"],
        "run_1_label":           run_1["label"],
        "run_2_label":           run_2["label"],
        "confidence_margin":     run_1["confidence_margin"],
        "new_error_attribution": new_error_attribution,
        "run_1":                 run_1,
        "run_2":                 run_2,
    }

                                                                                

def trace_rollout(
    question: str,
    prefix_steps: list,
    wrong_step: str,
    wrong_step_num: int,
    rollout: dict,
    attribution_source: str,
    judge_model,
    judge_tokenizer,
) -> dict:
    """
    Annotate all downstream steps in one rollout.
    Returns a dict with the rollout's is_correct flag and per-step labels.
    """
    full_reasoning = rollout.get("full_reasoning", "")
    rollout_steps  = parse_rollout_steps(full_reasoning)
    ds_steps       = downstream_steps(rollout_steps, wrong_step_num)

    step_annotations = []
    prior_downstream = []
    for step in ds_steps:
        annotation = classify_step(
            question, prefix_steps, wrong_step, wrong_step_num,
            step, attribution_source, judge_model, judge_tokenizer,
            prior_downstream_steps=prior_downstream if prior_downstream else None,
        )
        step_annotations.append(annotation)
        prior_downstream.append(step)                             
        attr_str = ""
        if annotation["new_error_attribution"]:
            attr_str = f"  new_error_src={annotation['new_error_attribution']['source']}"
        print(f"      step: {step[:70]!r}  => {annotation['label']}{attr_str}  "
              f"(consistent={annotation['self_consistent']}, conf={annotation['confidence_margin']:.3f})")

    return {
        "is_correct":        rollout.get("is_correct"),
        "downstream_steps":  step_annotations,
        "n_downstream":      len(step_annotations),
    }

def trace_sample(record: dict, attribution_source: str, judge_model, judge_tokenizer,
                 num_rollouts: int = NUM_ROLLOUTS) -> dict:
    """
    Annotate the first `num_rollouts` rollouts for one wrong-step sample.
    """
    question      = record["question"]
    prefix_steps  = parse_prefix_steps(record["prefix_text"])
    wrong_step    = record["wrong_step"]
    wrong_step_num = get_wrong_step_num(record["prefix_len"])

                                        
    candidates    = record.get("all_candidates", [])
    wrong_cand    = next((c for c in candidates if c["candidate_step"] == wrong_step), None)

    if wrong_cand is None:
        return {"error": "wrong candidate not found in all_candidates"}

    rollouts_to_trace = wrong_cand["rollouts"][:num_rollouts]
    rollout_traces = []
    for r_idx, rollout in enumerate(rollouts_to_trace):
        print(f"    Rollout {r_idx+1}/{len(rollouts_to_trace)}")
        trace = trace_rollout(
            question, prefix_steps, wrong_step, wrong_step_num,
            rollout, attribution_source, judge_model, judge_tokenizer,
        )
        rollout_traces.append(trace)

    return {
        "dataset_index":    record["dataset_index"],
        "question":         question,
        "prefix_text":      record["prefix_text"],
        "prefix_len":       record["prefix_len"],
        "wrong_step":       wrong_step,
        "wrong_step_num":   wrong_step_num,
        "attribution_source": attribution_source,
        "rollout_traces":   rollout_traces,
        "processed_at":     datetime.utcnow().isoformat(),
    }

                                                                                

def main():
    parser = argparse.ArgumentParser(description="Error trace annotation for wrong-step samples")
    parser.add_argument("--input",  default=INPUT_FILE,  help="Wrong-step JSONL (e.g. wrong_steps_50k.jsonl)")
    parser.add_argument("--attribution", default="judge_attribution_50k.jsonl",
                        help="Judge attribution JSONL (for attribution_source lookup)")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output error trace JSONL")
    parser.add_argument("--max",          type=int, default=MAX_SAMPLES,  help="Max samples to process")
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS, help="Rollouts to trace per sample (default: 3)")
    parser.add_argument("--part",         type=int, default=None,         help="Part index (0-based) for parallel jobs")
    parser.add_argument("--num-parts",    type=int, default=1,            help="Total number of parallel parts")
    args = parser.parse_args()

    judge_model, judge_tokenizer = load_judge()

                             
    all_records   = load_jsonl(args.input)
    wrong_records = [r for r in all_records if r.get("wrong_step_found") and r.get("wrong_step")]
    print(f"Wrong-step records: {len(wrong_records)}")

                                        
    if args.part is not None:
        chunk = len(wrong_records) // args.num_parts
        start = args.part * chunk
        end   = start + chunk if args.part < args.num_parts - 1 else len(wrong_records)
        wrong_records = wrong_records[start:end]
        if args.output == OUTPUT_FILE:
            args.output = f"error_trace_part_{args.part}.jsonl"
        print(f"Part {args.part}/{args.num_parts}: records {start}-{end} → {args.output}")

                                                                          
    attr_lookup = {}
    if Path(args.attribution).exists():
        for rec in load_jsonl(args.attribution):
            src = rec.get("llm_judge", {}).get("root_cause_source", "independent")
            attr_lookup[rec["dataset_index"]] = src
        print(f"Loaded attribution for {len(attr_lookup)} samples")
    else:
        print(f"Attribution file not found: {args.attribution} — defaulting to 'independent'")

                  
    processed  = load_processed_indices(args.output)
    to_process = [r for r in wrong_records if r["dataset_index"] not in processed]
    if args.part is None:
        to_process = to_process[:max(0, args.max - len(processed))]
    print(f"Already processed: {len(processed)}  |  Will process now: {len(to_process)}\n")

    output_path = Path(args.output)
    with open(output_path, "a") as out_f:
        for i, record in enumerate(to_process):
            idx  = record["dataset_index"]
            src  = attr_lookup.get(idx, "independent")
            ts   = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [{i+1}/{len(to_process)}] dataset_index={idx}  "
                  f"prefix_len={record['prefix_len']}  attribution={src}")

            result = trace_sample(record, src, judge_model, judge_tokenizer,
                                   num_rollouts=args.num_rollouts)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            print(f"  Saved. Rollouts traced: {len(result.get('rollout_traces', []))}\n")

    print(f"\nDone. Results written to: {output_path.resolve()}")

if __name__ == "__main__":
    main()

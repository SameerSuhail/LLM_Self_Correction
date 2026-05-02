"""
cc_self_correction_gen.py

Reads cc_attribution_comparison.jsonl and, for each record, fills a JSON object
``self_correction`` (added to the output line) with string fields written by
Qwen2.5-Coder-14B-Instruct:

  "error_trace"     — first-person trace of the mistake (not a Python attribute; JSON key)
  "error_diagnosis" — one-sentence diagnosis
  "corrected_step"  — corrected reasoning step line

Output: cc_self_correction_part_{N}.jsonl  (merge into cc_self_correction_sft.jsonl)

Usage:
  python cc_self_correction_gen.py --input cc_attribution_comparison.jsonl \
                                    --part 0 --num-parts 4
"""

import re
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME    = "Qwen/Qwen2.5-Coder-14B-Instruct"
_STEP_LABEL   = re.compile(r'^Step\s*\d+[:.]*\s*', re.IGNORECASE)


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
    return {
        rec["dataset_index"] for rec in load_jsonl(output_file)
        if rec.get("self_correction", {}).get("error_trace")
    }


def build_source_context(prefix_text: str, root_cause_source: str) -> str:
    """Human-readable description of the attribution source."""
    prefix_steps = [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]

    if root_cause_source == "problem":
        return "the original problem statement — I misread or misunderstood something in the problem"
    elif root_cause_source == "independent":
        return "an independent algorithmic or logical mistake — not caused by any prior step"
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        if m:
            step_num = int(m.group(1))
            if step_num <= len(prefix_steps):
                content = _STEP_LABEL.sub("", prefix_steps[step_num - 1]).strip()
                return f"Step {step_num} of my prior reasoning — I misapplied a concept from: \"{content[:120]}\""
    return root_cause_source


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Model loaded. Devices: {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(
    problem: str,
    prefix_text: str,
    wrong_step: str,
    root_cause_source: str,
) -> str:
    source_context = build_source_context(prefix_text, root_cause_source)
    ws_content     = _STEP_LABEL.sub("", wrong_step).strip()
    prefix_steps   = [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]
    ws_num         = len(prefix_steps) + 1

    if root_cause_source == "problem":
        lookback = ('start with "Let me look back at the problem..." then re-read the '
                    'problem statement and identify what you misread or misunderstood')
    elif root_cause_source == "independent":
        lookback = ('start with "Let me rethink this step..." then re-examine your '
                    'reasoning and identify the algorithmic or logical mistake you made')
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        step_num = m.group(1) if m else "?"
        lookback = (f'start with "Let me recheck Step {step_num}..." then re-read that '
                    f'step and identify what concept or result you misapplied from it')

    return (
        "You are solving a competitive programming problem step by step using algorithmic reasoning. "
        "You made an error in one of your reasoning steps. "
        "Reflect on your mistake and correct it.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Your correct reasoning so far:\n{prefix_text.strip()}\n\n"
        f"Your wrong step:\n"
        f"Step {ws_num}: {ws_content}\n\n"
        f"The source of your error has been identified as: {source_context}.\n\n"
        "Tasks:\n"
        f"1. Error trace: {lookback}. "
        "Be specific — reference the actual algorithm, data structure, condition, or constraint involved. "
        "Write in first person.\n"
        "2. Diagnosis: In one sentence, state precisely what went wrong. "
        "Write in first person (e.g. 'I incorrectly...', 'I misread...', 'I applied...').\n"
        "3. Corrected step: Write the correct version of the wrong reasoning step. "
        "Use the same format as the other steps (Step N: ...).\n\n"
        "Respond in this exact format:\n"
        "Error trace: <first-person trace starting with the look-back phrase>\n"
        "Diagnosis: <one-sentence first-person diagnosis>\n"
        f"Corrected step: Step {ws_num}: <corrected step text>"
    )


# ─── Generation ───────────────────────────────────────────────────────────────

def parse_output(text: str) -> tuple:
    """Parse error_trace, diagnosis, corrected_step from model output."""
    trace_match = re.search(r"Error trace:\s*(.+?)(?=Diagnosis:|$)",       text, re.DOTALL | re.IGNORECASE)
    diag_match  = re.search(r"Diagnosis:\s*(.+?)(?=Corrected step:|$)",    text, re.DOTALL | re.IGNORECASE)
    corr_match  = re.search(r"Corrected step:\s*(.+?)$",                   text, re.DOTALL | re.IGNORECASE)

    error_trace    = trace_match.group(1).strip() if trace_match else ""
    diagnosis      = diag_match.group(1).strip()  if diag_match  else ""
    corrected_step = corr_match.group(1).strip()  if corr_match  else ""
    return error_trace, diagnosis, corrected_step


def generate_self_correction(
    model,
    tokenizer,
    problem: str,
    prefix_text: str,
    wrong_step: str,
    root_cause_source: str,
    max_new_tokens: int = 1024,
) -> dict:
    prompt   = build_prompt(problem, prefix_text, wrong_step, root_cause_source)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    input_ids      = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(input_ids.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # Strip <think>...</think> if present (some Qwen models produce it)
    if "</think>" in generated:
        generated_clean = generated.split("</think>", 1)[1].strip()
    else:
        generated_clean = generated

    error_trace, diagnosis, corrected_step = parse_output(generated_clean)
    if not error_trace or not diagnosis:
        error_trace, diagnosis, corrected_step = parse_output(generated)

    return {
        "error_trace":     error_trace,
        "error_diagnosis": diagnosis,
        "corrected_step":  corrected_step,
        "raw_output":      generated,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="cc_attribution_comparison.jsonl")
    parser.add_argument("--output",    default=None,
                        help="Output file path (auto-set from --part if not given)")
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--part",      type=int, default=None,
                        help="Part index (0-based) for parallel jobs")
    parser.add_argument("--num-parts", type=int, default=1,
                        help="Total number of parallel parts")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # Split into parts for parallel jobs
    if args.part is not None:
        chunk  = len(records) // args.num_parts
        start  = args.part * chunk
        end    = start + chunk if args.part < args.num_parts - 1 else len(records)
        records = records[start:end]
        if args.output is None:
            args.output = f"cc_self_correction_part_{args.part}.jsonl"
        print(f"Part {args.part}/{args.num_parts}: records {start}–{end-1} → {args.output}")
    else:
        if args.output is None:
            args.output = "cc_self_correction_sft.jsonl"

    processed = load_processed_indices(args.output)
    print(f"Already processed: {len(processed)}")

    to_process = [r for r in records if r["dataset_index"] not in processed]
    if args.max is not None:
        to_process = to_process[:args.max]
    print(f"To process: {len(to_process)} records\n")

    if not to_process:
        print("Nothing to do.")
        return

    model, tokenizer = load_model()

    for i, record in enumerate(to_process):
        idx    = record["dataset_index"]
        source = (record.get("attributions", {}) or {}).get("llm_judge", {}) or {}
        root_cause = source.get("root_cause_source", "independent")
        error_type = source.get("error_type", "")

        print(f"[{i+1}/{len(to_process)}] idx={idx}  "
              f"problem={record.get('problem_name','')}  "
              f"source={root_cause}  error_type={error_type}", flush=True)

        try:
            sc = generate_self_correction(
                model, tokenizer,
                problem          = record["problem"],
                prefix_text      = record["prefix_text"],
                wrong_step       = record["wrong_step"],
                root_cause_source= root_cause,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            sc = {
                "error_trace":     "",
                "error_diagnosis": "",
                "corrected_step":  "",
                "raw_output":      f"ERROR: {e}",
            }

        out_record = dict(record)
        out_record["self_correction"] = sc
        append_jsonl(args.output, out_record)

        print(f"  Trace:     {sc['error_trace'][:120]}")
        print(f"  Diagnosis: {sc['error_diagnosis'][:120]}")
        print(f"  Corrected: {sc['corrected_step'][:120]}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()

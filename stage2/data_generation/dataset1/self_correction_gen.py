import re
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

                                                                                

JUDGE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

_STEP_LABEL_RE = re.compile(r'^Step\s*\d+[:.]*\s*', re.IGNORECASE)

                                                                                

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
                                                                    
    return {
        rec["dataset_index"] for rec in load_jsonl(output_file)
        if rec.get("self_correction", {}).get("error_trace")
    }

def build_source_context(prefix_text: str, root_cause_source: str) -> str:
    """Build a human-readable description of the attribution source."""
    prefix_steps = [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]

    if root_cause_source == "question":
        return "the original problem statement — I misread or misinterpreted something in the question"
    elif root_cause_source == "independent":
        return "an independent arithmetic or algebraic mistake — not caused by any prior step"
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        if m:
            step_num = int(m.group(1))
            if step_num <= len(prefix_steps):
                step_content = _STEP_LABEL_RE.sub("", prefix_steps[step_num - 1]).strip()
                return f"Step {step_num} of my prior reasoning — I misused a result from: \"{step_content[:120]}\""
    return root_cause_source

                                                                                

def load_judge():
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Judge loaded. Devices: {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer

                                                                                

def build_prompt(question: str, prefix_text: str, wrong_step: str, root_cause_source: str) -> str:
    source_context = build_source_context(prefix_text, root_cause_source)
    ws_content = _STEP_LABEL_RE.sub("", wrong_step).strip()
    prefix_steps = [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]
    ws_num = len(prefix_steps) + 1

                                                       
    if root_cause_source == "question":
        lookback = 'start with "Let me look back at the question..." then re-read the problem statement and identify what you misread or misinterpreted'
    elif root_cause_source == "independent":
        lookback = 'start with "Let me rethink this step..." then re-examine your calculation or logic and identify the arithmetic or algebraic mistake you made'
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        step_num = m.group(1) if m else "?"
        lookback = f'start with "Let me recheck Step {step_num}..." then re-read that step and identify what value or result you misused from it'

    return (
        "You are solving a math problem step by step. "
        "You made an error in one of your reasoning steps. "
        "Reflect on your mistake and correct it.\n\n"
        f"Problem:\n{question}\n\n"
        f"Your correct reasoning so far:\n{prefix_text.strip()}\n\n"
        f"Your wrong step:\n"
        f"Step {ws_num}: {ws_content}\n\n"
        f"The source of your error has been identified as: {source_context}.\n\n"
        "Tasks:\n"
        f"1. Error trace: {lookback}. "
        "Be specific — reference the actual values, conditions, or statements involved. "
        "Write in first person.\n"
        "2. Diagnosis: In one sentence, state precisely what went wrong. "
        "Write in first person (e.g. 'I incorrectly...', 'I misread...', 'I made...').\n"
        "3. Corrected step: Write the correct version of the wrong step.\n\n"
        "Respond in this exact format:\n"
        "Error trace: <first-person trace starting with the look-back phrase>\n"
        "Diagnosis: <one-sentence first-person diagnosis>\n"
        "Corrected step: <corrected step text>"
    )

                                                                                

def parse_output(text: str) -> tuple:
    """Parse error_trace, diagnosis, corrected_step from model output."""
    error_trace = ""
    diagnosis = ""
    corrected_step = ""

    trace_match = re.search(r"Error trace:\s*(.+?)(?=Diagnosis:|$)", text, re.DOTALL | re.IGNORECASE)
    diag_match  = re.search(r"Diagnosis:\s*(.+?)(?=Corrected step:|$)", text, re.DOTALL | re.IGNORECASE)
    corr_match  = re.search(r"Corrected step:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)

    if trace_match:
        error_trace = trace_match.group(1).strip()
    if diag_match:
        diagnosis = diag_match.group(1).strip()
    if corr_match:
        corrected_step = corr_match.group(1).strip()

    return error_trace, diagnosis, corrected_step

def generate_self_correction(
    model,
    tokenizer,
    question: str,
    prefix_text: str,
    wrong_step: str,
    root_cause_source: str,
    max_new_tokens: int = 2048,
) -> dict:
    prompt = build_prompt(question, prefix_text, wrong_step, root_cause_source)

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

                                               
    if "</think>" in generated:
        after_think = generated.split("</think>", 1)[1].strip()
    else:
        after_think = generated

                                                                
    error_trace, diagnosis, corrected_step = parse_output(after_think)
    if not error_trace or not diagnosis:
        error_trace, diagnosis, corrected_step = parse_output(generated)

    return {
        "error_trace":      error_trace,
        "error_diagnosis":  diagnosis,
        "corrected_step":   corrected_step,
        "raw_output":       generated,
    }

                                                                                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="judge_attribution_50k.jsonl")
    parser.add_argument("--output",    default="self_correction_50k.jsonl")
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--skip",      type=int, nargs="*", default=[])
    parser.add_argument("--part",      type=int, default=None,
                        help="Part index (0-based) for parallel jobs")
    parser.add_argument("--num-parts", type=int, default=1,
                        help="Total number of parallel parts")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

                                        
    if args.part is not None:
        chunk = len(records) // args.num_parts
        start = args.part * chunk
        end   = start + chunk if args.part < args.num_parts - 1 else len(records)
        records = records[start:end]
                                                                 
        if args.output == "self_correction_50k.jsonl":
            args.output = f"self_correction_part_{args.part}.jsonl"
        print(f"Part {args.part}/{args.num_parts}: records {start}-{end} → {args.output}")

    processed = load_processed_indices(args.output)
    skip_set  = set(args.skip)
    print(f"Already processed: {len(processed)} | Skipping: {skip_set}")

    to_process = [
        r for r in records
        if r["dataset_index"] not in processed
        and r["dataset_index"] not in skip_set
    ]
    if args.max is not None:
        to_process = to_process[:args.max]
    print(f"To process: {len(to_process)} records")

    if not to_process:
        print("Nothing to do.")
        return

    model, tokenizer = load_judge()

    for i, record in enumerate(to_process):
        idx = record["dataset_index"]
        print(f"[{i+1}/{len(to_process)}] dataset_index={idx} "
              f"source={record['llm_judge']['root_cause_source']}", flush=True)

        try:
            sc = generate_self_correction(
                model, tokenizer,
                question=record["question"],
                prefix_text=record["prefix_text"],
                wrong_step=record["wrong_step"],
                root_cause_source=record["llm_judge"]["root_cause_source"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            sc = {"error_trace": "", "error_diagnosis": "", "corrected_step": "", "raw_output": f"ERROR: {e}"}

        out_record = dict(record)
        out_record["self_correction"] = sc
        append_jsonl(args.output, out_record)

        print(f"  Trace:     {sc['error_trace'][:120]}")
        print(f"  Diagnosis: {sc['error_diagnosis'][:120]}")
        print(f"  Corrected: {sc['corrected_step'][:120]}")

    print(f"\nDone. Output: {args.output}")

if __name__ == "__main__":
    main()

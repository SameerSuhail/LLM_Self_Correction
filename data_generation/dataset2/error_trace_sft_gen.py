import re
import json
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

                                                                                

JUDGE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
_STEP_LABEL_RE   = re.compile(r'^Step\s*\d+[:.]*\s*', re.IGNORECASE)

                                                                                

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
        if rec.get("sft_output", {}).get("detection")
    }

def source_to_lookback(root_cause_source: str, prefix_text: str) -> str:
    """Build the 'look back' instruction based on attribution source."""
    if root_cause_source == "question":
        return 'start with "Let me look back at the question..." then trace how something in the problem statement was misread or misinterpreted'
    elif root_cause_source == "independent":
        return 'start with "Let me rethink this step..." then trace the independent arithmetic or algebraic mistake made in the wrong step'
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        if m:
            step_num = m.group(1)
            return f'start with "Let me recheck Step {step_num}..." then trace how a value or result from Step {step_num} was misused in the wrong step'
    return 'start with "Let me trace back..." and identify the source of the error'

def source_to_phrase(root_cause_source: str, prefix_text: str) -> str:
    """Human readable phrase for the attribution source."""
    prefix_steps = [l.strip() for l in prefix_text.strip().split("\n") if l.strip()]
    if root_cause_source == "question":
        return "the original problem statement — the wrong step misread or misinterpreted the question"
    elif root_cause_source == "independent":
        return "an independent arithmetic or algebraic mistake in the wrong step itself"
    else:
        m = re.match(r"step_(\d+)", root_cause_source)
        if m:
            step_num = int(m.group(1))
            if step_num <= len(prefix_steps):
                content = _STEP_LABEL_RE.sub("", prefix_steps[step_num - 1]).strip()[:100]
                return f'Step {step_num} of the prefix — the wrong step misused: "{content}"'
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

                                                                                

def select_rollout_and_detection_point(rollout_traces: list, rng: random.Random):
    """
    Pick a rollout with at least 2 downstream steps.
    Randomly sample a detection point from index 1 onwards (not immediate).
    Returns (rollout, detection_index) or None if no valid rollout.
    """
    valid = [r for r in rollout_traces if len(r.get("downstream_steps", [])) >= 2]
    if not valid:
        return None, None
    rollout = rng.choice(valid)
    steps = rollout["downstream_steps"]
                                                                              
    detection_idx = rng.randint(1, len(steps) - 1)
    return rollout, detection_idx

                                                                                

def build_prompt(
    question: str,
    prefix_text: str,
    wrong_step: str,
    wrong_step_num: int,
    downstream_steps: list,
    detection_idx: int,
    root_cause_source: str,
) -> str:
    detection_step_info = downstream_steps[detection_idx]
    detection_label     = detection_step_info["label"]
    new_error_attr      = detection_step_info.get("new_error_attribution") or {}
    new_error_source    = new_error_attr.get("source", "")

                                                               
    wrong_step_lookback = source_to_lookback(root_cause_source, prefix_text)
    wrong_step_phrase   = source_to_phrase(root_cause_source, prefix_text)

    ws_content = _STEP_LABEL_RE.sub("", wrong_step).strip()
    detection_step_num  = wrong_step_num + detection_idx + 1

                         
    chain_lines = [f"Step {wrong_step_num}: {ws_content}"]
    for i, step_info in enumerate(downstream_steps[:detection_idx + 1]):
        step_text = _STEP_LABEL_RE.sub("", step_info["step"]).strip()
        chain_lines.append(f"Step {wrong_step_num + i + 1}: {step_text}")
    chain_text = "\n".join(chain_lines)

                                                        
    if detection_label == "propagated":
        retrace_instruction = (
            f"2. Retrace: Trace backward from Step {detection_step_num} to identify Step {wrong_step_num} "
            f"as the error origin. Show how each intermediate step carried the wrong value forward."
        )
        error_trace_instruction = (
            f"3. Error trace: Now look back at the source of Step {wrong_step_num}'s error. "
            f"{wrong_step_lookback}. Be specific — reference the actual values or conditions involved."
        )
    else:             
        if new_error_source == f"step_{wrong_step_num}" or "step_" in new_error_source:
            level1 = (
                f"First, trace the mistake at Step {detection_step_num} back to its source "
                f"({new_error_source}) — explain how a value from that step was misused here. "
                f"Then trace further back to Step {wrong_step_num} as the root origin."
            )
        else:
            level1 = (
                f"First, identify what went wrong at Step {detection_step_num}. "
                f"Then trace back to Step {wrong_step_num} as the root origin."
            )
        retrace_instruction = (
            f"2. Retrace: {level1}"
        )
        error_trace_instruction = (
            f"3. Error trace: Now trace the root error in Step {wrong_step_num} itself. "
            f"{wrong_step_lookback}. Be specific — reference the actual values or conditions involved."
        )

    return (
        "You are solving a math problem step by step. "
        "You made an error earlier in your reasoning. "
        "At the current step, reflect that something is wrong and trace back to correct it.\n\n"
        f"Problem:\n{question}\n\n"
        f"Your correct reasoning so far:\n{prefix_text.strip()}\n\n"
        f"Your subsequent steps (including the error):\n{chain_text}\n\n"
        f"The root cause of Step {wrong_step_num}'s error is: {wrong_step_phrase}.\n\n"
        f"At Step {detection_step_num}, generate the self-correction in first person:\n"
        f"1. Detection: Notice something is wrong and decide to trace back.\n"
        f"{retrace_instruction}\n"
        f"{error_trace_instruction}\n"
        f"4. Diagnosis: In one sentence, state what went wrong in Step {wrong_step_num}. First person.\n"
        f"5. Correction: Write corrected versions of all steps from Step {wrong_step_num} to Step {detection_step_num}.\n\n"
        "Respond in this exact format:\n"
        "Detection: <first-person detection>\n"
        "Retrace: <backward trace to error origin>\n"
        "Error trace: <first-person look-back>\n"
        "Diagnosis: <one-sentence first-person diagnosis>\n"
        f"Correction: <corrected Step {wrong_step_num}> → ... → <corrected Step {detection_step_num}>"
    )

                                                                                

def parse_output(text: str) -> dict:
    fields = {}
    patterns = [
        ("detection",   r"Detection:\s*(.+?)(?=Retrace:|$)"),
        ("retrace",     r"Retrace:\s*(.+?)(?=Error trace:|$)"),
        ("error_trace", r"Error trace:\s*(.+?)(?=Diagnosis:|$)"),
        ("diagnosis",   r"Diagnosis:\s*(.+?)(?=Correction:|$)"),
        ("correction",  r"Correction:\s*(.+?)$"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        fields[key] = m.group(1).strip() if m else ""
    return fields

def generate_sft_output(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
) -> dict:
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

    parsed = parse_output(after_think)
    if not parsed["detection"]:
        parsed = parse_output(generated)

    parsed["raw_output"] = generated
    return parsed

                                                                                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace",     default="error_trace.jsonl")
    parser.add_argument("--output",    default="error_trace_sft.jsonl")
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--part",      type=int, default=None)
    parser.add_argument("--num-parts", type=int, default=1)
    args = parser.parse_args()

    trace_records = load_jsonl(args.trace)
    print(f"Loaded {len(trace_records)} trace records.")

    if args.part is not None:
        chunk = len(trace_records) // args.num_parts
        start = args.part * chunk
        end   = start + chunk if args.part < args.num_parts - 1 else len(trace_records)
        trace_records = trace_records[start:end]
        if args.output == "error_trace_sft.jsonl":
            args.output = f"error_trace_sft_part_{args.part}.jsonl"
        print(f"Part {args.part}/{args.num_parts}: records {start}-{end} → {args.output}")

    processed = load_processed_indices(args.output)
    print(f"Already processed: {len(processed)} — resuming.")

    to_process = [r for r in trace_records if r["dataset_index"] not in processed]
    if args.max is not None:
        to_process = to_process[:args.max]
    print(f"To process: {len(to_process)} records")

    if not to_process:
        print("Nothing to do.")
        return

    model, tokenizer = load_judge()
    rng = random.Random(args.seed)

    for i, record in enumerate(to_process):
        idx              = record["dataset_index"]
        root_cause_source = record.get("attribution_source", "independent")

        print(f"[{i+1}/{len(to_process)}] dataset_index={idx} source={root_cause_source}", flush=True)

        rollout, detection_idx = select_rollout_and_detection_point(
            record.get("rollout_traces", []), rng
        )

        if rollout is None:
            print("  SKIP — no rollout with ≥2 downstream steps")
            continue

        downstream_steps  = rollout["downstream_steps"]
        detection_step    = downstream_steps[detection_idx]
        wrong_step_num    = record["wrong_step_num"]
        detection_step_num = wrong_step_num + detection_idx + 1

        prompt = build_prompt(
            question=record["question"],
            prefix_text=record["prefix_text"],
            wrong_step=record["wrong_step"],
            wrong_step_num=wrong_step_num,
            downstream_steps=downstream_steps,
            detection_idx=detection_idx,
            root_cause_source=root_cause_source,
        )

        try:
            sft_output = generate_sft_output(model, tokenizer, prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            sft_output = {"detection": "", "retrace": "", "error_trace": "",
                          "diagnosis": "", "correction": "", "raw_output": f"ERROR: {e}"}

        out_record = {
            "dataset_index":          idx,
            "question":               record["question"],
            "prefix_text":            record["prefix_text"],
            "prefix_len":             record["prefix_len"],
            "wrong_step":             record["wrong_step"],
            "wrong_step_num":         wrong_step_num,
            "root_cause_source":      root_cause_source,
            "detection_step_num":     detection_step_num,
            "detection_label":        detection_step["label"],
            "new_error_attribution":  detection_step.get("new_error_attribution"),
            "downstream_context":     [s["step"] for s in downstream_steps[:detection_idx + 1]],
            "sft_output":             sft_output,
        }
        append_jsonl(args.output, out_record)

        print(f"  Detection point: Step {detection_step_num} [{detection_step['label']}]")
        print(f"  Detection:  {sft_output['detection'][:120]}")
        print(f"  Retrace:    {sft_output['retrace'][:120]}")
        print(f"  Error trace:{sft_output['error_trace'][:120]}")
        print(f"  Diagnosis:  {sft_output['diagnosis'][:120]}")

    print(f"\nDone. Output: {args.output}")

if __name__ == "__main__":
    main()

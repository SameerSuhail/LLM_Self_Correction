import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import re
import math
import json
import random
import argparse
import torch
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime

GENERATOR_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
JUDGE_MODEL_NAME     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

OUTPUT_FILE            = "wrong_steps_data_v2.jsonl"
NUM_DATASET_SAMPLES    = 700
MAX_CANDIDATES          = 6     
NUM_ROLLOUTS           = 8      
STEP_MAX_NEW_TOKENS    = 128
ROLLOUT_MAX_NEW_TOKENS = 512
STEP_TEMPERATURE       = 1.2
ROLLOUT_TEMPERATURE    = 0.8
TOP_P                  = 0.95

SYSTEM_PROMPT = """You are a math reasoning assistant.

Solve the user's math problem carefully.

Output must follow these rules exactly:
1. Write the solution as numbered reasoning steps.
2. Each reasoning step must be on its own line.
3. Use the exact format:
Step 1: ...
Step 2: ...
Step 3: ...
and so on.
4. Do not combine multiple reasoning steps into one line.
5. The very last line must be exactly in this format:
Final Answer: <answer>
6. Do not write anything after the final answer line.
7. Give only one final answer.
8. Keep the final answer concise.
"""

_transformations = standard_transformations + (implicit_multiplication_application,)

if not os.environ.get("ATTRIBUTION_IMPORT", ""):
    print("Loading generator model...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print(f"Generator on: {model.device}")

    print("Loading judge model...")
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    if torch.cuda.device_count() >= 2:
        judge_device_map = "cuda:1"
        print("Judge on: cuda:1")
    else:
        judge_device_map = "auto"
        print("Only 1 GPU detected — judge using device_map='auto' (may spill to CPU)")

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=judge_device_map,
    )
    print("Models loaded.")

def normalize_answer_string(s: str) -> str:
    s = s.strip()

                                        
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)                          
    s = re.sub(r'\\sqrt\{([^}]+)\}',             r'sqrt(\1)',    s)                         
    s = re.sub(r'\\boxed\{([^}]+)\}',            r'\1',          s)                   
    s = re.sub(r'\\left|\\right',                '',             s)                            
    s = re.sub(r'\\[a-zA-Z]+',                   '',             s)                                 
    s = re.sub(r'[${}]',                         '',             s)                         

                             
    s = s.replace("^", "**")
    s = s.replace("−", "-")
    s = s.replace("×", "*")
    s = s.replace("÷", "/")
    s = re.sub(r"\bpi\b", "pi", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsqrt\s*\(", "sqrt(", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_final_answer(text: str):
    matches = re.findall(r"Final Answer:\s*(.+)", text)
    if not matches:
        return None
    return matches[-1].strip()

def answers_equivalent(pred_answer: str, gold_answer: str, tol: float = 1e-6) -> bool:
    if pred_answer is None or gold_answer is None:
        return False

    pred = normalize_answer_string(pred_answer)
    gold = normalize_answer_string(gold_answer)

    def strip_units_and_text(ans: str) -> str:
        ans = ans.strip()

                               
        ans = re.sub(r"^(the answer is[:\s]*)", "", ans, flags=re.IGNORECASE)
        ans = re.sub(r"^(final answer[:\s]*)",  "", ans, flags=re.IGNORECASE)

                                  
        ans = ans.split("\n")[0].strip()

                              
        ans = re.sub(
            r"\b(inches|inch|feet|foot|ft|cm|mm|meters?|miles?|dollars?|units?)\b",
            "", ans, flags=re.IGNORECASE,
        ).strip()

                                                                                 
        if "=" in ans:
            ans = ans.split("=")[-1].strip()

                                                                                          
        try:
            val = parse_expr(ans, transformations=_transformations, evaluate=True)
            return str(sp.simplify(val))
        except Exception:
            pass

                                                                       
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|sqrt\([^)]+\)", ans)
        if matches:
            ans = matches[-1].strip()
        return ans.strip()

    pred = strip_units_and_text(pred)
    gold = strip_units_and_text(gold)

    if pred == gold:
        return True

    try:
        pred_expr = parse_expr(pred, transformations=_transformations, evaluate=True)
        gold_expr = parse_expr(gold, transformations=_transformations, evaluate=True)
        if sp.simplify(pred_expr - gold_expr) == 0:
            return True
        pred_val = float(sp.N(pred_expr))
        gold_val = float(sp.N(gold_expr))
        if math.isclose(pred_val, gold_val, rel_tol=tol, abs_tol=tol):
            return True
    except Exception:
        pass

    return False

def extract_gold_answer(gold_text: str):
    answer = extract_final_answer(gold_text)
    if answer is None:
        match = re.search(r"The answer is:\s*(.+)", gold_text)
        if match:
            answer = match.group(1).strip()
    return answer

def score_continuation(mdl, tok, prompt, continuation):
    device = next(mdl.parameters()).device
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    cont_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = mdl(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    prompt_len = prompt_ids.shape[1]
    cont_len = cont_ids.shape[1]
    cont_logits = logits[:, prompt_len - 1:prompt_len - 1 + cont_len, :]
    cont_targets = target_ids[:, prompt_len - 1:prompt_len - 1 + cont_len]
    log_probs = torch.log_softmax(cont_logits, dim=-1)
    token_log_probs = log_probs.gather(2, cont_targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()

def batch_llm_judge_full_solution(question: str, solutions: list, gold_answer: str):
    """
    Judge multiple rollout solutions in a single batched forward pass.
    Replaces N sequential calls to llm_judge_full_solution (each doing 2 forward passes)
    with one batched forward pass over all N solutions.
    Returns a list of (is_correct, judge_raw) tuples.
    """
    prompts = []
    correct_labels = []
    text_options = []

    for solution in solutions:
        if random.random() < 0.5:
            text_a = "The generated solution is correct."
            text_b = "The generated solution is incorrect."
            correct_label_if_a = True
        else:
            text_a = "The generated solution is incorrect."
            text_b = "The generated solution is correct."
            correct_label_if_a = False

        prompt = f"""You are judging a math solution.

Problem:
{question}

Generated solution:
{solution}

Gold final answer:
{gold_answer}

Choose exactly one option.

A. {text_a}
B. {text_b}

Answer:"""
        prompts.append(prompt)
        correct_labels.append(correct_label_if_a)
        text_options.append((text_a, text_b))

    device = next(judge_model.parameters()).device

    orig_padding_side = judge_tokenizer.padding_side
    judge_tokenizer.padding_side = "left"
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    encoded = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    ).to(device)
    judge_tokenizer.padding_side = orig_padding_side

    token_a = judge_tokenizer(" A", add_special_tokens=False).input_ids[0]
    token_b = judge_tokenizer(" B", add_special_tokens=False).input_ids[0]

    try:
        with torch.no_grad():
            outputs = judge_model(**encoded)
                                                               
        last_logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        scores_a = log_probs[:, token_a].tolist()
        scores_b = log_probs[:, token_b].tolist()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("      OOM in batch judge — falling back to sequential")
        results = []
        for sol in solutions:
            results.append(llm_judge_full_solution(question, sol, gold_answer))
        return results

    results = []
    for i in range(len(solutions)):
        chosen = "A" if scores_a[i] > scores_b[i] else "B"
        is_correct = correct_labels[i] if chosen == "A" else not correct_labels[i]
        results.append((is_correct, {
            "score_a": scores_a[i],
            "score_b": scores_b[i],
            "chosen": chosen,
            "option_a": text_options[i][0],
            "option_b": text_options[i][1],
            "method": "forced_choice_ab_batched",
        }))
    return results

def llm_judge_full_solution(question: str, generated_solution: str, gold_answer: str):
    if random.random() < 0.5:
        text_a = "The generated solution is correct."
        text_b = "The generated solution is incorrect."
        correct_label_if_a = True
    else:
        text_a = "The generated solution is incorrect."
        text_b = "The generated solution is correct."
        correct_label_if_a = False

    prompt = f"""You are judging a math solution.

Problem:
{question}

Generated solution:
{generated_solution}

Gold final answer:
{gold_answer}

Choose exactly one option.

A. {text_a}
B. {text_b}

Answer:"""

                                                                                          
    device = next(judge_model.parameters()).device
    input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = judge_model(input_ids=input_ids)
    last_logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)

    token_a = judge_tokenizer(" A", add_special_tokens=False).input_ids[0]
    token_b = judge_tokenizer(" B", add_special_tokens=False).input_ids[0]
    score_a = log_probs[token_a].item()
    score_b = log_probs[token_b].item()

    chosen = "A" if score_a > score_b else "B"
    is_correct = correct_label_if_a if chosen == "A" else not correct_label_if_a

    return is_correct, {
        "score_a": score_a,
        "score_b": score_b,
        "chosen": chosen,
        "option_a": text_a,
        "option_b": text_b,
        "method": "forced_choice_ab",
    }

def batched_generate(mdl, input_ids, attention_mask, num_rollouts, **gen_kwargs):
    """
    Try to generate all rollouts in one batched call.
    If OOM, halve the batch size and retry, down to 1.
    Logic is identical — same outputs, just chunked differently under memory pressure.
    """
    batch_sizes = [num_rollouts, 4, 2, 1]

    seen = set()
    batch_sizes = [x for x in batch_sizes if x not in seen and not seen.add(x)]

    for batch_size in batch_sizes:
        try:
            results = []
            for start in range(0, num_rollouts, batch_size):
                end = min(start + batch_size, num_rollouts)
                out = mdl.generate(
                    input_ids=input_ids[start:end],
                    attention_mask=attention_mask[start:end],
                    **gen_kwargs,
                )
                results.append(out)
                torch.cuda.empty_cache()
            return torch.cat(results, dim=0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise  
            print(f"      OOM at batch_size={batch_size}, retrying with batch_size={batch_size // 2}...")
            continue

def sample_wrong_next_step(
    question,
    prefix_text,
    prefix_len,
    gold_text,
    gold_answer,
    num_rollouts=NUM_ROLLOUTS,
    step_max_new_tokens=STEP_MAX_NEW_TOKENS,
    rollout_max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
    step_temperature=STEP_TEMPERATURE,
    rollout_temperature=ROLLOUT_TEMPERATURE,
    top_p=TOP_P,
):
    step_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""You are given a math problem and a correct reasoning prefix extracted from a gold solution.

Your task is to generate the next reasoning step after the prefix.

Rules:
1. Output exactly one line.
2. Use the exact format:
Step {prefix_len + 1}: ...
3. Do not generate any further steps.
4. Do not generate the final answer.
5. Do not generate any extra text.

Problem:
{question}

Correct reasoning prefix:
{prefix_text}"""}
    ]

    step_inputs = tokenizer.apply_chat_template(
        step_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    step_prompt_len = step_inputs["input_ids"].shape[1]

    with torch.no_grad():
        step_outputs = model.generate(
            **step_inputs,
            max_new_tokens=step_max_new_tokens,
            do_sample=True,
            temperature=step_temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    step_generated_tokens = step_outputs[0][step_prompt_len:]
    candidate_step = tokenizer.decode(
        step_generated_tokens, skip_special_tokens=True
    ).strip().split("\n")[0].strip()

    fixed_reasoning = prefix_text + "\n" + candidate_step

    rollout_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""You are given a math problem and a partial reasoning trace.

Continue the reasoning from the given steps and finish the solution.

Rules:
1. Continue from the given reasoning.
2. Write subsequent reasoning steps, each on its own line.
3. Use the exact format:
Step {prefix_len + 2}: ...
Step {prefix_len + 3}: ...
and so on.
4. The very last line must be exactly:
Final Answer: <answer>
5. Do not write anything after the final answer line.

Problem:
{question}

Current reasoning:
{fixed_reasoning}"""}
    ]

    rollout_inputs = tokenizer.apply_chat_template(
        rollout_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    rollout_prompt_len = rollout_inputs["input_ids"].shape[1]

    input_ids      = rollout_inputs["input_ids"].repeat(num_rollouts, 1)
    attention_mask = rollout_inputs["attention_mask"].repeat(num_rollouts, 1)

    rollout_outputs = batched_generate(
        model,
        input_ids,
        attention_mask,
        num_rollouts,
        max_new_tokens=rollout_max_new_tokens,
        do_sample=True,
        temperature=rollout_temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
 
    rollout_results = []
    all_wrong = True

    for i in range(num_rollouts):
        generated_tokens = rollout_outputs[i][rollout_prompt_len:]
        continuation = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        full_reasoning = fixed_reasoning + ("\n" + continuation if continuation else "")

        is_correct, judge_raw = llm_judge_full_solution(
            question=question,
            generated_solution=full_reasoning,
            gold_answer=gold_answer,
        )

        rollout_results.append({
            "full_reasoning": full_reasoning,
            "is_correct": is_correct,
            "judge_raw": judge_raw,
        })

        if is_correct:
            all_wrong = False
            for _ in range(i + 1, num_rollouts):
                rollout_results.append({
                    "full_reasoning": None,
                    "is_correct": None,
                    "judge_raw": {"method": "skipped"},
                })
            break

    return {
        "candidate_step": candidate_step,
        "is_wrong_step": all_wrong,
        "rollouts": rollout_results,
    }

def find_wrong_next_step(
    question,
    prefix_text,
    prefix_len,
    gold_text,
    gold_answer,
    max_candidates=MAX_CANDIDATES,
    num_rollouts=NUM_ROLLOUTS,
):
    tried_steps = set()
    candidate_results = []

    for attempt in range(max_candidates):
        result = sample_wrong_next_step(
            question=question,
            prefix_text=prefix_text,
            prefix_len=prefix_len,
            gold_text=gold_text,
            gold_answer=gold_answer,
            num_rollouts=num_rollouts,
        )

        candidate_step = result["candidate_step"]

        if candidate_step in tried_steps:
            print(f"      [attempt {attempt+1}] duplicate step, skipping")
            continue

                                                               
        last_char = candidate_step.rstrip()[-1] if candidate_step.rstrip() else ""
        if last_char in "=,+-(:[{":
            print(f"      [attempt {attempt+1}] incomplete step (ends with {last_char!r}), skipping")
            continue

        tried_steps.add(candidate_step)
        candidate_results.append(result)

        correct_count = sum(r["is_correct"] for r in result["rollouts"] if r["is_correct"] is not None)
        print(f"      [attempt {attempt+1}] step: {candidate_step[:80]!r}")
        print(f"               correct rollouts: {correct_count}/{num_rollouts}  |  is_wrong={result['is_wrong_step']}")

        if result["is_wrong_step"]:
            return result, candidate_results

    return None, candidate_results

def build_prefix(gold_solution: str):

    lines = [l.strip() for l in gold_solution.split("\n") if l.strip()]
    step_lines = [l for l in lines if not l.lower().startswith("final answer")]

    max_prefix = max(1, len(step_lines) // 2)
    prefix_len = random.randint(1, max_prefix)
    prefix_steps = step_lines[:prefix_len]
    
    cleaned = [re.sub(r'^Step\s*\d+[:.]*\s*', '', s) for s in prefix_steps]
    prefix_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(cleaned))
    '''
    prefix_text = "\n".join(
        f"Step {i+1}: {re.sub(r'^Step\\s*\\d+[:.]*\\s*', '', step)}"
        for i, step in enumerate(prefix_steps)
    )
    '''
    return prefix_text, prefix_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None, help="Global dataset start index (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="Global dataset end index (exclusive)")
    parser.add_argument("--part",  type=int, default=None, help="Part index for output file naming")
    args = parser.parse_args()

    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else (NUM_DATASET_SAMPLES if NUM_DATASET_SAMPLES is not None else len(dataset))
    end   = min(end, len(dataset))

    dataset = dataset.select(range(start, end))

    if args.part is not None:
        output_file = f"wrong_steps_part_{args.part}.jsonl"
    else:
        output_file = OUTPUT_FILE

    print(f"Dataset range: [{start}, {end})  ({len(dataset)} samples)")
    print(f"Output file  : {output_file}\n")

    output_path = Path(output_file)
    processed_indices = set()
    found_count = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_indices.add(rec["dataset_index"])
                    if rec.get("wrong_step_found"):
                        found_count += 1
                except Exception:
                    pass
        print(f"Resuming — {len(processed_indices)} already processed, {found_count} wrong steps found.\n")

    total = len(dataset)

    with open(output_path, "a") as out_f:
        for local_idx in range(total):
            idx = start + local_idx                         
            if idx in processed_indices:
                continue

            sample = dataset[local_idx]
            question      = sample["query"]
            gold_solution = sample["response"]
            gold_answer   = extract_gold_answer(gold_solution)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Sample {idx+1}/{start+total}  (found so far: {found_count})")
            print(f"   Q: {question[:120]!r}")
            print(f"   Gold answer: {gold_answer!r}")

            if gold_answer is None:
                print("     Could not extract gold answer — skipping.\n")
                continue

            prefix_text, prefix_len = build_prefix(gold_solution)
            print(f"   Prefix length: {prefix_len} step(s)")
            print(f"   Prefix:\n{prefix_text}\n")

            wrong_result, all_candidates = find_wrong_next_step(
                question=question,
                prefix_text=prefix_text,
                prefix_len=prefix_len,
                gold_text=gold_solution,
                gold_answer=gold_answer,
            )

            found = wrong_result is not None
            if found:
                found_count += 1
                print(f"  Wrong step found: {wrong_result['candidate_step']!r}")
            else:
                print(f"  No wrong step found in {MAX_CANDIDATES} attempts.")

            record = {
                "dataset_index": idx,
                "question": question,
                "gold_solution": gold_solution,
                "gold_answer": gold_answer,
                "prefix_text": prefix_text,
                "prefix_len": prefix_len,
                "wrong_step_found": found,
                "wrong_step": wrong_result["candidate_step"] if found else None,
                "all_candidates": [
                    {
                        "candidate_step": c["candidate_step"],
                        "is_wrong_step": c["is_wrong_step"],
                        "rollouts": [
                            {
                                "full_reasoning": r["full_reasoning"],
                                "is_correct": r["is_correct"],
                                "judge_raw": r.get("judge_raw"),
                            }
                            for r in c["rollouts"]
                        ],
                    }
                    for c in all_candidates
                ],
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"   Saved. Total found: {found_count}/{idx+1} processed.\n")

    print(f"\nDone. {found_count} wrong steps found across {total} samples.")
    print(f"Results written to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
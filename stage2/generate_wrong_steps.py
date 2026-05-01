"""
generate_wrong_steps.py

Iterates over MetaMathQA, samples wrong reasoning steps using MCTS-style rollouts,
and saves results to a JSONL file.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import re
import math
import json
import random
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

OUTPUT_FILE            = "wrong_steps_data.jsonl"
NUM_DATASET_SAMPLES    = None   
MAX_CANDIDATES         = 10     
NUM_ROLLOUTS           = 8      
STEP_MAX_NEW_TOKENS    = 64
ROLLOUT_MAX_NEW_TOKENS = 256
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


print("Loading generator model...")
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=judge_device_map,
)
print("Models loaded.")


def normalize_answer_string(s: str) -> str:
    s = s.strip()
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
        if "=" in ans:
            ans = ans.split("=")[-1].strip()
        ans = re.sub(r"^(the answer is[:\s]*)", "", ans, flags=re.IGNORECASE)
        ans = re.sub(r"^(final answer[:\s]*)", "", ans, flags=re.IGNORECASE)
        ans = ans.split("\n")[0].strip()
        ans = re.sub(
            r"\b(inches|inch|feet|foot|ft|cm|mm|meters|meter|miles|mile|dollars|dollar|units?)\b",
            "",
            ans,
            flags=re.IGNORECASE,
        ).strip()
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

    score_a = score_continuation(judge_model, judge_tokenizer, prompt, " A")
    score_b = score_continuation(judge_model, judge_tokenizer, prompt, " B")
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

        pred_answer = extract_final_answer(full_reasoning)
        if pred_answer is not None:
            is_correct = answers_equivalent(pred_answer, gold_answer)
            judge_raw = {"method": "symbolic"}
        else:

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

    prefix_text = "\n".join(
        f"Step {i+1}: {re.sub(r'^Step\\s*\\d+[:.]*\\s*', '', step)}"
        for i, step in enumerate(prefix_steps)
    )
    return prefix_text, prefix_len



def main():
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    if NUM_DATASET_SAMPLES is not None:
        dataset = dataset.select(range(NUM_DATASET_SAMPLES))

    print(f"Dataset size: {len(dataset)}")
    print(f"Output file : {OUTPUT_FILE}\n")

    output_path = Path(OUTPUT_FILE)
    processed_indices = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_indices.add(rec["dataset_index"])
                except Exception:
                    pass
        print(f"Resuming — {len(processed_indices)} already processed.\n")

    total = len(dataset)
    found_count = len(processed_indices)

    with open(output_path, "a") as out_f:
        for idx in range(total):
            if idx in processed_indices:
                continue

            sample = dataset[idx]
            question      = sample["query"]
            gold_solution = sample["response"]
            gold_answer   = extract_gold_answer(gold_solution)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Sample {idx+1}/{total}  (found so far: {found_count})")
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
                
                
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"   Saved. Total found: {found_count}/{idx+1} processed.\n")

    print(f"\nDone. {found_count} wrong steps found across {total} samples.")
    print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

"""
code_contests_data_gen.py

Iterates over the CodeContests dataset, generates gold reasoning traces from GT solutions,
then samples wrong reasoning steps using MCTS-style rollouts evaluated by executing
generated Python code against public test cases.

Two models:
  - Qwen2.5-Coder-14B-Instruct (cuda:0) — gold reasoning generation
  - Qwen2.5-Coder-7B-Instruct  (cuda:1) — wrong step sampling, rollouts
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import random
import argparse
import subprocess
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime


GOLD_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_NAME      = "Qwen/Qwen2.5-Coder-7B-Instruct"

OUTPUT_FILE              = "code_contests_wrong_steps.jsonl"
NUM_DATASET_SAMPLES      = 5000
MAX_CANDIDATES           = 6
NUM_ROLLOUTS             = 12
MAX_PUBLIC_TESTS         = 3      # test cases used per rollout evaluation
CODE_EXEC_TIMEOUT        = 5.0   # seconds per test case

STEP_MAX_NEW_TOKENS      = 128
GOLD_MAX_NEW_TOKENS      = 768
ROLLOUT_MAX_NEW_TOKENS   = 1024  # reasoning continuation + full Python code
STEP_TEMPERATURE         = 1.2
ROLLOUT_TEMPERATURE      = 0.3
GOLD_TEMPERATURE         = 0.3   # low temp — want faithful reasoning from GT code
TOP_P                    = 0.95


if not os.environ.get("ATTRIBUTION_IMPORT"):
    print("Loading gold model (Qwen2.5-Coder-14B) on cuda:0...")
    gold_tokenizer = AutoTokenizer.from_pretrained(GOLD_MODEL_NAME)
    gold_model = AutoModelForCausalLM.from_pretrained(
        GOLD_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print("Gold model loaded on cuda:0.")

    print("Loading rollout model (Qwen2.5-Coder-7B) on cuda:1...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:1",
    )
    print("Rollout model loaded on cuda:1.")
else:
    # Placeholders — will be patched by the attribution script before use
    gold_model, gold_tokenizer, model, tokenizer = None, None, None, None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> sections produced by DeepSeek-R1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_reasoning_steps(text: str) -> list:
    """
    Extract numbered reasoning steps from generated text.
    Splits on newlines and strips any leading numbering (1. / 1) / Step 1:).
    """
    text = strip_think_tags(text)
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        content = re.sub(r"^(?:Step\s*)?\d+[:.)\s]\s*", "", line).strip()
        if not content:
            continue
        steps.append(f"Step {len(steps) + 1}: {content}")
    return steps


def extract_python_code(text: str):
    """Extract Python code from a ```python ... ``` block. Returns None if not found."""
    text = strip_think_tags(text)
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: any fenced code block that looks like Python
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if any(kw in candidate for kw in ("def ", "import ", "print(", "input(", "for ", "while ")):
            return candidate
    return None


def run_single_test(code: str, input_str: str, expected: str) -> bool:
    """Execute code against one test case. Returns True iff stdout matches expected."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT,
        )
        return result.stdout.strip() == expected.strip()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def passes_public_tests(code: str, public_tests: dict) -> bool:
    """Return True iff code passes all (up to MAX_PUBLIC_TESTS) public test cases."""
    inputs  = public_tests.get("input",  [])
    outputs = public_tests.get("output", [])
    n = min(len(inputs), len(outputs), MAX_PUBLIC_TESTS)
    if n == 0:
        return False
    for i in range(n):
        if not run_single_test(code, inputs[i], outputs[i]):
            return False
    return True


def get_gt_solution(sample):
    """
    Extract a GT solution and its language label.
    Prefers Python; falls back to the first available solution.
    Returns (solution_text, language_label).
    """
    solutions  = sample.get("solutions", {})
    languages  = solutions.get("language", [])
    sol_texts  = solutions.get("solution", [])

    # CodeContests language codes: 0=UNKNOWN, 1=PYTHON, 2=CPP, 3=PYTHON3, 4=JAVA
    for lang_code, sol in zip(languages, sol_texts):
        if lang_code in (1, 3):
            return sol, "Python"

    lang_names = {0: "Unknown", 1: "Python", 2: "C++", 3: "Python3", 4: "Java"}
    if sol_texts:
        label = lang_names.get(languages[0], f"lang_{languages[0]}") if languages else "Unknown"
        return sol_texts[0], label

    return "", "Unknown"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate(messages: list, max_new_tokens: int, temperature: float, do_sample: bool = True, add_generation_prompt: bool = True, _model=None, _tokenizer=None) -> str:
    """Generate using the model. Returns decoded text."""
    m = _model     if _model     is not None else model
    t = _tokenizer if _tokenizer is not None else tokenizer

    inputs = t.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(next(m.parameters()).device)
    attention_mask = inputs["attention_mask"].to(input_ids.device)
    prompt_len     = input_ids.shape[1]

    with torch.no_grad():
        out = m.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=TOP_P if do_sample else 1.0,
            pad_token_id=t.eos_token_id,
        )

    decoded = t.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return strip_think_tags(decoded)



def batched_generate(input_ids, attention_mask, num_rollouts: int, **gen_kwargs):
    """
    Generate all rollouts in one batched call.
    Halves batch size on OOM down to 1.
    """
    batch_sizes = sorted({num_rollouts, 4, 2, 1}, reverse=True)

    for batch_size in batch_sizes:
        try:
            results = []
            for start in range(0, num_rollouts, batch_size):
                end = min(start + batch_size, num_rollouts)
                out = model.generate(
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
            print(f"      OOM at batch_size={batch_size}, retrying with {batch_size // 2}...")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def generate_gold_reasoning(problem: str, gt_solution: str) -> list:
    """
    Generate a gold reasoning trace (list of Step N: ... strings) from the
    problem statement and its GT code solution.
    """
    messages = [{"role": "user", "content": f"""You are given a competitive programming problem and its correct solution.
Explain the algorithm as numbered reasoning steps that are specific enough that someone could re-implement the solution from your steps alone.

Rules:
1. Be specific — name the exact values, indices, or conditions the solution uses, not just what category of thing it does
2. Each step should describe a concrete action, not a vague intention (e.g. "check if a[i] + a[i+1] > a[i+2]" not "check the array elements")
3. Do NOT include code syntax
4. Aim for 4 to 15 steps depending on the complexity of the solution
5. Output ONLY the numbered steps — no preamble, no explanation

Problem:
{problem}

Correct solution:
{gt_solution}

Reasoning steps:"""}]

    raw = _generate(messages, max_new_tokens=GOLD_MAX_NEW_TOKENS, temperature=GOLD_TEMPERATURE,
                    _model=gold_model, _tokenizer=gold_tokenizer)
    return parse_reasoning_steps(raw)


def is_substantive_step(candidate_step: str) -> bool:
    """
    Use the gold model to decide whether a candidate step contains meaningful
    algorithmic content, or is only trivial I/O / initialization.

    Constrained to YES/NO: does a single forward pass, compares the summed
    probability mass on YES-variant tokens vs NO-variant tokens, returns True
    if YES wins.
    """
    messages = [{"role": "user", "content": f"""You are reviewing a reasoning step for a competitive programming solution.

Determine whether the step contains meaningful algorithmic content — i.e., it specifies something to compute, a condition to check, a formula to apply, or a non-trivial data structure operation.

Answer YES if the step describes an algorithmic action beyond just reading input, writing output, or trivial initialization (e.g. "set x to 0", "initialize empty list", "declare variable", "read n integers").
Answer NO if the step is only: reading input, writing output, trivial initialization, or declaring a variable/function with no algorithmic content.

Step: {candidate_step}

Answer (YES or NO):"""}]

    inputs = gold_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(next(gold_model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(input_ids.device)

    with torch.no_grad():
        logits = gold_model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :]

    probs = torch.softmax(logits, dim=-1)

    def vocab_prob(words):
        seen = set()
        total = 0.0
        for w in words:
            ids = gold_tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                total += probs[ids[0]].item()
        return total

    yes_prob = vocab_prob(["YES", "Yes", "yes"])
    no_prob  = vocab_prob(["NO",  "No",  "no"])
    return yes_prob > no_prob


def build_prefix(gold_steps: list):
    """
    Sample a prefix of 3 to 2*len(steps)//3 steps from the gold reasoning.
    Returns (prefix_text, prefix_len).
    """
    max_prefix = max(3, 2 * len(gold_steps) // 3)
    prefix_len = random.randint(3, max_prefix)
    prefix_steps = gold_steps[:prefix_len]

    cleaned = []
    for i, step in enumerate(prefix_steps):
        content = re.sub(r"^Step\s*\d+[:.]\s*", "", step).strip()
        cleaned.append(f"Step {i + 1}: {content}")

    return "\n".join(cleaned), prefix_len


def sample_wrong_next_step(
    problem: str,
    prefix_text: str,
    prefix_len: int,
    public_tests: dict,
    num_rollouts: int = NUM_ROLLOUTS,
):
    """
    1. Sample one candidate next reasoning step (high temperature).
    2. Run num_rollouts rollouts from (prefix + candidate step).
    3. Each rollout continues reasoning then generates Python code.
    4. Evaluate code against public test cases.
    5. Return result dict with is_wrong_step=True if ALL rollouts fail.
    """
    # ── Step generation ──────────────────────────────────────────────────
    step_messages = [
        {"role": "user", "content": f"""You are given a competitive programming problem and a correct reasoning prefix.
Generate the next reasoning step after the prefix.

Rules:
1. Output exactly one line, nothing else
2. Use the exact format: Step {prefix_len + 1}: ...
3. The step must be a concrete algorithmic action (what to compute, check, or do next)
4. Do not generate any further steps
5. Do not generate code
6. Do not add any preamble or commentary

Problem:
{problem}

Correct reasoning prefix:
{prefix_text}"""},
    ]

    step_raw = _generate(step_messages, max_new_tokens=STEP_MAX_NEW_TOKENS, temperature=STEP_TEMPERATURE)

    # Extract first valid Step N: line from output
    step_lines = [l.strip() for l in step_raw.split("\n") if l.strip()]
    candidate_step = next(
        (l for l in step_lines if re.match(r"^Step\s*\d+[:.]\s*.+", l)),
        step_lines[0] if step_lines else f"Step {prefix_len + 1}: (empty)"
    )

    # ── Substantive step gate ────────────────────────────────────────────
    if not is_substantive_step(candidate_step):
        return {
            "candidate_step": candidate_step,
            "is_wrong_step":  False,
            "gate_rejected":  True,
            "rollouts":       [],
        }

    fixed_reasoning = prefix_text + "\n" + candidate_step

    # ── Rollout generation ───────────────────────────────────────────────
    rollout_messages = [{"role": "user", "content": f"""You are given a competitive programming problem and a partial reasoning trace.
First complete all remaining reasoning steps, then write a full Python solution.

Rules:
1. Continue the reasoning steps in the format: Step N: ... until the reasoning is complete
2. Do not write any code until all reasoning steps are complete
3. After all reasoning steps, write the complete Python solution between ```python and ``` tags
4. The code must read from stdin and write to stdout
5. Do not write anything after the closing ``` tag

Problem:
{problem}

Current reasoning:
{fixed_reasoning}"""}]

    rollout_inputs = tokenizer.apply_chat_template(
        rollout_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    input_ids      = rollout_inputs["input_ids"].to(device).repeat(num_rollouts, 1)
    attention_mask = rollout_inputs["attention_mask"].to(device).repeat(num_rollouts, 1)
    rollout_prompt_len = rollout_inputs["input_ids"].shape[1]

    rollout_outputs = batched_generate(
        input_ids,
        attention_mask,
        num_rollouts,
        max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=ROLLOUT_TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )

    # ── Evaluate rollouts ────────────────────────────────────────────────
    rollout_results = []
    all_wrong = True

    for i in range(num_rollouts):
        generated_tokens = rollout_outputs[i][rollout_prompt_len:]
        continuation = strip_think_tags(
            tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        )

        code   = extract_python_code(continuation)
        passes = False

        if code is not None:
            passes = passes_public_tests(code, public_tests)

        rollout_results.append({
            "continuation":   continuation,
            "code":           code,
            "passes_tests":   passes,
            "code_extracted": code is not None,
        })

        if passes:
            all_wrong = False
            # Skip remaining rollouts — not needed once one passes
            for _ in range(i + 1, num_rollouts):
                rollout_results.append({
                    "continuation":   None,
                    "code":           None,
                    "passes_tests":   None,
                    "code_extracted": None,
                    "skipped":        True,
                })
            break

    return {
        "candidate_step": candidate_step,
        "is_wrong_step":  all_wrong,
        "rollouts":       rollout_results,
    }


def find_wrong_next_step(
    problem: str,
    prefix_text: str,
    prefix_len: int,
    public_tests: dict,
    max_candidates: int = MAX_CANDIDATES,
    num_rollouts: int   = NUM_ROLLOUTS,
):
    """Try up to max_candidates candidate steps; return the first confirmed wrong step."""
    tried_steps      = set()
    candidate_results = []

    for attempt in range(max_candidates):
        result         = sample_wrong_next_step(problem, prefix_text, prefix_len, public_tests, num_rollouts)
        candidate_step = result["candidate_step"]

        if candidate_step in tried_steps:
            print(f"      [attempt {attempt + 1}] duplicate step — skipping")
            continue

        # Discard steps that don't follow the Step N: format
        if not re.match(r"^Step\s*\d+[:.]\s*.+", candidate_step):
            print(f"      [attempt {attempt + 1}] not a valid Step N: line — skipping: {candidate_step[:60]!r}")
            continue

        # Discard deliberation text — content after "Step N:" starts with meta-commentary
        step_content = re.sub(r"^Step\s*\d+[:.]\s*", "", candidate_step).strip()
        deliberation_phrases = ("okay", "alright", "let me", "let's", "so i", "i need", "i'm", "i am", "so let", "now i", "first,", "well,")
        if step_content.lower().startswith(deliberation_phrases):
            print(f"      [attempt {attempt + 1}] deliberation text — skipping: {candidate_step[:60]!r}")
            continue

        # Discard obviously truncated steps
        last_char = candidate_step.rstrip()[-1] if candidate_step.rstrip() else ""
        if last_char in "=,+-(:[{":
            print(f"      [attempt {attempt + 1}] incomplete step (ends with {last_char!r}) — skipping")
            continue

        # Discard trivial steps rejected by the substantive gate
        if result.get("gate_rejected"):
            print(f"      [attempt {attempt + 1}] gate rejected (trivial step) — skipping: {candidate_step[:80]!r}")
            continue

        tried_steps.add(candidate_step)
        candidate_results.append(result)

        passing = sum(1 for r in result["rollouts"] if r.get("passes_tests") is True)
        print(f"      [attempt {attempt + 1}] step: {candidate_step[:80]!r}")
        print(f"               passing rollouts: {passing}/{num_rollouts}  |  is_wrong={result['is_wrong_step']}")

        if result["is_wrong_step"]:
            return result, candidate_results

    return None, candidate_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None, help="Dataset start index (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="Dataset end index (exclusive)")
    parser.add_argument("--part",  type=int, default=None, help="Part index for output file naming")
    args = parser.parse_args()

    print("Loading CodeContests dataset...")
    dataset = load_dataset("deepmind/code_contests", split="train")

    # For testing: pre-select a small pool before filtering to avoid scanning all 19GB.
    # 3x NUM_DATASET_SAMPLES gives enough headroom after filtering out hard/incomplete problems.
    POOL_SIZE = min(NUM_DATASET_SAMPLES * 3, len(dataset))
    dataset = dataset.select(range(POOL_SIZE))

    # Keep only A/B/C difficulty problems (Codeforces-style) with public tests + solutions.
    # Difficulty enum: A=7, B=8, C=9. Filtering to these avoids overwhelming the 7B model
    # with hard problems that produce noisy/always-failing rollouts.
    EASY_DIFFICULTIES = {7, 8}  # A, B
    dataset = dataset.filter(
        lambda x: (
            x["difficulty"] in EASY_DIFFICULTIES and
            len(x["public_tests"]["input"]) > 0 and
            len(x["solutions"]["solution"]) > 0
        )
    )
    print(f"After filtering (A/B/C difficulty, public tests, solutions): {len(dataset)} problems.")

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else min(NUM_DATASET_SAMPLES, len(dataset))
    end   = min(end, len(dataset))
    dataset = dataset.select(range(start, end))

    output_file = f"code_contests_wrong_steps_part_{args.part}.jsonl" if args.part is not None else OUTPUT_FILE

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

    with open(output_path, "a") as out_f:
        for local_idx in range(len(dataset)):
            idx = start + local_idx
            if idx in processed_indices:
                continue

            sample       = dataset[local_idx]
            problem      = sample["description"]
            public_tests = sample["public_tests"]
            problem_name = sample.get("name", f"problem_{idx}")
            gt_solution, gt_lang = get_gt_solution(sample)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Sample {idx + 1}/{start + len(dataset)}  (found so far: {found_count})")
            print(f"   Problem: {problem_name!r}")
            print(f"   GT language: {gt_lang} | Public tests: {len(public_tests['input'])}")

            if not gt_solution:
                print("   No GT solution — skipping.\n")
                continue

            # ── Stage 1: Generate gold reasoning ─────────────────────────
            print("   Generating gold reasoning...")
            gold_steps = generate_gold_reasoning(problem, gt_solution)

            if len(gold_steps) < 4:
                print(f"   Only {len(gold_steps)} reasoning step(s) extracted — skipping.\n")
                continue

            gold_reasoning = "\n".join(gold_steps)
            print(f"   Gold steps ({len(gold_steps)}):")
            for s in gold_steps:
                print(f"     {s}")

            # ── Stage 2: Build prefix ─────────────────────────────────────
            prefix_text, prefix_len = build_prefix(gold_steps)
            print(f"   Prefix ({prefix_len} step(s)):\n{prefix_text}")

            # ── Stage 3: Find wrong step ──────────────────────────────────
            wrong_result, all_candidates = find_wrong_next_step(
                problem=problem,
                prefix_text=prefix_text,
                prefix_len=prefix_len,
                public_tests=public_tests,
            )

            found = wrong_result is not None
            if found:
                found_count += 1
                print(f"   Wrong step found: {wrong_result['candidate_step']!r}")
            else:
                print(f"   No wrong step found in {MAX_CANDIDATES} attempts.")

            record = {
                "dataset_index":        idx,
                "problem_name":         problem_name,
                "problem":              problem,
                "gt_solution":          gt_solution,
                "gt_solution_language": gt_lang,
                "gold_reasoning":       gold_reasoning,
                "gold_reasoning_steps": gold_steps,
                "prefix_text":          prefix_text,
                "prefix_len":           prefix_len,
                "num_public_tests":     len(public_tests["input"]),
                "wrong_step_found":     found,
                "wrong_step":           wrong_result["candidate_step"] if found else None,
                "all_candidates": [
                    {
                        "candidate_step": c["candidate_step"],
                        "is_wrong_step":  c["is_wrong_step"],
                        "rollouts": [
                            {
                                "continuation":   r["continuation"],
                                "code":           r["code"],
                                "passes_tests":   r["passes_tests"],
                                "code_extracted": r.get("code_extracted"),
                                "skipped":        r.get("skipped", False),
                            }
                            for r in c["rollouts"]
                        ],
                    }
                    for c in all_candidates
                ],
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"   Saved. Total found: {found_count}/{local_idx + 1} processed.\n")

    print(f"\nDone. {found_count} wrong steps found.")
    print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

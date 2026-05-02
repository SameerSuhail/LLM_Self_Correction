"""Evaluate models on HumanEval, MBPP, and Codeforces/CodeContests."""
import argparse
import json
import os
import sys
from pathlib import Path
import re
import tempfile
import subprocess
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def check_correctness(code, prompt, test, entry_point=None, dataset_name=""):
    """Sandboxed execution for HumanEval, MBPP, and CodeContests.
    Returns (bool, str) where str contains execution details.
    """
    dataset_name_lower = dataset_name.lower()
    details = []
    
    if "codecontests" in dataset_name_lower or "codeforces" in dataset_name_lower:
        try:
            test_cases = json.loads(test)
        except Exception as e:
            msg = f"  [CF-DEBUG] Could not parse test JSON: {e}"
            return False, msg
        
        if not test_cases:
            msg = f"  [CF-DEBUG] No public test cases for this problem — skipping (marking pass)"
            return True, msg
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            f.write(code)
            fname = f.name
            
        try:
            for tc_idx, tc in enumerate(test_cases):
                input_str = tc['input']
                expected_output = tc['output'].strip()
                
                try:
                    result = subprocess.run(
                        ['python3', fname],
                        input=input_str,
                        capture_output=True, text=True, timeout=10,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
                    )
                except subprocess.TimeoutExpired:
                    msg = f"Test case {tc_idx}: TIMEOUT"
                    details.append(msg)
                    return False, "\n".join(details)
                
                if result.returncode != 0:
                    msg = f"Test case {tc_idx}: RUNTIME ERROR\nError: {result.stderr.strip()}"
                    details.append(msg)
                    return False, "\n".join(details)
                
                actual = result.stdout.strip()
                if actual != expected_output:
                    msg = (
                        f"Test case {tc_idx}: WRONG OUTPUT\n"
                        f"  Input:    {input_str.strip()!r}\n"
                        f"  Expected: {expected_output!r}\n"
                        f"  Got:      {actual!r}"
                    )
                    details.append(msg)
                    return False, "\n".join(details)
            return True, "All test cases passed."
        except Exception as e:
            return False, f"Unexpected error: {e}"
        finally:
            try:
                os.unlink(fname)
            except:
                pass

    # Function-level execution (HumanEval/MBPP)
    has_signature = False
    if entry_point:
        pattern = rf"def\s+{re.escape(entry_point)}\s*\("
        if re.search(pattern, code):
            has_signature = True

    if "mbpp" in dataset_name_lower:
        full_code = code + "\n" + test
    else:
        if has_signature:
            full_code = code + "\n" + test + f"\ncheck({entry_point})"
        else:
            full_code = prompt + "\n" + code + "\n" + test + (f"\ncheck({entry_point})" if entry_point else "")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(full_code)
        fname = f.name

    try:
        result = subprocess.run(
            ['python3', fname],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        )
        if result.returncode == 0:
            return True, "Passed."
        else:
            return False, f"RUNTIME ERROR\nStdout: {result.stdout}\nStderr: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"Unexpected error: {e}"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def build_prompt(instruction, tokenizer, dataset_name=""):
    # IMPORTANT: This must match the training prompt format exactly.
    # Matches training text format assembled by scripts/data_generation/assemble_sft_jsonl.py
    #   <|im_start|>user\nProblem:\n{problem}\n\nSolve step by step.<|im_end|>
    
    # EXACT training prompt format
    prompt_text = f"Problem:\n{instruction}\n\nSolve step by step."
    
    messages = [
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def python2_to_3_shim(code, verbose=False):
    """Add aliases and fixes for common Python 2 hallucinations using lib2to3 and heuristics."""
    if not code:
        return ""
        
    # 1. Heuristic fixes for LLM-specific failure patterns
    # Fix eval(input(...)) which fails on strings in Python 3
    # Matches eval(input()), eval(input(" ")), eval(raw_input()), etc.
    if verbose:
        print(f"DEBUG: Before Fix 1: {code}")
    code = re.sub(r'eval\s*\(\s*(?:raw_)?input\s*\((.*?)\)\s*\)', r'input(\1)', code)
    if verbose:
        print(f"DEBUG: After Fix 1: {code}")
    # Fix int(eval(input(...)))
    code = re.sub(r'int\s*\(\s*eval\s*\(\s*(?:raw_)?input\s*\((.*?)\)\s*\)\s*\)', r'int(input(\1))', code)
    
    # Fix starred expression typos like print((*res))
    # Matches print((*res)), print( (*res) ), etc.
    code = re.sub(r'print\s*\(\s*\(\s*\*(.*?)\s*\)\s*\)', r'print(*\1)', code)
    
    # Fix legacy sys.maxint
    code = code.replace('sys.maxint', 'sys.maxsize')
    code = re.sub(r'\bmaxint\b', 'maxsize', code)
        
    # 2. Add common aliases at the top
    shim = [
        "import sys",
        "import math",
        "if sys.version_info[0] >= 3:",
        "    try:",
        "        raw_input = input",
        "        xrange = range",
        "        # Handle missing maxint in sys",
        "        if not hasattr(sys, 'maxint'): sys.maxint = sys.maxsize",
        "    except NameError: pass",
        ""
    ]
    
    # 3. Use lib2to3 for robust conversion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_fname = f.name
    
    try:
        subprocess.run(
            [sys.executable, "-m", "lib2to3", "-w", "-n", temp_fname],
            capture_output=True, text=True, timeout=10
        )
        with open(temp_fname, 'r') as f:
            converted_code = f.read()
    except Exception as e:
        if verbose:
            print(f"  [DEBUG] lib2to3 conversion failed: {e}")
        converted_code = code
    finally:
        try:
            os.unlink(temp_fname)
        except:
            pass
            
    return '\n'.join(shim) + converted_code

def extract_code(text, entry_point=None, verbose=False):
    """Aggressive code island extraction.
    Finds markdown blocks or islands between markers.
    """
    if not text:
        return ""

    # 1. Try to extract from Markdown blocks first (the most reliable)
    # Skip blocks explicitly labeled as non-python (like cpp)
    md_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    code = ""
    for lang, content in md_blocks:
        lang = (lang or "").lower()
        if lang in ["python", "py", ""]:
            code = content.strip()
            break
            
    if not code:
        # 2. Look for markers like "### Code:" or "Here is the correct code:"
        marker_match = re.search(r'(?:### Code:|Here is the correct code:)\s*\n(.*?)(?:\n### End|$)', text, re.DOTALL | re.IGNORECASE)
        if marker_match:
            code = marker_match.group(1).strip()
        else:
            # 3. Fallback to island extraction logic
            lines = text.split('\n')
            # CP scripts start with imports, from, assignments, or defs.
            # We omit '#' to avoid picking up C++ comments/includes.
            code_start_patterns = [
                r'^import\s+', r'^from\s+', r'^def\s+', r'^class\s+',
                r'^[a-z0-9_,\s]+=[^=]', r'^if\s+__name__',
                r'^try\:', r'^with\s+', r'^while\s+', r'^for\s+'
            ]
            
            first_idx = -1
            last_idx = -1
            extracted_lines = []
            
            for i, line in enumerate(lines):
                l_strip = line.strip()
                if not l_strip:
                    if first_idx != -1:
                        extracted_lines.append(line)
                    continue
                
                # Ignore reasoning steps and markdown markers as START candidates
                is_prose = re.match(r'^(Step \d+:|Problem:|Note:|###|Analysis:|Diagnosis:)', l_strip, re.IGNORECASE)
                is_md = l_strip.startswith('```')
                
                if is_prose or is_md:
                    if first_idx != -1:
                        # Stop if we hit prose or a new block marker
                        break
                    continue
                    
                # Check if this line looks like Python code
                if any(re.match(p, l_strip) for p in code_start_patterns):
                    if first_idx == -1:
                        first_idx = i
                    last_idx = i
                    extracted_lines.append(line)
                elif first_idx != -1:
                    # Once we've started, we continue until we hit a prose break
                    last_idx = i
                    extracted_lines.append(line)

            if extracted_lines:
                # Trim trailing empty lines
                while extracted_lines and not extracted_lines[-1].strip():
                    extracted_lines.pop()
                code = '\n'.join(extracted_lines)
            else:
                return ""

    # Final cleanup: dedent and apply shim
    code = textwrap.dedent(code).strip()
    code = python2_to_3_shim(code, verbose=verbose)
    
    return code



def load_data(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    tasks = []
    if "human-eval" in dataset_name.lower() or "humaneval" in dataset_name.lower():
        try:
            ds = load_dataset("openai_humaneval", split="test")
        except:
            ds = load_dataset("openai/human-eval", split="test")
        for item in ds:
            tasks.append({
                'id': item['task_id'],
                'prompt': item['prompt'],
                'test': item['test'],
                'entry_point': item['entry_point']
            })
    elif "mbpp" in dataset_name.lower():
        try:
            ds = load_dataset("google-research-datasets/mbpp", split="test")
        except:
            ds = load_dataset("mbpp", split="test")
        for item in ds:
            desc = item['text']
            test_setup = item.get('test_setup_code', '')
            test_list = item['test_list']
            
            # Extract expected function name from the first test case
            entry_point = None
            if test_list:
                # Matches 'assert func_name('
                m = re.search(r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test_list[0])
                if m:
                    entry_point = m.group(1)
            
            # Prompt instructs the model to output in a labeled section format:
            #   ### Code:\n<code>\n### End
            # This is easier to parse than JSON (no escaping) and more reliable than
            # markdown (no ambiguity with prose code snippets).
            # The prompt still uses the "Problem: ... Solve step by step." prefix
            # so LoRA models (trained on that format) can also follow it.
            if entry_point:
                prompt = (
                    f"Write a python function named `{entry_point}` that solves: "
                    f"{desc}. "
                    f"The function must accept arguments directly (do NOT use input()). "
                    f"After your reasoning, place ONLY the final function code in this exact format:\n"
                    f"### Code:\n<your code here>\n### End"
                )
            else:
                prompt = (
                    f"{desc}. "
                    f"The function must accept arguments directly (do NOT use input()). "
                    f"After your reasoning, place ONLY the final function code in this exact format:\n"
                    f"### Code:\n<your code here>\n### End"
                )

            tasks.append({
                'id': str(item['task_id']),
                'prompt': prompt,
                'test': test_setup + "\n" + "\n".join(test_list),
                'entry_point': entry_point
            })
    elif "codecontests" in dataset_name.lower() or "codeforces" in dataset_name.lower():
        # Loading CodeContests (Codeforces) dataset.
        # We use streaming to avoid downloading the entire 25GB dataset.
        ds = load_dataset("deepmind/code_contests", split="test", streaming=True)
        
        is_task_a_only = "codeforces-a" in dataset_name.lower()
        count = 0
        for item in ds:
            # If codeforces-a, filter for Task A problems
            if is_task_a_only and item.get('cf_index') != 'A':
                continue
                
            # Limiting to 50 for a reasonable eval time
            if count >= 50: break
            
            # CodeContests tests are in public_tests
            test_cases = []
            if 'public_tests' in item:
                for inp, out in zip(item['public_tests']['input'], item['public_tests']['output']):
                    test_cases.append({'input': inp, 'output': out})
            
            tasks.append({
                'id': item['name'],
                'prompt': item['description'],
                'test': json.dumps(test_cases), # Store as JSON string
                'entry_point': None
            })
            count += 1
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to base/baseline model")
    parser.add_argument('--adapter', default=None, help="Path to LoRA adapter (if any)")
    parser.add_argument('--dataset', required=True, 
                        help="Dataset name (openai/human-eval, mbpp, codeforces, codeforces-a)")
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=1536)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Extra stderr diagnostics (failed tasks, shim debug)',
    )
    args = parser.parse_args()

    cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)

    tasks = load_data(args.dataset)
    if not tasks:
        print(f"Skipping evaluation for {args.dataset} due to empty tasks.")
        return
        
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model in bfloat16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    
    if args.adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        
    model.eval()

    print(f"Evaluating on {len(tasks)} problems from {args.dataset}")

    n_pass = 0
    per_item = []
    trial_log = []

    for i, item in enumerate(tasks):
        prompt_str = build_prompt(item['prompt'], tokenizer, args.dataset)
        inputs = tokenizer(prompt_str, return_tensors='pt', truncation=True,
                           max_length=3072).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        code = extract_code(text, item.get('entry_point'), verbose=args.verbose)
        passed, details = check_correctness(code, item['prompt'], item['test'], item.get('entry_point'), args.dataset)

        # Log for humans/LLMs
        trial_entry = [
            "="*80,
            f"TASK ID: {item['id']}",
            f"INDEX: {i}",
            "-"*40,
            "PROBLEM DESCRIPTION:",
            item['prompt'],
            "-"*40,
            "FULL MODEL RESPONSE:",
            text,
            "-"*40,
            "EXTRACTED CODE:",
            code,
            "-"*40,
            "EXECUTION RESULTS:",
            details,
            f"RESULT: {'PASSED' if passed else 'FAILED'}",
            "="*80,
            "\n"
        ]
        trial_log.append("\n".join(trial_entry))

        if not passed and args.verbose:
            print(f"  [DEBUG] Task {item['id']} failed.")
            if len(details) < 200:
                print(f"    Info: {details}")

        per_item.append({
            'idx': i,
            'task_id': item['id'],
            'response': text,
            'extracted_code': code,
            'passed': passed,
            'details': details,
        })

        n_pass += int(passed)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] pass@1={n_pass/(i+1)*100:.1f}%")

    n = len(tasks)
    results = {
        'model_path': args.model_path,
        'adapter': args.adapter,
        'dataset': args.dataset,
        'n': n,
        'pass_at_1': n_pass / n if n else 0,
        'n_passed': n_pass,
        'per_item': per_item,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed TXT report
    report_path = out_path.with_suffix('.report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(trial_log))

    print(f"=== Results for {args.dataset} ===")
    print(f"  Pass@1: {results['pass_at_1']*100:.2f}% ({n_pass}/{n})")
    print(f"  JSON results: {out_path}")
    print(f"  Detailed report: {report_path}")

if __name__ == '__main__':
    main()

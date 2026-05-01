"""Smoke test: run 5 questions per model to verify the full pipeline works.

Usage:
    python scripts/smoke_test.py --model llama
    python scripts/smoke_test.py --model mistral
    python scripts/smoke_test.py --model llama --no-vllm
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference import ModelWrapper, build_initial_prompt, build_correction_prompt
from src.evaluation import check_gsm8k, check_triviaqa, check_strategyqa, classify_correction
from src.confidence_parser import parse_confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['llama', 'mistral', 'qwen'])
    parser.add_argument('--no-vllm', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Smoke test for {args.model}")
    print(f"{'='*60}\n")

    # Load model
    t0 = time.time()
    model = ModelWrapper(args.model, use_vllm=not args.no_vllm)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Load 5 GSM8K questions
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(data_dir, 'gsm8k_test.json')) as f:
        gsm8k = json.load(f)[:5]

    # Test 1: Basic generation
    print("--- Test 1: Basic generation (GSM8K) ---")
    for item in gsm8k:
        messages = build_initial_prompt('gsm8k', item['question'])
        prompt = model.format_chat(messages)
        output = model.generate([prompt], temperature=0.0)[0][0]
        correct = check_gsm8k(output, item['answer'])
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] Q: {item['question'][:60]}...")
        print(f"           A: {output[:100]}...")
        print()

    # Test 2: Correction (S1)
    print("--- Test 2: S1 correction (first 2 questions) ---")
    for item in gsm8k[:2]:
        messages_y0 = build_initial_prompt('gsm8k', item['question'])
        prompt_y0 = model.format_chat(messages_y0)
        y0 = model.generate([prompt_y0], temperature=0.0)[0][0]
        y0_correct = check_gsm8k(y0, item['answer'])

        corr_msgs = build_correction_prompt('s1', 'gsm8k', item['question'], y0)
        full_msgs = messages_y0 + [{'role': 'assistant', 'content': y0}] + corr_msgs
        prompt_y1 = model.format_chat(full_msgs)
        y1 = model.generate([prompt_y1], temperature=0.0)[0][0]
        y1_correct = check_gsm8k(y1, item['answer'])

        cat = classify_correction(y0_correct, y1_correct)
        print(f"  [{cat}] Q: {item['question'][:60]}...")
        print()

    # Test 3: Confidence parsing (S4)
    print("--- Test 3: Confidence score (first question) ---")
    item = gsm8k[0]
    messages_y0 = build_initial_prompt('gsm8k', item['question'])
    prompt_y0 = model.format_chat(messages_y0)
    y0 = model.generate([prompt_y0], temperature=0.0)[0][0]

    conf_msgs = build_correction_prompt('s4_confidence', 'gsm8k', item['question'], y0)
    full = messages_y0 + [{'role': 'assistant', 'content': y0}] + conf_msgs
    prompt_conf = model.format_chat(full)
    conf_raw = model.generate([prompt_conf], temperature=0.0)[0][0]
    confidence = parse_confidence(conf_raw)
    print(f"  Raw confidence output: '{conf_raw[:100]}'")
    print(f"  Parsed confidence: {confidence}/10")
    print()

    # Test 4: Self-consistency (N=3)
    print("--- Test 4: Self-consistency (first question, N=3) ---")
    messages_cot = build_initial_prompt('gsm8k', item['question'], cot=True)
    prompt_cot = model.format_chat(messages_cot)
    outputs = model.generate([prompt_cot], temperature=0.7, n=3)[0]
    for i, out in enumerate(outputs):
        correct = check_gsm8k(out, item['answer'])
        print(f"  Sample {i+1}: {'CORRECT' if correct else 'WRONG'} - {out[:80]}...")
    print()

    # Test 5: TriviaQA
    print("--- Test 5: TriviaQA (2 questions) ---")
    with open(os.path.join(data_dir, 'triviaqa_test.json')) as f:
        triviaqa = json.load(f)[:2]
    for item in triviaqa:
        messages = build_initial_prompt('triviaqa', item['question'])
        prompt = model.format_chat(messages)
        output = model.generate([prompt], temperature=0.0)[0][0]
        correct = check_triviaqa(output, item['answer'])
        print(f"  [{'CORRECT' if correct else 'WRONG'}] Q: {item['question'][:60]}...")
        print(f"           A: {output[:80]}...")
        print(f"           Expected: {item['answer']['value']}")
        print()

    print(f"{'='*60}")
    print("Smoke test complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

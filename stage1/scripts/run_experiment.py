"""Main experiment runner: runs a single (model, dataset, strategy) condition.

Usage:
    python scripts/run_experiment.py \
        --model llama --dataset gsm8k --strategy s1 \
        --output results/raw/llama_gsm8k_s1.json

Strategies: baseline, cot, self_consistency, s1, s2, s3, s4, s5
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference import ModelWrapper, build_initial_prompt, build_correction_prompt
from src.evaluation import (
    check_gsm8k, check_triviaqa, check_strategyqa, check_humaneval,
    classify_correction, compute_correction_metrics
)
from src.confidence_parser import parse_confidence


def load_dataset_items(dataset_name):
    """Load preprocessed dataset from JSON."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    path = os.path.join(data_dir, f'{dataset_name}_test.json')
    with open(path) as f:
        return json.load(f)


def check_answer(dataset_name, predicted, item):
    """Check if predicted answer is correct for the given dataset."""
    if dataset_name == 'gsm8k':
        return check_gsm8k(predicted, item['answer'])
    elif dataset_name == 'triviaqa':
        return check_triviaqa(predicted, item['answer'])
    elif dataset_name == 'strategyqa':
        return check_strategyqa(predicted, item['answer'])
    elif dataset_name == 'humaneval':
        return check_humaneval(predicted, item['prompt'], item['test'], item['entry_point'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_baseline(model, dataset_name, items, cot=False):
    """Run no-correction baseline (or CoT baseline)."""
    results = []
    prompts = []
    for item in items:
        question = item.get('prompt', item.get('question', ''))
        messages = build_initial_prompt(dataset_name, question, cot=cot)
        prompts.append(model.format_chat(messages))

    # Batch inference
    print(f"  Generating {len(prompts)} initial responses...")
    all_outputs = model.generate(prompts, temperature=0.0)

    for item, outputs in zip(items, all_outputs):
        y0 = outputs[0]
        y0_correct = check_answer(dataset_name, y0, item)
        results.append({
            'id': item['id'],
            'question': item.get('prompt', item.get('question', '')),
            'y0': y0,
            'y0_correct': y0_correct,
        })

    return results


def run_self_consistency(model, dataset_name, items, n=3):
    """Run self-consistency baseline with N independent samples."""
    results = []
    prompts = []
    for item in items:
        question = item.get('prompt', item.get('question', ''))
        messages = build_initial_prompt(dataset_name, question, cot=True)
        prompts.append(model.format_chat(messages))

    print(f"  Generating {len(prompts)} × {n} samples for self-consistency...")
    all_outputs = model.generate(prompts, temperature=0.7, n=n)

    for item, outputs in zip(items, all_outputs):
        # Majority vote
        votes = []
        for resp in outputs:
            correct = check_answer(dataset_name, resp, item)
            votes.append(correct)

        # For self-consistency, we pick the most common answer
        # Since we're checking correctness directly, majority of correct = correct
        majority_correct = sum(votes) > len(votes) / 2

        results.append({
            'id': item['id'],
            'question': item.get('prompt', item.get('question', '')),
            'responses': outputs,
            'votes': votes,
            'majority_correct': majority_correct,
            'n_correct': sum(votes),
            'n_total': len(votes),
        })

    return results


def run_correction_strategy(model, dataset_name, items, strategy, threshold=5):
    """Run a correction strategy (S1-S5) on all items.

    For S3 (iterative), runs up to 3 correction rounds.
    For S4 (confidence-gated), only corrects if confidence < threshold.
    For S5 (explain-then-verify), runs two-step correction.
    """
    results = []

    # Step 1: Generate initial responses (y0)
    prompts_y0 = []
    for item in items:
        question = item.get('prompt', item.get('question', ''))
        messages = build_initial_prompt(dataset_name, question)
        prompts_y0.append(model.format_chat(messages))

    print(f"  Generating {len(prompts_y0)} initial responses (y0)...")
    all_y0 = model.generate(prompts_y0, temperature=0.0)

    # Step 2: Apply correction strategy
    if strategy == 's3':
        # Iterative: up to 3 rounds
        return _run_iterative(model, dataset_name, items, all_y0, max_rounds=3)
    elif strategy == 's4':
        return _run_confidence_gated(model, dataset_name, items, all_y0, threshold)
    elif strategy == 's5':
        return _run_explain_verify(model, dataset_name, items, all_y0)
    else:
        # S1 or S2: single correction round
        return _run_single_correction(model, dataset_name, items, all_y0, strategy)


def _run_single_correction(model, dataset_name, items, all_y0, strategy):
    """Run a single correction round (S1 or S2)."""
    results = []
    prompts_y1 = []

    for item, y0_list in zip(items, all_y0):
        y0 = y0_list[0]
        question = item.get('prompt', item.get('question', ''))
        messages_y0 = build_initial_prompt(dataset_name, question)
        messages_corr = build_correction_prompt(strategy, dataset_name, question, y0)
        # Build full conversation: initial Q + y0 + correction prompt
        full_messages = messages_y0 + [{'role': 'assistant', 'content': y0}] + messages_corr
        prompts_y1.append(model.format_chat(full_messages))

    print(f"  Generating {len(prompts_y1)} corrections ({strategy})...")
    all_y1 = model.generate(prompts_y1, temperature=0.0)

    for item, y0_list, y1_list in zip(items, all_y0, all_y1):
        y0, y1 = y0_list[0], y1_list[0]
        y0_correct = check_answer(dataset_name, y0, item)
        y1_correct = check_answer(dataset_name, y1, item)
        category = classify_correction(y0_correct, y1_correct)

        results.append({
            'id': item['id'],
            'question': item.get('prompt', item.get('question', '')),
            'y0': y0,
            'y1': y1,
            'y0_correct': y0_correct,
            'y1_correct': y1_correct,
            'category': category,
        })

    return results


def _run_iterative(model, dataset_name, items, all_y0, max_rounds=3):
    """Run iterative correction (S3) for up to max_rounds."""
    results = []

    for item, y0_list in zip(items, all_y0):
        question = item.get('prompt', item.get('question', ''))
        y0 = y0_list[0]
        y0_correct = check_answer(dataset_name, y0, item)

        rounds = [{'response': y0, 'correct': y0_correct}]
        current_response = y0
        messages_so_far = build_initial_prompt(dataset_name, question) + [
            {'role': 'assistant', 'content': y0}
        ]

        for r in range(max_rounds):
            corr_messages = build_correction_prompt('s3', dataset_name, question, current_response)
            full_messages = messages_so_far + corr_messages
            prompt = model.format_chat(full_messages)
            output = model.generate([prompt], temperature=0.0)[0][0]

            correct = check_answer(dataset_name, output, item)
            rounds.append({'response': output, 'correct': correct})

            messages_so_far = messages_so_far + corr_messages + [
                {'role': 'assistant', 'content': output}
            ]
            current_response = output

        # Final y1 = last round's response
        y1 = rounds[-1]['response']
        y1_correct = rounds[-1]['correct']
        category = classify_correction(y0_correct, y1_correct)

        results.append({
            'id': item['id'],
            'question': question,
            'y0': y0,
            'y1': y1,
            'y0_correct': y0_correct,
            'y1_correct': y1_correct,
            'category': category,
            'rounds': rounds,
        })

    print(f"  Completed iterative correction for {len(results)} items")
    return results


def _run_confidence_gated(model, dataset_name, items, all_y0, threshold=5):
    """Run confidence-gated correction (S4)."""
    results = []

    # First get confidence scores for all items
    conf_prompts = []
    for item, y0_list in zip(items, all_y0):
        y0 = y0_list[0]
        question = item.get('prompt', item.get('question', ''))
        messages_y0 = build_initial_prompt(dataset_name, question)
        conf_messages = build_correction_prompt('s4_confidence', dataset_name, question, y0)
        full = messages_y0 + [{'role': 'assistant', 'content': y0}] + conf_messages
        conf_prompts.append(model.format_chat(full))

    print(f"  Getting confidence scores for {len(conf_prompts)} items...")
    conf_outputs = model.generate(conf_prompts, temperature=0.0)

    # Now correct only low-confidence items
    to_correct = []  # (index, item, y0)
    for idx, (item, y0_list, conf_list) in enumerate(zip(items, all_y0, conf_outputs)):
        y0 = y0_list[0]
        confidence = parse_confidence(conf_list[0])

        if confidence < threshold:
            to_correct.append((idx, item, y0, confidence))

    # Generate corrections for low-confidence items
    correction_results = {}
    if to_correct:
        corr_prompts = []
        for idx, item, y0, conf in to_correct:
            question = item.get('prompt', item.get('question', ''))
            messages_y0 = build_initial_prompt(dataset_name, question)
            corr_messages = build_correction_prompt('s1', dataset_name, question, y0)
            full = messages_y0 + [{'role': 'assistant', 'content': y0}] + corr_messages
            corr_prompts.append(model.format_chat(full))

        print(f"  Correcting {len(corr_prompts)}/{len(items)} low-confidence items (τ={threshold})...")
        corr_outputs = model.generate(corr_prompts, temperature=0.0)

        for (idx, item, y0, conf), y1_list in zip(to_correct, corr_outputs):
            correction_results[idx] = y1_list[0]

    # Build results
    for idx, (item, y0_list, conf_list) in enumerate(zip(items, all_y0, conf_outputs)):
        y0 = y0_list[0]
        confidence = parse_confidence(conf_list[0])
        y0_correct = check_answer(dataset_name, y0, item)

        if idx in correction_results:
            y1 = correction_results[idx]
            corrected = True
        else:
            y1 = y0  # Keep original
            corrected = False

        y1_correct = check_answer(dataset_name, y1, item)
        category = classify_correction(y0_correct, y1_correct)

        results.append({
            'id': item['id'],
            'question': item.get('prompt', item.get('question', '')),
            'y0': y0,
            'y1': y1,
            'y0_correct': y0_correct,
            'y1_correct': y1_correct,
            'category': category,
            'confidence': confidence,
            'corrected': corrected,
        })

    return results


def _run_explain_verify(model, dataset_name, items, all_y0):
    """Run explain-then-verify correction (S5)."""
    results = []

    # Step 1: Get explanations
    explain_prompts = []
    for item, y0_list in zip(items, all_y0):
        y0 = y0_list[0]
        question = item.get('prompt', item.get('question', ''))
        messages_y0 = build_initial_prompt(dataset_name, question)
        explain_msgs = build_correction_prompt('s5_explain', dataset_name, question, y0)
        full = messages_y0 + [{'role': 'assistant', 'content': y0}] + explain_msgs
        explain_prompts.append(model.format_chat(full))

    print(f"  Getting explanations for {len(explain_prompts)} items (S5 step 1)...")
    explanations = model.generate(explain_prompts, temperature=0.0)

    # Step 2: Verify based on explanation
    verify_prompts = []
    for item, y0_list, expl_list in zip(items, all_y0, explanations):
        y0 = y0_list[0]
        explanation = expl_list[0]
        question = item.get('prompt', item.get('question', ''))
        messages_y0 = build_initial_prompt(dataset_name, question)
        explain_msgs = build_correction_prompt('s5_explain', dataset_name, question, y0)
        verify_msgs = build_correction_prompt('s5_verify', dataset_name, question, y0, explanation)
        full = (messages_y0 +
                [{'role': 'assistant', 'content': y0}] +
                explain_msgs +
                [{'role': 'assistant', 'content': explanation}] +
                verify_msgs)
        verify_prompts.append(model.format_chat(full))

    print(f"  Verifying {len(verify_prompts)} items (S5 step 2)...")
    verifications = model.generate(verify_prompts, temperature=0.0)

    for item, y0_list, expl_list, ver_list in zip(items, all_y0, explanations, verifications):
        y0 = y0_list[0]
        y1 = ver_list[0]
        y0_correct = check_answer(dataset_name, y0, item)
        y1_correct = check_answer(dataset_name, y1, item)
        category = classify_correction(y0_correct, y1_correct)

        results.append({
            'id': item['id'],
            'question': item.get('prompt', item.get('question', '')),
            'y0': y0,
            'y1': y1,
            'y0_correct': y0_correct,
            'y1_correct': y1_correct,
            'category': category,
            'explanation': expl_list[0],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Run self-correction experiment')
    parser.add_argument('--model', required=True, choices=['llama', 'mistral', 'qwen'])
    parser.add_argument('--dataset', required=True, choices=['gsm8k', 'triviaqa', 'strategyqa', 'humaneval'])
    parser.add_argument('--strategy', required=True,
                        choices=['baseline', 'cot', 'self_consistency', 's1', 's2', 's3', 's4', 's5'])
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--threshold', type=int, default=5, help='S4 confidence threshold (default: 5)')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit to N samples (for testing)')
    parser.add_argument('--no-vllm', action='store_true', help='Force HF transformers instead of vLLM')
    args = parser.parse_args()

    # Load data
    items = load_dataset_items(args.dataset)
    if args.max_samples:
        items = items[:args.max_samples]

    print(f"\n{'='*60}")
    print(f"Experiment: {args.model} / {args.dataset} / {args.strategy}")
    print(f"Samples: {len(items)}")
    print(f"{'='*60}\n")

    # Load model
    t0 = time.time()
    model = ModelWrapper(args.model, use_vllm=False)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Run experiment
    t0 = time.time()
    if args.strategy == 'baseline':
        results = run_baseline(model, args.dataset, items)
    elif args.strategy == 'cot':
        results = run_baseline(model, args.dataset, items, cot=True)
    elif args.strategy == 'self_consistency':
        results = run_self_consistency(model, args.dataset, items, n=3)
    else:
        results = run_correction_strategy(
            model, args.dataset, items, args.strategy, threshold=args.threshold
        )

    elapsed = time.time() - t0
    print(f"\n  Inference completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Compute summary metrics
    if args.strategy in ('baseline', 'cot'):
        n_correct = sum(1 for r in results if r['y0_correct'])
        accuracy = n_correct / len(results) if results else 0
        summary = {
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': len(results),
        }
    elif args.strategy == 'self_consistency':
        n_correct = sum(1 for r in results if r['majority_correct'])
        accuracy = n_correct / len(results) if results else 0
        summary = {
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': len(results),
        }
    else:
        categories = [r['category'] for r in results]
        summary = compute_correction_metrics(categories)

    print(f"\n  Summary: {json.dumps(summary, indent=2)}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        'model': args.model,
        'dataset': args.dataset,
        'strategy': args.strategy,
        'n_samples': len(results),
        'elapsed_seconds': elapsed,
        'summary': summary,
        'results': results,
    }
    if args.strategy == 's4':
        output_data['threshold'] = args.threshold

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()

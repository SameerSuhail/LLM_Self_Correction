"""Download and preprocess all 4 datasets, save as JSON for fast loading."""

import json
import random
import os
import sys

# Ensure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def prepare_gsm8k():
    """GSM8K: 1,319 test + 7,473 train questions."""
    print("Loading GSM8K...")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")

    test_items = []
    for i, item in enumerate(ds_test):
        test_items.append({
            'id': f'gsm8k_test_{i}',
            'question': item['question'],
            'answer': item['answer'],  # contains #### NUMBER
        })

    train_items = []
    for i, item in enumerate(ds_train):
        train_items.append({
            'id': f'gsm8k_train_{i}',
            'question': item['question'],
            'answer': item['answer'],
        })

    with open(os.path.join(DATA_DIR, 'gsm8k_test.json'), 'w') as f:
        json.dump(test_items, f, indent=2)
    with open(os.path.join(DATA_DIR, 'gsm8k_train.json'), 'w') as f:
        json.dump(train_items, f, indent=2)

    print(f"  GSM8K test: {len(test_items)} questions")
    print(f"  GSM8K train: {len(train_items)} questions")


def prepare_triviaqa():
    """TriviaQA: subsample 1,000 from validation (seed=42)."""
    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    random.seed(42)
    indices = random.sample(range(len(ds)), 1000)
    subset = ds.select(indices)

    items = []
    for i, item in enumerate(subset):
        items.append({
            'id': f'triviaqa_{i}',
            'question': item['question'],
            'answer': {
                'value': item['answer']['value'],
                'aliases': item['answer'].get('aliases', []),
            },
        })

    with open(os.path.join(DATA_DIR, 'triviaqa_test.json'), 'w') as f:
        json.dump(items, f, indent=2)

    print(f"  TriviaQA: {len(items)} questions (subsampled)")


def prepare_strategyqa():
    """StrategyQA: 2,290 test questions."""
    print("Loading StrategyQA...")
    # metaeval/strategy-qa only has a 'train' split (2,290 examples = the full test set)
    ds = load_dataset("metaeval/strategy-qa", split="train")

    items = []
    for i, item in enumerate(ds):
        # Dataset may use 'answer' (bool) or 'answer_text' (yes/no)
        answer = item.get('answer', None)
        if isinstance(answer, str):
            answer = answer.lower().strip() == 'yes'

        items.append({
            'id': f'strategyqa_{i}',
            'question': item['question'],
            'answer': answer,  # boolean
        })

    with open(os.path.join(DATA_DIR, 'strategyqa_test.json'), 'w') as f:
        json.dump(items, f, indent=2)

    print(f"  StrategyQA: {len(items)} questions")


def prepare_humaneval():
    """HumanEval: 164 coding problems."""
    print("Loading HumanEval...")
    ds = load_dataset("openai_humaneval", split="test")

    items = []
    for i, item in enumerate(ds):
        items.append({
            'id': f'humaneval_{i}',
            'task_id': item['task_id'],
            'prompt': item['prompt'],  # function signature + docstring
            'canonical_solution': item['canonical_solution'],
            'test': item['test'],  # test function
            'entry_point': item['entry_point'],
        })

    with open(os.path.join(DATA_DIR, 'humaneval_test.json'), 'w') as f:
        json.dump(items, f, indent=2)

    print(f"  HumanEval: {len(items)} problems")


if __name__ == '__main__':
    print("=" * 60)
    print("Preparing datasets for Second Thoughts project")
    print("=" * 60)

    prepare_gsm8k()
    prepare_triviaqa()
    prepare_strategyqa()
    prepare_humaneval()

    print("\nDone! All datasets saved to:", os.path.abspath(DATA_DIR))
    print("\nFiles created:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.json'):
            path = os.path.join(DATA_DIR, f)
            size = os.path.getsize(path)
            print(f"  {f}: {size / 1024:.1f} KB")

"""Build decision framework: predict when self-correction helps vs hurts.

This is Experiment 6 — the star of the project per professor's request.

Usage:
    python scripts/build_predictor.py
"""

import json
import os
import csv
import numpy as np
from collections import Counter

# Import sklearn components
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'prediction_model')

ALL_MODELS = ['llama', 'mistral', 'qwen']
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
MODELS = [m for m in ALL_MODELS
          if any(os.path.exists(os.path.join(RESULTS_DIR, f'{m}_{d}_baseline.json'))
                 for d in DATASETS)]
STRATEGIES = ['s1', 's2', 's3', 's4', 's5']

TASK_TYPE_MAP = {
    'gsm8k': 'math',
    'triviaqa': 'factual_qa',
    'strategyqa': 'commonsense',
    'humaneval': 'code',
}


def count_steps(text):
    """Count reasoning steps in response."""
    if not text:
        return 0
    import re
    steps = len(re.findall(r'(?:Step \d|^\d+[\.\)]|\d+\s*[+\-×÷*/=])', text, re.MULTILINE))
    if steps == 0:
        sentences = text.split('.')
        steps = sum(1 for s in sentences if re.search(r'\d+\s*[+\-*/=×÷]', s))
    return steps


def has_hedging(text):
    """Check for hedging language."""
    if not text:
        return False
    import re
    pattern = r'\b(I think|probably|not sure|might be|possibly|perhaps|could be|may be|uncertain)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_answer_format(text, dataset):
    """Check if response contains expected answer format."""
    if not text:
        return False
    import re
    if dataset == 'gsm8k':
        return bool(re.search(r'####|[Tt]he (?:final )?answer is|\\boxed', text))
    elif dataset == 'strategyqa':
        return bool(re.search(r'\b(yes|no)\b', text.lower()[:100]))
    elif dataset == 'humaneval':
        return bool(re.search(r'def |return ', text))
    else:  # triviaqa
        return len(text.strip()) > 0


def extract_features():
    """Extract per-question features from all correction experiments."""
    all_rows = []

    for model in MODELS:
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                path = os.path.join(RESULTS_DIR, f'{model}_{dataset}_{strategy}.json')
                if not os.path.exists(path):
                    continue

                with open(path) as f:
                    data = json.load(f)

                results = data.get('results', [])
                for r in results:
                    y0c = r.get('y0_correct', False)
                    y1c = r.get('y1_correct', False)

                    # Correction outcome label
                    if not y0c and y1c:
                        label = 1    # helped (W->C)
                    elif y0c and not y1c:
                        label = -1   # hurt (C->W)
                    else:
                        label = 0    # neutral

                    y0_text = r.get('y0', '')
                    confidence = r.get('confidence', None)

                    row = {
                        'model': model,
                        'dataset': dataset,
                        'strategy': strategy,
                        'task_type': TASK_TYPE_MAP[dataset],
                        'response_length': len(y0_text.split()) if y0_text else 0,
                        'num_reasoning_steps': count_steps(y0_text),
                        'has_hedging': int(has_hedging(y0_text)),
                        'has_answer_format': int(has_answer_format(y0_text, dataset)),
                        'confidence': confidence if confidence is not None else -1,
                        'y0_correct': int(y0c),
                        'label': label,
                    }
                    all_rows.append(row)

    return all_rows


def build_decision_tree(features):
    """Train decision tree and extract human-readable rules."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare feature matrix
    # Encode categoricals
    model_enc = LabelEncoder()
    dataset_enc = LabelEncoder()
    strategy_enc = LabelEncoder()
    task_enc = LabelEncoder()

    models = [f['model'] for f in features]
    datasets = [f['dataset'] for f in features]
    strategies = [f['strategy'] for f in features]
    tasks = [f['task_type'] for f in features]

    model_encoded = model_enc.fit_transform(models)
    dataset_encoded = dataset_enc.fit_transform(datasets)
    strategy_encoded = strategy_enc.fit_transform(strategies)
    task_encoded = task_enc.fit_transform(tasks)

    feature_names = [
        'model', 'dataset', 'strategy', 'task_type',
        'response_length', 'num_reasoning_steps',
        'has_hedging', 'has_answer_format', 'confidence',
    ]

    X = np.column_stack([
        model_encoded,
        dataset_encoded,
        strategy_encoded,
        task_encoded,
        [f['response_length'] for f in features],
        [f['num_reasoning_steps'] for f in features],
        [f['has_hedging'] for f in features],
        [f['has_answer_format'] for f in features],
        [f['confidence'] for f in features],
    ])

    y = np.array([f['label'] for f in features])

    # For binary classification: does correction help?
    # Simplify: 1 = helped, 0 = did not help (neutral or hurt)
    y_binary = (y == 1).astype(int)

    # Also: does correction hurt?
    # 1 = hurt, 0 = did not hurt
    y_hurt = (y == -1).astype(int)

    print(f"\nDataset: {len(features)} samples")
    print(f"  Helped (W->C): {sum(y == 1)}")
    print(f"  Neutral:       {sum(y == 0)}")
    print(f"  Hurt (C->W):   {sum(y == -1)}")
    print(f"  % Helped:      {sum(y == 1) / len(y) * 100:.1f}%")
    print(f"  % Hurt:        {sum(y == -1) / len(y) * 100:.1f}%")

    # --- Binary: Will correction HELP? ---
    print("\n" + "=" * 70)
    print("  DECISION TREE: Will correction HELP? (W->C)")
    print("=" * 70)

    dt_help = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, class_weight='balanced',
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(dt_help, X, y_binary, cv=cv, scoring='accuracy')
    print(f"\n5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    f1_scores = cross_val_score(dt_help, X, y_binary, cv=cv, scoring='f1')
    print(f"5-fold CV F1:       {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Fit on all data
    dt_help.fit(X, y_binary)

    # Print tree
    tree_text = export_text(dt_help, feature_names=feature_names, max_depth=4)
    print(f"\nDecision Tree Rules (helps):\n{tree_text}")

    # Feature importance
    importances = dt_help.feature_importances_
    print("\nFeature Importance (helps):")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        if imp > 0.01:
            print(f"  {name:<25} {imp:.3f}")

    # --- Binary: Will correction HURT? ---
    print("\n" + "=" * 70)
    print("  DECISION TREE: Will correction HURT? (C->W)")
    print("=" * 70)

    dt_hurt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, class_weight='balanced',
        random_state=42
    )

    scores_hurt = cross_val_score(dt_hurt, X, y_hurt, cv=cv, scoring='accuracy')
    print(f"\n5-fold CV accuracy: {scores_hurt.mean():.3f} ± {scores_hurt.std():.3f}")

    f1_hurt = cross_val_score(dt_hurt, X, y_hurt, cv=cv, scoring='f1')
    print(f"5-fold CV F1:       {f1_hurt.mean():.3f} ± {f1_hurt.std():.3f}")

    dt_hurt.fit(X, y_hurt)

    tree_text_hurt = export_text(dt_hurt, feature_names=feature_names, max_depth=4)
    print(f"\nDecision Tree Rules (hurts):\n{tree_text_hurt}")

    importances_hurt = dt_hurt.feature_importances_
    print("\nFeature Importance (hurts):")
    for name, imp in sorted(zip(feature_names, importances_hurt), key=lambda x: -x[1]):
        if imp > 0.01:
            print(f"  {name:<25} {imp:.3f}")

    # --- 3-class: help / neutral / hurt ---
    print("\n" + "=" * 70)
    print("  DECISION TREE: 3-CLASS (help / neutral / hurt)")
    print("=" * 70)

    dt_3class = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, class_weight='balanced',
        random_state=42
    )

    scores_3 = cross_val_score(dt_3class, X, y, cv=cv, scoring='accuracy')
    print(f"\n5-fold CV accuracy: {scores_3.mean():.3f} ± {scores_3.std():.3f}")

    dt_3class.fit(X, y)
    tree_text_3 = export_text(dt_3class, feature_names=feature_names, max_depth=4)
    print(f"\nDecision Tree Rules (3-class):\n{tree_text_3}")

    # Classification report
    y_pred = dt_3class.predict(X)
    print("\nClassification Report (full data):")
    print(classification_report(y, y_pred, target_names=['hurt(-1)', 'neutral(0)', 'helped(1)']))

    # Save results
    results = {
        'n_samples': len(features),
        'class_distribution': {
            'helped': int(sum(y == 1)),
            'neutral': int(sum(y == 0)),
            'hurt': int(sum(y == -1)),
        },
        'help_model': {
            'cv_accuracy': float(scores.mean()),
            'cv_f1': float(f1_scores.mean()),
            'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importances)},
            'tree_rules': tree_text,
        },
        'hurt_model': {
            'cv_accuracy': float(scores_hurt.mean()),
            'cv_f1': float(f1_hurt.mean()),
            'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importances_hurt)},
            'tree_rules': tree_text_hurt,
        },
        'three_class_model': {
            'cv_accuracy': float(scores_3.mean()),
            'tree_rules': tree_text_3,
        },
        'encoders': {
            'model': dict(zip(model_enc.classes_.tolist(), range(len(model_enc.classes_)))),
            'dataset': dict(zip(dataset_enc.classes_.tolist(), range(len(dataset_enc.classes_)))),
            'strategy': dict(zip(strategy_enc.classes_.tolist(), range(len(strategy_enc.classes_)))),
            'task_type': dict(zip(task_enc.classes_.tolist(), range(len(task_enc.classes_)))),
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'decision_framework.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'rules.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION FRAMEWORK: When does self-correction help?\n")
        f.write("=" * 70 + "\n\n")
        f.write("WILL CORRECTION HELP? (predicting W->C)\n")
        f.write("-" * 40 + "\n")
        f.write(tree_text + "\n\n")
        f.write("WILL CORRECTION HURT? (predicting C->W)\n")
        f.write("-" * 40 + "\n")
        f.write(tree_text_hurt + "\n\n")
        f.write("3-CLASS MODEL (help / neutral / hurt)\n")
        f.write("-" * 40 + "\n")
        f.write(tree_text_3 + "\n")

    print(f"\nSaved results to {OUTPUT_DIR}/")

    # Human-readable summary
    print("\n" + "=" * 70)
    print("  KEY FINDINGS FOR THE PAPER")
    print("=" * 70)

    # Analyze patterns by task type
    for task in ['math', 'factual_qa', 'commonsense', 'code']:
        mask = [f['task_type'] == task for f in features]
        task_features = [f for f, m in zip(features, mask) if m]
        if not task_features:
            continue
        helped = sum(1 for f in task_features if f['label'] == 1)
        hurt = sum(1 for f in task_features if f['label'] == -1)
        total = len(task_features)
        print(f"\n  {task.upper()}: {helped}/{total} helped ({helped/total*100:.1f}%), {hurt}/{total} hurt ({hurt/total*100:.1f}%)")

    # By strategy
    print()
    for strat in STRATEGIES:
        strat_features = [f for f in features if f['strategy'] == strat]
        if not strat_features:
            continue
        helped = sum(1 for f in strat_features if f['label'] == 1)
        hurt = sum(1 for f in strat_features if f['label'] == -1)
        total = len(strat_features)
        print(f"  {strat.upper()}: {helped}/{total} helped ({helped/total*100:.1f}%), {hurt}/{total} hurt ({hurt/total*100:.1f}%)")


def main():
    print("Extracting features from correction results...")
    features = extract_features()

    if not features:
        print("No correction results found. Run experiments first.")
        return

    print(f"Extracted {len(features)} feature rows")
    build_decision_tree(features)


if __name__ == '__main__':
    main()

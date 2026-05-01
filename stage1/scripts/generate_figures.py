"""Generate all figures for the paper.

Usage:
    python scripts/generate_figures.py
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'prediction_model')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

ALL_MODELS = ['llama', 'mistral', 'qwen']
# Only include models that have at least some result files
MODELS = [m for m in ALL_MODELS
          if any(os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', f'{m}_{d}_baseline.json'))
                 for d in ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval'])]
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
STRATEGIES = ['s1', 's2', 's3', 's4', 's5']
DATASET_LABELS = {'gsm8k': 'GSM8K', 'triviaqa': 'TriviaQA',
                  'strategyqa': 'StrategyQA', 'humaneval': 'HumanEval'}
STRATEGY_LABELS = {'s1': 'S1: Simple', 's2': 'S2: Structured',
                   's3': 'S3: Iterative', 's4': 'S4: Confidence',
                   's5': 'S5: Explain'}
MODEL_LABELS = {'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B', 'qwen': 'Qwen-2.5-7B'}


def load_result(model, dataset, strategy):
    path = os.path.join(RESULTS_DIR, f'{model}_{dataset}_{strategy}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_correction_metrics(data):
    if data is None:
        return None
    results = data.get('results', [])
    n = len(results)
    cc = cw = wc = ww = 0
    for r in results:
        y0c = r.get('y0_correct', False)
        y1c = r.get('y1_correct', False)
        if y0c and y1c: cc += 1
        elif y0c and not y1c: cw += 1
        elif not y0c and y1c: wc += 1
        else: ww += 1
    return {
        'n': n, 'cc': cc, 'cw': cw, 'wc': wc, 'ww': ww,
        'acc_before': (cc+cw)/n*100, 'acc_after': (cc+wc)/n*100,
        'net': wc-cw,
        'fix_rate': wc/(wc+ww)*100 if (wc+ww)>0 else 0,
        'reg_rate': cw/(cc+cw)*100 if (cc+cw)>0 else 0,
    }


def get_baseline_acc(model, dataset, strategy):
    data = load_result(model, dataset, strategy)
    if data is None:
        return None
    results = data.get('results', [])
    n = len(results)
    if strategy == 'self_consistency':
        correct = sum(1 for r in results if r.get('majority_correct', False))
    else:
        correct = sum(1 for r in results if r.get('y0_correct', False))
    return correct / n * 100 if n > 0 else 0


def fig1_net_score_heatmap():
    """Net score heatmap: strategies × datasets for each model."""
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5.5),
                             gridspec_kw={'wspace': 0.35})
    if n_models == 1:
        axes = [axes]

    strat_labels = [STRATEGY_LABELS[s] for s in STRATEGIES]
    ds_labels = [DATASET_LABELS[d] for d in DATASETS]

    for idx, model in enumerate(MODELS):
        matrix = []
        annot = []
        for strat in STRATEGIES:
            row = []
            annot_row = []
            for dataset in DATASETS:
                data = load_result(model, dataset, strat)
                m = get_correction_metrics(data)
                val = m['net'] if m else np.nan
                row.append(val)
                annot_row.append(f'{val:+.0f}' if not np.isnan(val) else '—')
            matrix.append(row)
            annot.append(annot_row)

        df = pd.DataFrame(matrix, index=strat_labels, columns=ds_labels)
        annot_arr = np.array(annot)
        ax = axes[idx]

        sns.heatmap(df, annot=annot_arr, fmt='', cmap='RdYlGn',
                    center=0, linewidths=1.5, linecolor='white',
                    cbar=idx == (n_models - 1),
                    cbar_kws={'label': 'Net Score (W->C minus C->W)',
                              'shrink': 0.8} if idx == (n_models - 1) else {},
                    ax=ax, annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                    vmin=-1000, vmax=100)

        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=10, rotation=0)
        ax.tick_params(axis='y', labelsize=10, rotation=0)

    fig.suptitle('Net Score by Strategy and Dataset', fontsize=15, fontweight='bold', y=1.0)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_net_score_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 1: Net score heatmap")


def fig2_correction_matrix_bars():
    """Stacked bar chart of correction matrix categories."""
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 5 * n_models))
    if n_models == 1:
        axes = np.array([axes])

    colors = {'C→C': '#2ecc71', 'C→W': '#e74c3c', 'W→C': '#3498db', 'W→W': '#95a5a6'}

    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            strats = []
            cc_vals, cw_vals, wc_vals, ww_vals = [], [], [], []

            for strat in STRATEGIES:
                data = load_result(model, dataset, strat)
                m = get_correction_metrics(data)
                if m is None:
                    continue
                strats.append(STRATEGY_LABELS[strat])
                total = m['n']
                cc_vals.append(m['cc']/total*100)
                cw_vals.append(m['cw']/total*100)
                wc_vals.append(m['wc']/total*100)
                ww_vals.append(m['ww']/total*100)

            if not strats:
                ax.set_visible(False)
                continue

            x = np.arange(len(strats))
            w = 0.6

            ax.bar(x, cc_vals, w, label='C→C', color=colors['C→C'])
            ax.bar(x, cw_vals, w, bottom=cc_vals, label='C→W', color=colors['C→W'])
            bottom2 = [a+b for a,b in zip(cc_vals, cw_vals)]
            ax.bar(x, wc_vals, w, bottom=bottom2, label='W→C', color=colors['W→C'])
            bottom3 = [a+b for a,b in zip(bottom2, wc_vals)]
            ax.bar(x, ww_vals, w, bottom=bottom3, label='W→W', color=colors['W→W'])

            ax.set_xticks(x)
            ax.set_xticklabels(strats, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_title(f'{MODEL_LABELS[model]} — {DATASET_LABELS[dataset]}', fontsize=11)
            if j == 0:
                ax.set_ylabel('% of questions')

    handles = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=12,
              bbox_to_anchor=(0.5, 0.02))
    plt.suptitle('Correction Matrix Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_correction_matrix_bars.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 2: Correction matrix bars")


def fig3_accuracy_comparison():
    """Grouped bar chart: baseline vs strategies accuracy."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    bar_colors = {
        'baseline': '#34495e', 'cot': '#7f8c8d', 'self_consistency': '#9b59b6',
        's1': '#e74c3c', 's2': '#e67e22', 's3': '#f1c40f',
        's4': '#2ecc71', 's5': '#3498db',
    }
    all_strats = ['baseline', 'cot', 'self_consistency', 's1', 's2', 's3', 's4', 's5']
    labels = {
        'baseline': 'Base', 'cot': 'CoT', 'self_consistency': 'SC',
        's1': 'S1', 's2': 'S2', 's3': 'S3', 's4': 'S4', 's5': 'S5',
    }

    for j, dataset in enumerate(DATASETS):
        ax = axes[j]
        x = np.arange(len(MODELS))
        width = 0.1
        n_bars = len(all_strats)
        offsets = np.arange(n_bars) - (n_bars - 1) / 2

        for k, strat in enumerate(all_strats):
            vals = []
            for model in MODELS:
                if strat in ['baseline', 'cot', 'self_consistency']:
                    acc = get_baseline_acc(model, dataset, strat)
                    vals.append(acc if acc is not None else 0)
                else:
                    data = load_result(model, dataset, strat)
                    m = get_correction_metrics(data)
                    vals.append(m['acc_after'] if m else 0)

            ax.bar(x + offsets[k] * width, vals, width,
                   label=labels[strat], color=bar_colors[strat], edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
        ax.set_title(DATASET_LABELS[dataset], fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)

    handles = [mpatches.Patch(color=bar_colors[s], label=labels[s]) for s in all_strats]
    fig.legend(handles=handles, loc='upper center', ncol=len(all_strats),
              fontsize=10, bbox_to_anchor=(0.5, 0.02))
    plt.suptitle('Accuracy Across All Conditions', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_accuracy_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 3: Accuracy comparison")


def fig4_regression_vs_fix():
    """Scatter plot: fix rate vs regression rate for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    markers = {'s1': 'o', 's2': 's', 's3': 'P', 's4': 'D', 's5': '^'}
    colors = {'llama': '#3498db', 'mistral': '#e74c3c', 'qwen': '#9b59b6'}

    # Collect all points for smart labeling
    points = []
    for model in MODELS:
        for strat in STRATEGIES:
            for dataset in DATASETS:
                data = load_result(model, dataset, strat)
                m = get_correction_metrics(data)
                if m is None:
                    continue
                points.append({
                    'x': m['fix_rate'], 'y': m['reg_rate'],
                    'model': model, 'strat': strat, 'dataset': dataset,
                    'label': f"{DATASET_LABELS[dataset]}"
                })
                ax.scatter(m['fix_rate'], m['reg_rate'],
                          c=colors[model], marker=markers[strat],
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          zorder=3)

    # Smart labeling: only label points that are "interesting"
    # (high fix, high regression, or far from others)
    texts = []
    try:
        from adjustText import adjust_text
        for p in points:
            # Skip near-zero cluster (fix<3 and reg<3) — too dense
            if p['x'] < 3 and p['y'] < 3:
                continue
            t = ax.text(p['x'], p['y'],
                       f"  {p['label']}",
                       fontsize=7, alpha=0.75, zorder=4)
            texts.append(t)
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray',
                    alpha=0.4, lw=0.5), expand=(1.2, 1.4))
    except ImportError:
        # Fallback: only label points outside the dense cluster
        labeled_zones = []
        for p in sorted(points, key=lambda p: p['x']**2 + p['y']**2, reverse=True):
            if p['x'] < 3 and p['y'] < 3:
                continue
            # Check if too close to an already-labeled point
            too_close = False
            for lx, ly in labeled_zones:
                if abs(p['x'] - lx) < 5 and abs(p['y'] - ly) < 5:
                    too_close = True
                    break
            if too_close:
                continue
            labeled_zones.append((p['x'], p['y']))
            # Place label away from diagonal
            if p['y'] > p['x']:
                xytext = (-8, 8)
            else:
                xytext = (8, -8)
            ax.annotate(p['label'],
                       (p['x'], p['y']),
                       textcoords="offset points", xytext=xytext,
                       fontsize=7, alpha=0.7,
                       arrowprops=dict(arrowstyle='-', color='gray',
                                      alpha=0.3, lw=0.5))

    # Add diagonal line (break-even) and shaded regions
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
    ax.fill_between([0, 100], [0, 100], 100, alpha=0.03, color='red')
    ax.fill_between([0, 100], 0, [0, 100], alpha=0.03, color='green')
    ax.text(55, 20, 'Net Positive', fontsize=10, alpha=0.3, color='green',
            fontstyle='italic', rotation=0)
    ax.text(10, 85, 'Net Negative', fontsize=10, alpha=0.3, color='red',
            fontstyle='italic', rotation=0)

    ax.set_xlabel('Fix Rate (%)', fontsize=13)
    ax.set_ylabel('Regression Rate (%)', fontsize=13)
    ax.set_title('Fix Rate vs Regression Rate\n(below diagonal = net positive)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-2, max(p['x'] for p in points) + 5)
    ax.set_ylim(-2, max(p['y'] for p in points) + 5)

    # Legend
    model_handles = [mpatches.Patch(color=c, label=MODEL_LABELS[m])
                    for m, c in colors.items()]
    strat_handles = [plt.Line2D([0], [0], marker=markers[s], color='gray',
                    linestyle='', markersize=8, label=STRATEGY_LABELS[s])
                    for s in STRATEGIES]
    ax.legend(handles=model_handles + strat_handles, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_regression_vs_fix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 4: Fix vs regression scatter")


def fig5_confidence_distribution():
    """Confidence score distributions for S4: correct vs incorrect y0."""
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 4 * n_models))
    if n_models == 1:
        axes = np.array([axes])

    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            data = load_result(model, dataset, 's4')
            if data is None:
                ax.set_visible(False)
                continue

            results = data['results']
            correct_conf = [r['confidence'] for r in results
                          if r.get('y0_correct', False) and 'confidence' in r]
            wrong_conf = [r['confidence'] for r in results
                        if not r.get('y0_correct', False) and 'confidence' in r]

            bins = np.arange(0.5, 11.5, 1)
            ax.hist(correct_conf, bins=bins, alpha=0.6, color='green',
                   label='Correct', density=True)
            ax.hist(wrong_conf, bins=bins, alpha=0.6, color='red',
                   label='Wrong', density=True)
            ax.set_title(f'{MODEL_LABELS[model]} — {DATASET_LABELS[dataset]}', fontsize=10)
            ax.set_xlabel('Confidence (1-10)')
            ax.set_ylabel('Density')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.suptitle('Confidence Distributions: Correct vs Wrong Answers',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_confidence_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 5: Confidence distributions")


def fig6_feature_importance():
    """Feature importance from decision framework."""
    pred_file = os.path.join(PRED_DIR, 'decision_framework.json')
    if not os.path.exists(pred_file):
        print("  Fig 6: Skipped (no prediction model)")
        return

    with open(pred_file) as f:
        pred = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (key, title) in enumerate([
        ('help_model', 'Will Correction Help?'),
        ('hurt_model', 'Will Correction Hurt?')
    ]):
        ax = axes[idx]
        imp = pred[key]['feature_importance']
        # Sort by importance
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names = [n for n, v in sorted_imp if v > 0.005]
        values = [v for n, v in sorted_imp if v > 0.005]

        bars = ax.barh(range(len(names)), values, color='#3498db')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()

        # Add CV accuracy
        cv_acc = pred[key]['cv_accuracy']
        ax.text(0.95, 0.95, f'CV Acc: {cv_acc:.1%}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Decision Framework Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_feature_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 6: Feature importance")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")

    fig1_net_score_heatmap()
    fig2_correction_matrix_bars()
    fig3_accuracy_comparison()
    fig4_regression_vs_fix()
    fig5_confidence_distribution()
    fig6_feature_importance()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()

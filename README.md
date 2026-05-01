# Second Thoughts: When LLM Self-Correction Helps vs Hurts

A three-stage empirical study of LLM self-correction. Course project for **CSCE 638 NLP, Spring 2026, Texas A&M University**, advised by Prof. Kuan-Hao Huang.

**Team:** Ashish Molakalapalli, Sameer Suhail, Samhitha Kondeti, Varenya Sri.

---

## The question

Self-correction — asking a model to review its own answer — is everywhere in deployed LLM systems. But the same parameters that produced the answer are now reviewing it. Same training data, same biases, same blind spots. Does it actually work?

We ask three sub-questions, in order:
1. **Does prompting-based correction work, given a fair setup?**
2. **If not, does training-based correction fix it?**
3. **Does a math-trained correction recipe transfer to code?**

Each stage corresponds to one folder in this repo.

---

## Headline findings

| Stage | Finding |
|---|---|
| **1: Prompting** | Across 96 conditions and 71,595 question–strategy pairs on three open-source 7–8B models, correction broke **14.7%** of correct answers and helped only **4.4%** — a **3.4× damage ratio**. Self-consistency at near-equal compute beats simple-review correction in 11 of 12 conditions. |
| **2: Training (math)** | LoRA fine-tuning on D1+D2 raises Mistral-7B from 47.61% → **65.43%** on GSM8K (**+17.82**). The attribution signal alone contributes **+2.96** points. |
| **3: Training (code)** | The same recipe **regresses** on Qwen-2.5-Coder-7B across all three code benchmarks (MBPP −10.27, HumanEval −5.87, Codeforces-A −15.22). External verifiers (the compiler) appear to make intrinsic self-correction supervision counterproductive. |

---

## Repository layout

```
LLM_Self_Correction/
├── stage1/                     Stage 1: Prompting-based correction
│   ├── src/                    Inference, evaluation, confidence parser
│   ├── scripts/                Experiment runners + analysis pipeline
│   ├── configs/prompts/        S1–S5 prompt templates + initial generation
│   ├── data/                   Test JSONs (GSM8K, TriviaQA, StrategyQA, HumanEval)
│   ├── results/                96 raw condition JSONs + aggregated CSVs + decision tree
│   ├── figures/                Heatmap, calibration curves, etc.
│   └── README.md               Stage 1 details + reproduction commands
│
├── stage2/                     Stage 2: Training-based math correction
│   ├── data_generation/        Wrong-step + judge orchestration
│   ├── generate_wrong_steps.py Wrong-step generation entrypoint
│   ├── sft_data/               Generated D1 and D2 datasets
│   ├── test_data/              Held-out evaluation problems
│   ├── training/               LoRA fine-tuning configs
│   └── README.md               Stage 2 details (D1, D2, masking scheme)
│
├── stage3/                     Stage 3: Training-based code correction
│   ├── configs/                Default hyperparameters (YAML)
│   ├── scripts/                Training, eval, SLURM, lightweight + upstream data-gen
│   ├── data/                   Stage 3 training datasets (committed JSONL)
│   ├── results/                Final JSON outputs + detailed reports
│   ├── docs/                   E2E reproduction + overview maps
│   └── README.md               Stage 3 entrypoint → docs/REPRODUCTION.md
│
└── README.md                   This file
```

---

## Stage 1 — Prompting-based correction

`stage1/`

A controlled **3 × 5 × 4** grid: three open-source 7–8B models (Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B), five correction strategies (S1 simple, S2 structured, S3 iterative, S4 confidence-gated, S5 explain-then-verify), four datasets (GSM8K, TriviaQA, StrategyQA, HumanEval), plus three baselines (no correction, CoT, self-consistency). 96 conditions, 71,595 question–strategy pairs.

We score every condition with a per-instance correction matrix in the style of Yang et al. (2024) — `C→C`, `C→W`, `W→C`, `W→W` — instead of just before/after accuracy. This exposes regression risk that aggregate accuracy hides.

Built on top of this, a depth-4 decision tree predicts whether correction will help on a given instance with **75.7%** cross-validated accuracy. Top features: confidence (37%), task type (25%), model identity (22%). The most useful single rule: *if self-reported confidence > 4, do not correct.*

See [`stage1/README.md`](stage1/README.md) for the full experimental protocol and reproduction commands.

---

## Stage 2 — Training-based math correction

`stage2/`

Two SFT datasets built from MetaMathQA, using Mistral-7B as the wrong-step generator and DeepSeek-R1-Distill-Qwen-14B as the judge:

- **D1 — single-step self-correction**: model detects and rewrites the wrong step in place. 1,805 valid examples.
- **D2 — multi-step late detection**: error propagates several steps; model detects late, retraces to the origin, and re-solves forward. 1,571 valid examples.

Both use the **all-rollouts-wrong filter**: a candidate step is kept only if all 8 rollouts past it produce wrong final answers. This selects steps from which the base model has no recovery path on its own — supervision is non-redundant.

LoRA fine-tuning (rank 8, α=16, target modules `{q,k,v,o}_proj`) on the combined D1+D2 set lifts Mistral-7B by +17.82 points on GSM8K. Holding everything else fixed, the **attribution signal** alone — keying the error-trace opener on which prior input was misused — is worth +2.96 points.

See [`stage2/README.md`](stage2/README.md) for the full data-generation pipeline and supervision format.

---

## Stage 3 — Training-based code correction

`stage3/`

We adapt the Stage 2 recipe to code: training Qwen-2.5-Coder-7B-Instruct on CodeContests, with a 14B teacher generating self-correction cycles. Unlike Stage 2, Stage 3 uses only the single-step (D1) format, no multi-step retrace, no attribution signal; LoRA at rank 16 targeting all linear projection layers.

Result: regression on every code benchmark (MBPP −10.27, HumanEval −5.87, Codeforces-A −15.22). We attribute this to (1) instruction conflict with an already heavily code-tuned base, (2) low-rank adapter capacity even at r=16, and (3) the existence of an external verifier (compiler + unit tests) that makes intrinsic self-correction supervision redundant where math has no analog.

See [`stage3/README.md`](stage3/README.md) and the end-to-end steps in [`stage3/docs/REPRODUCTION.md`](stage3/docs/REPRODUCTION.md).

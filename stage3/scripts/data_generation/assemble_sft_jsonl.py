#!/usr/bin/env python3
"""
Assemble CodeContests-style SFT JSONL files from intermediate JSONL shards.

This wraps the same logic used for the paper runs. Expected inputs live in a single
working directory (use --work-dir):

  - code_contests_wrong_steps_part_*.jsonl  (wrong-step corpus with gold traces)
  - cc_self_correction_part_*.jsonl         (records with filled self_correction)

Outputs (written to --out-dir, default same as --work-dir):

  - cc_sft_dataset_qwen.jsonl       (positive / error-trace corrective format)
  - cc_sft_dataset_baseline.jsonl   (standard CoT-only baselines from wrong-step shards)
  - cc_sft_dataset_qwen_mixed.jsonl (corrective positives + duplicated negatives at 1:5)

Shipped checkpoints in stage3/data/ were produced by this pipeline.

Usage (from repo root or stage3/):

    python scripts/data_generation/assemble_sft_jsonl.py \\
        --work-dir /path/to/intermediates \\
        --out-dir  /path/to/stage3/data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


def _load_gt_map(work_dir: Path) -> dict:
    gt_map = {}
    for file_path in sorted(work_dir.glob("code_contests_wrong_steps_part_*.jsonl")):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    gt_map[data["problem_name"]] = {
                        "solution": data["gt_solution"],
                        "language": data["gt_solution_language"].lower()
                        if data.get("gt_solution_language")
                        else "python",
                    }
                except Exception:
                    pass
    return gt_map


def build_qwen_jsonl(work_dir: Path, out_path: Path) -> int:
    gt_map = _load_gt_map(work_dir)
    pattern = str(work_dir / "cc_self_correction_part_*.jsonl")
    input_files = sorted(glob.glob(pattern))
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as outfile:
        for file_path in input_files:
            with open(file_path, encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        problem = data.get("problem", "").strip()
                        prob_name = data.get("problem_name", "")
                        prefix_text = data.get("prefix_text", "").strip()
                        prefix_len = data.get("prefix_len", 0)
                        wrong_step = data.get("wrong_step", "").strip()
                        self_correction = data.get("self_correction", {})
                        error_trace = self_correction.get("error_trace", "").replace(
                            "Error trace:", ""
                        ).strip()
                        diagnosis = self_correction.get("error_diagnosis", "").replace(
                            "Diagnosis:", ""
                        ).strip()
                        corrected_step = self_correction.get("corrected_step", "").replace(
                            "Corrected step:", ""
                        ).strip()

                        gold_steps = data.get("gold_reasoning", "").strip().split("\n")
                        remaining_steps = gold_steps[prefix_len + 1 :]
                        remaining_str = "\n".join(remaining_steps).strip()

                        gt_info = gt_map.get(
                            prob_name, {"solution": "# Code not found", "language": "python"}
                        )
                        gt_solution = gt_info["solution"].strip()
                        gt_lang = gt_info["language"].replace("c++", "cpp")

                        text_format = (
                            f"<|im_start|>user\nProblem:\n{problem}\n\nSolve step by step.<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                            f"{prefix_text}\n"
                            f"{wrong_step}\n"
                            f"Error trace: {error_trace}\n"
                            f"Diagnosis: {diagnosis}\n"
                            f"Corrected step: {corrected_step}\n"
                        )

                        if remaining_str:
                            text_format += f"{remaining_str}\n"

                        text_format += (
                            f"\nHere is the correct code:\n"
                            f"```{gt_lang}\n"
                            f"{gt_solution}\n"
                            f"```<|im_end|>"
                        )

                        outfile.write(json.dumps({"text": text_format}) + "\n")
                        count += 1
                    except Exception as exc:
                        print(f"Skipping line in {file_path}: {exc}")
    return count


def build_baseline_jsonl(work_dir: Path, out_path: Path) -> int:
    random.seed(42)
    raw_negatives = []
    for file_path in sorted(work_dir.glob("code_contests_wrong_steps_part_*.jsonl")):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problem = data.get("problem", "").strip()
                    gold_reasoning = data.get("gold_reasoning", "").strip()
                    gt_solution = data.get("gt_solution", "").strip()
                    if not problem or not gold_reasoning or not gt_solution:
                        continue
                    lang_raw = data.get("gt_solution_language")
                    lang = lang_raw.lower().replace("c++", "cpp") if lang_raw else "python"

                    text_format = (
                        f"<|im_start|>user\nProblem:\n{problem}\n\nSolve step by step.<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                        f"{gold_reasoning}\n\n"
                        f"Here is the correct code:\n"
                        f"```{lang}\n"
                        f"{gt_solution}\n"
                        f"```<|im_end|>"
                    )
                    raw_negatives.append({"text": text_format})
                except Exception:
                    continue

    unique_texts = set()
    unique_negatives = []
    for item in raw_negatives:
        if item["text"] not in unique_texts:
            unique_texts.add(item["text"])
            unique_negatives.append(item)

    random.shuffle(unique_negatives)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in unique_negatives:
            f.write(json.dumps(item) + "\n")
    return len(unique_negatives)


def mix_jsonl(work_dir: Path, quwen_jsonl: Path, out_path: Path) -> int:
    random.seed(42)
    positives = []
    with open(quwen_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positives.append(json.loads(line))
    p_count = len(positives)
    if p_count == 0:
        raise SystemExit(f"No positives in {quwen_jsonl}; run build corrective first.")

    raw_negatives = []
    for file_path in sorted(work_dir.glob("code_contests_wrong_steps_part_*.jsonl")):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problem = data.get("problem", "").strip()
                    gold_reasoning = data.get("gold_reasoning", "").strip()
                    gt_solution = data.get("gt_solution", "").strip()
                    if not problem or not gold_reasoning or not gt_solution:
                        continue
                    lang_raw = data.get("gt_solution_language")
                    lang = lang_raw.lower().replace("c++", "cpp") if lang_raw else "python"
                    text_format = (
                        f"<|im_start|>user\nProblem:\n{problem}\n\nSolve step by step.<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                        f"{gold_reasoning}\n\n"
                        f"Here is the correct code:\n"
                        f"```{lang}\n"
                        f"{gt_solution}\n"
                        f"```<|im_end|>"
                    )
                    raw_negatives.append({"text": text_format})
                except Exception:
                    pass

    if not raw_negatives:
        raise SystemExit(
            "No negative samples loaded from code_contests_wrong_steps_part_*.jsonl"
        )

    n_target = p_count * 5
    negative_samples = []
    idx = 0
    while len(negative_samples) < n_target:
        negative_samples.append(raw_negatives[idx % len(raw_negatives)])
        idx += 1

    combined = positives + negative_samples
    random.shuffle(combined)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Stage 3 SFT JSONL datasets.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Directory containing code_contests_wrong_steps_part_*.jsonl "
        "and cc_self_correction_part_*.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write outputs (defaults to --work-dir)",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    out_dir = (args.out_dir or work_dir).resolve()

    corrective_path = out_dir / "cc_sft_dataset_qwen.jsonl"
    baseline_path = out_dir / "cc_sft_dataset_baseline.jsonl"
    mixed_path = out_dir / "cc_sft_dataset_qwen_mixed.jsonl"

    n_corr = build_qwen_jsonl(work_dir, corrective_path)
    print(f"Wrote {corrective_path} ({n_corr} lines)")

    n_base = build_baseline_jsonl(work_dir, baseline_path)
    print(f"Wrote {baseline_path} ({n_base} lines)")

    n_mix = mix_jsonl(work_dir, corrective_path, mixed_path)
    print(f"Wrote {mixed_path} ({n_mix} lines)")


if __name__ == "__main__":
    main()

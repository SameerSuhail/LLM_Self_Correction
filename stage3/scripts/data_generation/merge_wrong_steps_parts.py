#!/usr/bin/env python3
"""
Concatenate code_contests_wrong_steps_part_*.jsonl into one file.

root_cause_attribution_code.py defaults to INPUT_FILE =
``code_contests_wrong_steps_all.jsonl``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Folder with code_contests_wrong_steps_part_*.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("code_contests_wrong_steps_all.jsonl"),
        help="Merged output JSONL filename (written inside --work-dir unless absolute)",
    )
    args = p.parse_args()
    work_dir = args.work_dir.resolve()
    parts = sorted(glob.glob(str(work_dir / "code_contests_wrong_steps_part_*.jsonl")))
    if not parts:
        raise SystemExit(f"No part files found in {work_dir}")
    out = args.output if args.output.is_absolute() else work_dir / args.output
    with open(out, "w", encoding="utf-8") as fout:
        for path in parts:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"Merged {len(parts)} shards -> {out}")


if __name__ == "__main__":
    main()

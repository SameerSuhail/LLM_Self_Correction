#!/bin/bash
#SBATCH --job-name=lora_v2
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=142682194159

VARIANT="${1:?usage: sbatch -J lora_v2_<variant> $0 <attr|noattr|ds2_spon|combined>}"
case "$VARIANT" in
    attr|noattr|ds2_spon|combined) ;;
    *) echo "Unknown variant: $VARIANT (expected one of: attr, noattr, ds2_spon, combined)" >&2
       exit 1 ;;
esac

module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0
export PYTHONPATH=/scratch/user/sameersuhail/packages:$PYTHONPATH
export HF_HOME=/scratch/user/sameersuhail/hf_cache
export TRANSFORMERS_CACHE=/scratch/user/sameersuhail/hf_cache
export HF_DATASETS_CACHE=/scratch/user/sameersuhail/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/user/sameersuhail/llm_self_correction_data_gen
mkdir -p logs "checkpoints/lora_v2_${VARIANT}"

echo "=== Job info ==="
echo "Variant: $VARIANT"
echo "Job ID:  $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
echo ""

case "$VARIANT" in
    attr)     echo "--- Build mix: 2x SC (D1 v2 attr) + 10k MetaMath GSM ---" ;;
    noattr)   echo "--- Build mix: 2x SC (D1 v2 noattr) + 10k MetaMath GSM ---" ;;
    ds2_spon) echo "--- Build mix: 2x SC (D2 spon v2) + 10k MetaMath GSM ---" ;;
    combined) echo "--- Build mix: D1 + D2 spon (no 2x) + 10k MetaMath GSM ---" ;;
esac

VARIANT="$VARIANT" python3 - << 'PYEOF'
import json, os, random
import pyarrow as pa

ARROW = "/scratch/user/sameersuhail/hf_cache/meta-math___meta_math_qa/default/0.0.0/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/meta_math_qa-train.arrow"
GSM = {"GSM_Rephrased","GSM_SV","GSM_AnsAug","GSM_FOBAR"}

variant = os.environ["VARIANT"]

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def clean(row):
    return {"type":"clean_solution",
            "messages":[{"role":"user","content":f"Problem:\n{row['query']}\n\nSolve step by step."},
                        {"role":"assistant","content":row["response"]}],
            "wrong_step_text":""}

if variant == "attr":
    sc_t = load("sft_dataset1_contv2_train.jsonl")
    sc_v = load("sft_dataset1_contv2_val.jsonl")
    print(f"SC train {len(sc_t)} val {len(sc_v)}")
elif variant == "noattr":
    sc_t = load("sft_dataset1_contv2_noattr_train.jsonl")
    sc_v = load("sft_dataset1_contv2_noattr_val.jsonl")
    print(f"SC train {len(sc_t)} val {len(sc_v)}")
elif variant == "ds2_spon":
    sc_t = load("sft_dataset2_spon_contv2_train.jsonl")
    sc_v = load("sft_dataset2_spon_contv2_val.jsonl")
    print(f"SC train {len(sc_t)} val {len(sc_v)}")
elif variant == "combined":
    d1_t = load("sft_dataset1_contv2_train.jsonl")
    d1_v = load("sft_dataset1_contv2_val.jsonl")
    d2_t = load("sft_dataset2_spon_contv2_train.jsonl")
    d2_v = load("sft_dataset2_spon_contv2_val.jsonl")
    print(f"D1 train {len(d1_t)} val {len(d1_v)} | D2 train {len(d2_t)} val {len(d2_v)}")

with pa.memory_map(ARROW,"r") as m:
    t = pa.ipc.open_stream(m).read_all()
rng = random.Random(42)
rows = [{c:t[c][i].as_py() for c in t.column_names}
        for i in range(len(t))
        if t["type"][i].as_py() in GSM]
rng.shuffle(rows)
mt = rows[:10000]
mv = rows[10000:11000]

if variant == "combined":
    def tag_d1(r): return {**r, "type": "self_correction_d1"}
    def tag_d2(r): return {**r, "type": "self_correction_d2"}
    train_records = (
        [tag_d1(r) for r in d1_t] +
        [tag_d2(r) for r in d2_t] +
        [clean(r)  for r in mt]
    )
    rng.shuffle(train_records)
    val_records = (
        [tag_d1(r) for r in d1_v] +
        [tag_d2(r) for r in d2_v] +
        [clean(r)  for r in mv]
    )
    rng.shuffle(val_records)
    d1_count  = sum(1 for r in train_records if r["type"]=="self_correction_d1")
    d2_count  = sum(1 for r in train_records if r["type"]=="self_correction_d2")
    cln_count = sum(1 for r in train_records if r["type"]=="clean_solution")
    print(f"\nWrote sft_v2_combined_train.jsonl: {len(train_records)} ({d1_count} D1 + {d2_count} D2 + {cln_count} clean)")
    print(f"Wrote sft_v2_combined_val.jsonl:   {len(val_records)}")
else:
    def tag(r): return {**r, "type": "self_correction"}
    tr = [tag(r) for r in sc_t] + [tag(r) for r in sc_t] + [clean(r) for r in mt]
    rng.shuffle(tr)
    va = [tag(r) for r in sc_v] + [clean(r) for r in mv]
    rng.shuffle(va)
    train_records, val_records = tr, va
    print(f"train {len(train_records)} val {len(val_records)}")

train_path = f"sft_v2_{variant}_train.jsonl"
val_path   = f"sft_v2_{variant}_val.jsonl"
with open(train_path, "w") as f:
    for r in train_records: f.write(json.dumps(r)+"\n")
with open(val_path, "w") as f:
    for r in val_records: f.write(json.dumps(r)+"\n")
PYEOF

echo ""
echo "--- LoRA training (5 epochs) ---"
python3 run_lora.py \
    --train  "sft_v2_${VARIANT}_train.jsonl" \
    --val    "sft_v2_${VARIANT}_val.jsonl" \
    --output "checkpoints/lora_v2_${VARIANT}" \
    --epochs 5 --lr 2e-4 --lora_r 8 --lora_alpha 16

echo ""
echo "=== Finished: $(date) ==="

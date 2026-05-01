#!/bin/bash
#SBATCH --job-name=baseline_sft_fast4x
#SBATCH --partition=gpu
#SBATCH --nodes=1                          # 1 node only!
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # Max CPUs
#SBATCH --mem=256G                         # RAM for Full SFT
#SBATCH --gres=gpu:a100:2                  # Request 2 A100s
#SBATCH --time=12:00:00                    # Baseline usually faster as dataset is smaller
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load WebProxy
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

source $SCRATCH/envs/sft_env/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export TMPDIR=$SCRATCH/tmp

# 2-GPU DDP Config
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

echo "Starting BASELINE FULL FINETUNING (No LoRA) with 2 A100 GPUs..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STAGE3_ROOT"

# Using torchrun for 2-GPU Data Parallelism
torchrun --nproc_per_node=2 scripts/training/run_baseline.py \
    --model_id "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --data_path "$STAGE3_ROOT/data/cc_sft_dataset_baseline.jsonl" \
    --output_dir "$SCRATCH/checkpoints/baseline_sft_cc_qwen_fast2x" \
    --cache_dir "$SCRATCH/hf_cache" \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --epochs 3 \
    --max_seq_length 2048 \
    --learning_rate 2e-5

echo "Job completed."

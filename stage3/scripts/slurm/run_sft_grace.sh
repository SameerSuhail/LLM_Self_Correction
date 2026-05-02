#!/bin/bash
#SBATCH --job-name=lora_sft_fast4x
#SBATCH --partition=gpu
#SBATCH --nodes=1                          # 1 node only!
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # Max CPUs
#SBATCH --mem=128G                         # RAM for LoRA
#SBATCH --gres=gpu:a100:2                  # Request 4 A100s
#SBATCH --time=24:00:00                    # Safety margin per your request
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

# 4-GPU DDP Config
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

echo "Starting LoRA FINETUNING with 4 A100 GPUs on 1 FASTER node..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STAGE3_ROOT"

# Using torchrun for 4-GPU Data Parallelism
torchrun --nproc_per_node=2 scripts/training/run_sft.py \
    --model_id "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --data_path "$STAGE3_ROOT/data/cc_sft_dataset_qwen_mixed.jsonl" \
    --output_dir "$SCRATCH/checkpoints/lora_cc_qwen_fast4x" \
    --cache_dir "$SCRATCH/hf_cache" \
    --batch_size 2 \
    --gradient_accumulation 16 \
    --epochs 10 \
    --max_seq_length 4096 \
    --learning_rate 1e-4

echo "Job completed."

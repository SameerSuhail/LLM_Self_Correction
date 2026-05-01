#!/bin/bash
#SBATCH --job-name=eval_baseline_openai_human-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load WebProxy
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers

source $SCRATCH/envs/sft_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE3_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STAGE3_ROOT"
echo "Starting evaluation job: eval_baseline_openai_human-eval"
python scripts/evaluation/eval_benchmarks.py \
    --model_path "$SCRATCH/checkpoints/baseline_sft_cc_qwen_fast2x/final" \
    --dataset "openai/human-eval" \
    --output "results/json/baseline_openai_human-eval.json"

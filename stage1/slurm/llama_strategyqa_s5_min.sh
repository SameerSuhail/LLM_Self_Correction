#!/bin/bash
#SBATCH --job-name=selfcorrect_llama_strategyqa_s5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/user/ashish_molakalapalli/nlp_project/logs/%x_%j.out
#SBATCH --error=/scratch/user/ashish_molakalapalli/nlp_project/logs/%x_%j.err
#SBATCH --account=142681700299

module load CUDA/12.4.0
module load WebProxy

export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/scratch/user/ashish_molakalapalli/nlp_project/data/models

cd /scratch/user/ashish_molakalapalli/nlp_project

/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python scripts/run_experiment.py \
    --model llama --dataset strategyqa --strategy s5 \
    --output results/raw/llama_strategyqa_s5.json

#!/bin/bash
#SBATCH --job-name=rag_gen_exp
#SBATCH --output=logs/rag_gen_%j.out
#SBATCH --error=logs/rag_gen_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -C vram80
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oyilmazel@umass.edu

module load conda/latest
module load cuda/12.6

echo "Job started on $(hostname)"
nvidia-smi
lscpu

conda activate lamp

SPLIT=test       # Change this to validation or test as needed
NUM_CONTEXTS=10
LIMIT_TARGET=0
MODEL=Qwen/Qwen2.5-7B-Instruct
CACHE_DIR=/scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/

for PREFIX in ae lp sc; do
  echo "Running for domain: ${PREFIX} (${SPLIT})"

  python -B asymmetric_baselines.py \
    --cache_dir "${CACHE_DIR}" \
    --keep_domains ae,lp,sc \
    --model_addr "${MODEL}" \
    --inputs_addr data/processed/${PREFIX}_processed_${SPLIT}.json \
    --output_addr data/out/rag/full_profile/${PREFIX}_rag_full_profile_${SPLIT}_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
    --num_contexts "${NUM_CONTEXTS}" --rag \
    --limit_target "${LIMIT_TARGET}"

  python -c "import torch; torch.cuda.empty_cache()"
  sleep 10
done

#!/bin/bash
#SBATCH --job-name=nopers_gen_exp
#SBATCH --output=logs/nopers_gen_%j.out
#SBATCH --error=logs/nopers_gen_%j.err
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

conda activate lamp

SPLIT=train         # Change to train/test/validation
ENTROPY_PERCENTILE=0     # Change as needed
MODEL=Qwen/Qwen2.5-7B-Instruct
CACHE_DIR=/scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

for PREFIX in ae lp sc; do
  echo "Running ${PREFIX} (${SPLIT}) at entropy ≥ ${ENTROPY_PERCENTILE}"

  python asymmetric_baselines.py \
    --cache_dir "${CACHE_DIR}" \
    --model_addr "${MODEL}" \
    --inputs_addr data/processed/${PREFIX}_processed_${SPLIT}.json \
    --output_addr data/out/nopers/${PREFIX}_${SPLIT}_${ENTROPY_PERCENTILE}_output.json \
    --entropy_percentile "${ENTROPY_PERCENTILE}"

  python -c "import torch; torch.cuda.empty_cache()"
  sleep 10
done

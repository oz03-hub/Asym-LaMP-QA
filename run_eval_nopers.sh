#!/bin/bash
#SBATCH --job-name=nopers_eval
#SBATCH --output=logs/nopers_eval_%j.out
#SBATCH --error=logs/nopers_eval_%j.err
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

SPLIT=train                        # Change to train/test/validation
ENTROPY=0                        # Entropy threshold
MODEL=Qwen/Qwen2.5-32B-Instruct
CACHE_DIR=/scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

for PREFIX in ae lp sc; do
  echo "Evaluating ${PREFIX} (${SPLIT}) at entropy ≥ ${ENTROPY}"

  python evaluate_responses.py \
    --cache_dir "${CACHE_DIR}" \
    --evaluator_llm "${MODEL}" \
    --inputs_addr data/processed/${PREFIX}_processed_${SPLIT}.json \
    --response_addr data/out/nopers/${PREFIX}_${SPLIT}_${ENTROPY}_output.json \
    --score_addr data/scores/nopers/${PREFIX}_${ENTROPY}_${SPLIT}.json

  python -c "import torch; torch.cuda.empty_cache()"
  sleep 10
done

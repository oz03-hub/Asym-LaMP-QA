#!/bin/bash
#SBATCH --job-name=eval_rag
#SBATCH --output=logs/eval_rag_%j.out
#SBATCH --error=logs/eval_rag_%j.err
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

SPLIT=test  # Change to test/train/validation as needed
NUM_CONTEXTS=10
LIMIT_TARGET=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

for PREFIX in ae lp sc; do
  echo python evaluate_responses.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ --evaluator_llm Qwen/Qwen2.5-32B-Instruct --inputs_addr data/processed/${PREFIX}_processed_${SPLIT}.json --response_addr data/out/rag/full_profile/${PREFIX}_rag_full_profile_${SPLIT}_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json --score_addr data/scores/rag/full_profile/${PREFIX}_rag_full_profile_${SPLIT}_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

  python evaluate_responses.py \
    --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
    --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
    --inputs_addr data/processed/${PREFIX}_processed_${SPLIT}.json \
    --response_addr data/out/rag/full_profile/${PREFIX}_rag_full_profile_${SPLIT}_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
    --score_addr data/scores/rag/full_profile/${PREFIX}_rag_full_profile_${SPLIT}_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

    sleep 10
done

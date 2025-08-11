#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/asym_%j.out
#SBATCH --error=logs/asym_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

module load conda/latest
module load cuda/12.6

conda activate lamp

python evaluate_responses.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ --evaluator_llm Qwen/Qwen2.5-7B-Instruct --inputs_addr data/processed/ae_processed_test.json --response_addr data/out/rag/full_profile/ae_rag_full_profile_test_2_output.json --score_addr data/scores/rag/full_profile/ae_full_profile_test_2.json

python evaluate_responses.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ --evaluator_llm Qwen/Qwen2.5-7B-Instruct --inputs_addr data/processed/lp_processed_test.json --response_addr data/out/rag/full_profile/lp_rag_full_profile_test_2_output.json --score_addr data/scores/rag/full_profile/lp_full_profile_test_2.json

python evaluate_responses.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ --evaluator_llm Qwen/Qwen2.5-7B-Instruct --inputs_addr data/processed/sc_processed_test.json --response_addr data/out/rag/full_profile/sc_rag_full_profile_test_2_output.json --score_addr data/scores/rag/full_profile/sc_full_profile_test_2.json

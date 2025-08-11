#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --output=logs/rank_%j.out
#SBATCH --error=logs/rank_%j.err
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

python produce_filtered_rank.py --input data/processed/ae_processed_test.json --output ae_test_ranks.txt --target ae

python produce_filtered_rank.py --input data/processed/lp_processed_test.json --output lp_test_ranks.txt --target lp

python produce_filtered_rank.py --input data/processed/sc_processed_test.json --output sc_test_ranks.txt --target sc

#!/bin/bash
#SBATCH --job-name=MACE
#SBATCH --time=22:30:00
#SBATCH --output=results_24.out
#SBATCH --error=results_24.err
#SBATCH --nodes=1
##SBATCH --gres=gpu:4g.40gb:1
#SBATCH --gpus=1
##SBATCH --mem=500G

ulimit -s unlimited

python MACE_lobster_node.py
#python MACE_matminer_lobster_node.py

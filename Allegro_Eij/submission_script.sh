#!/bin/bash
#SBATCH --job-name=megnet
#SBATCH --time=20:30:00
#SBATCH --output=results.out
#SBATCH --error=results.err
#SBATCH --nodes=1
##SBATCH --gres=gpu:2g.20gb:1
#SBATCH --gpus=1
 
 
python EijAllegro_icoxx_data_plot.py

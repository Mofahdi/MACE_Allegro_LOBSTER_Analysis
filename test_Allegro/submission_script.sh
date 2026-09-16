#!/bin/bash
#SBATCH --job-name=allegro
#SBATCH --time=20:30:00
#SBATCH --output=results.out
#SBATCH --error=results.err
#SBATCH --nodes=1
##SBATCH --gres=gpu:4g.40gb:1
#SBATCH --gpus=1
 
#python run_allegro_eij_icoxx_analysis.py \
  #--num-dcor-samples 40 \
  #--dcor-sample-size 50000 \
  #--dpi 800
 
python run_allegro_eij_icoxx_analysis.py \
  --num-dcor-samples 20 \
  --dcor-sample-size 50000 \
  --dpi 800

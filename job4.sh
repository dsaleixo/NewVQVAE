#!/bin/bash
#SBATCH --job-name=meu_job
#SBATCH --output=saida4.log
#SBATCH --error=erro.log
#SBATCH --time=8:15:00
#SBATCH --account=aip-lelis
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32G 
#SBATCH --cpus-per-task=1

source $HOME/VQVAE/bin/activate





python  dualDecoder/loop4.py
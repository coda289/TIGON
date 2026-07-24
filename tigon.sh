#!/bin/bash
set -e

#$ -cwd

#$ -j y

#$ -N NN_first

#$ -l highp

#$ -l h_rt=96:00:00

#$ -l h_data=16G

#$ -l gpu

#$ -pe shared 20


source ~/miniconda3/etc/profile.d/conda.sh

conda activate neuralode

python3 TIGON.py
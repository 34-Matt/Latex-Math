#!/bin/bash

#SBATCH --job-name=TrainCNN
#SBATCH --output=out.txt
#SBATCH --error=err.txt

#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=18:00:00

module purge
module load apps/python3
pip install --user --upgrade tensorflow
pip install tensorflow-gpu
pip install --user numpy


python TrainCNN.py
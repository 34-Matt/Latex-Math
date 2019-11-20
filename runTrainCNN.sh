#!/bin/bash

#SBATCH --job-name=TrainCNN
#SBATCH --output=out.txt
#SBATCH --error=err.txt

#SBATCH --ntasks=1
#SBATCH --time=1:00:00

module purge
module load apps/python3
pip install tensorflow==2.0
pip install numpy


python TrainCNN.py
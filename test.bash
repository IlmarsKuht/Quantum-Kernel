#!/bin/bash
#SBATCH --job-name=Training_Kernel
#SBATCH --account=Project_2004128
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --output=testing.txt

export SING_IMAGE="env.sif"

#apptainer_wrapper exec python3 main.py ./datasets/iris_small.csv 10 10 5 20000 2 0.9
srun apptainer_wrapper exec python3 stress_test.py


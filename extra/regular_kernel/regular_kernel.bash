#!/bin/bash
#SBATCH --job-name=Kernel
#SBATCH --account=Project_2004128
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=small
##SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --output=%A_%a.txt

#RUN batch.bash, not this script directly !!!!!!!!!!!!!

export SING_IMAGE="../../../env.sif"
#export SING_FLAGS=--nv

# Params passed from batch wrapper
num_combinations=$1
out_dir=$2
data_dir=$3

# Calculate the current file and combination indices
file_index=$(( ($SLURM_ARRAY_TASK_ID-1) / num_combinations + 1 ))
combination_index=$(( ($SLURM_ARRAY_TASK_ID-1) % num_combinations + 1 ))

# Get the current file name
file=$(ls $data_dir | sed -n "${file_index}p")
folder="${data_dir}/${file}"

# Read the line corresponding to the array index from the combinations file
params=$(sed -n "${combination_index}p" combinations.txt)

# Run your job with the parameters
srun apptainer_wrapper exec python3 -u ../../src/main.py \
					--dataset_dir $folder \
					$params \
					&> "${out_dir}/${file}/${combination_index}.txt"


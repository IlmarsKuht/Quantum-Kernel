#!/bin/bash
#SBATCH --job-name=SVM_Test
#SBATCH --account=Project_2004128
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=small
##SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --output=%A_%a.txt

#RUN batch.bash, not this script directly

export SING_IMAGE="../../../env.sif"
#export SING_FLAGS=--nv

# Params passed from batch wrapper
out_dir=$1
data_dir=$2

# Calculate the current file and combination indices
file_index=$SLURM_ARRAY_TASK_ID

# Get the current file name
file=$(ls $data_dir | sed -n "${file_index}p")
folder="${data_dir}/${file}"

# Run your job with the parameters
srun apptainer_wrapper exec python3 -u ../../src/classicalSVM.py \
					--dataset_dir $folder \
					&> "${out_dir}/${file}"


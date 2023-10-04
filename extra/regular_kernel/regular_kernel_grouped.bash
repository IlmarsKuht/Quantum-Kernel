#!/bin/bash
#SBATCH --job-name=Kernel
#SBATCH --account=Project_2004128
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=large
##SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --output=%A_%a.txt

#RUN batch_grouped.bash, not this script directly !!!!!!!!!!!!!!!!

export SING_IMAGE="../../../env.sif"
#export SING_FLAGS=--nv

# Params passed from batch wrapper
num_combinations=$1
out_dir=$2
data_dir=$3

# Calculate the current file and combination indices
file_index=$(( ($SLURM_ARRAY_TASK_ID-1) / num_combinations + 1 ))

# Get the current file name
file=$(ls $data_dir | sed -n "${file_index}p")
folder="${data_dir}/${file}"

# Run srun in parallel for each combination
for combination_index in $(seq 1 $num_combinations); do
    # Read the line corresponding to the array index from the combinations file
    params=$(sed -n "${combination_index}p" combinations.txt)
    
    # Run your job with the parameters
    srun --exclusive -n1 -c1 apptainer_wrapper exec python3 -u ../../src/main.py \
                    --dataset_dir $folder \
                    $params \
                    &> "${out_dir}/${file}/${combination_index}.txt" &
done

# Wait for all background jobs to finish
wait

#!/bin/bash
#SBATCH --job-name=Kernel_alignment_1
#SBATCH --account=Project_2004128
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=small
#SBATCH --array=0-11
#SBATCH --output=kernAlign2_%A_%a.txt

export SING_IMAGE="env.sif"

# Create output directory
out_dir="./run_small_1"
folder="./datasets"  # Specify the folder path
mkdir -p $out_dir

# Parameters
wires=(3 5)  # Number of wires
layers=(5 7)  # Layers
lr_rates=(0.01 0.03 0.1)  # Learning rates

# Other Parameters
batch_size=10
optim_iter=500000
prune_after=3

# Calculate indices for parameter arrays
i=$SLURM_ARRAY_TASK_ID
i_wire=$((i / 6))
i_layer=$(((i / 3) % 2))
i_lr=$((i % 3))

# Retrieve parameters from arrays
wire=${wires[$i_wire]}
layer=${layers[$i_layer]}
lr=${lr_rates[$i_lr]}

# Execute python script with selected parameters
srun apptainer_wrapper exec \
	python3 \
	-u generalKernel.py \
	$folder \
	$wire \
	$layer \
	$batch_size \
	$optim_iter \
	$prune_after \
	$lr \
	&> "$out_dir/${layer}_${wire}_${lr}.txt"


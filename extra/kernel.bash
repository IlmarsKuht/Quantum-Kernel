#!/bin/bash
#SBATCH --job-name=Proj_Kernel_Test
#SBATCH --account=Project_2004128
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=small
##SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=10
#SBATCH --output=%A_%a.txt
#SBATCH --array=1-15

export SING_IMAGE="../../env.sif"
#export SING_FLAGS=--nv

# LOOP THROUGH THE FILES AND SUBMIT BATCH JOBS
out_dir="./../../outputs_$SLURM_JOB_ID"
folder="../datasets/train"  # Specify the folder path
# Takes the nth file (n is SLURM_ARRAY_TASK_ID)
file=$(ls -p "$folder" | sed -n "${SLURM_ARRAY_TASK_ID}p")
mkdir -p "$out_dir/$file"
# Clean the dir in case it was filled already
rm "$out_dir/$file/"*


# Define num wires
wires=(3 5 7)  # Add more parameters as needed
#Define layers
layers=(3 5 7)

for layer in "${layers[@]}"; do
        for wire in "${wires[@]}"; do
		for lr in "${lr_rates[@]}"; do
			srun -n 1 -c 1 --exclusive apptainer_wrapper exec \
				python3 \
				-u ../src/main.py \
				"$folder/$file" \
				$wire \
				$layer \
				&> "$out_dir/${file}/${layer}_${wire}_${lr}.txt" &
		done
	done
done

srun -n 1 -c 1 --exclusive apptainer_wrapper exec \
        python3 classicalSVM.py "$folder/$file" &> "$out_dir/${file}/classical.txt" &
wait

# Combine all output files into one
cat "$out_dir/$file"/*.txt > "$out_dir/$file/outputs.txt"

#Remove all the files except output
find "$out_dir/$file"/* -type f -not -name 'outputs.txt' -delete

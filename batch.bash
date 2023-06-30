#!/bin/bash
#SBATCH --job-name=Test_job
#SBATCH --account=Project_2004128
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=large
##SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=80
#SBATCH --output=%A_%a.txt
#SBATCH --array=1-1

export SING_IMAGE="env.sif"
#export SING_FLAGS=--nv


# LOOP THROUGH THE FILES AND SUBMIT BATCH JOBS
folder="./datasets"  # Specify the folder path
file=$(ls -p "$folder" | sed -n "${SLURM_ARRAY_TASK_ID}p")
mkdir -p "outputs/$file"
rm "outputs/$file/"*
# Call the file twice with different parameters
# Define num wires
wires=(3 5 9 12)  # Add more parameters as needed
#Define layers
layers=(3 5 9 12)
#Define batch size
batch_size=5
#Define iterations for optimization
optim_iter=100
#Define when to stop optimization if not improving in 50s
prune_after=2
#Learning rates for the optimizer
lr_rates=(0.005 0.01 0.05 0.1 0.2 0.5)

for layer in "${layers[@]}"; do
        for wire in "${wires[@]}"; do
		for lr in "${lr_rates[@]}"; do
			srun --ntasks=1 -c 1 apptainer_wrapper exec \
				python3 \
				-u main.py \
				"$folder/$file" \
				$wire \
				$layer \
				$batch_size \
				$optim_iter \
				$prune_after \
				$lr \
				&> "outputs/${file}/${layer}_${wire}_${lr}.txt" &
		done
	done
done

srun --ntasks=1 -c 1 --exclusive apptainer_wrapper exec \
        python3 classicalSVM.py "$folder/$file" &> "outputs/${file}/classical.txt" &
wait

# Combine all output files into one
cat "outputs/$file"/*.txt > "outputs/$file/outputs.txt"

#Remove all the files except output
find "outputs/$file"/* -type f -not -name 'outputs.txt' -delete


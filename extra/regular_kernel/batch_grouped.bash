#!/bin/bash

export SING_IMAGE="../../../env.sif"

# Call the Python script to generate the combinations.txt
apptainer_wrapper exec python3 ../parameters.py

#wait for the python file to fully write out (is this really necessary?)
sleep 6

out_dir="../../../outputs/fullTest"
data_dir="../../datasets/train"

# Calculate the total array size for batch job
num_files=$(ls $data_dir | wc -l)
num_combinations=$(wc -l < combinations.txt)

# create output directories for all datasets
for file in ${data_dir}/*; do
    file_name=$(basename "$file")
    mkdir -p "${out_dir}/${file_name}"

    # Clear directory if not empty
    if [ "$(ls -A ${out_dir}/${file_name})" ]; then
            rm -r "${out_dir}/${file_name}"/*
    fi
done

# Submit the SLURM job with the calculated array size
sbatch --array=1-$num_files --ntasks=$num_combinations regular_kernel_grouped.bash $num_combinations $out_dir $data_dir


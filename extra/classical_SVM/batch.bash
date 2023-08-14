#!/bin/bash

export SING_IMAGE="../../../env.sif"


out_dir="../../../outputs/classical_svm"
data_dir="../../datasets/train"

# Calculate the total array size for batch job
num_files=$(ls $data_dir | wc -l)
total_size=$num_files

# create output directory
mkdir -p "${out_dir}"

# Clear directory if not empty
if [ "$(ls -A ${out_dir})" ]; then
    rm -r "${out_dir}"/*
fi

# Submit the SLURM job with the calculated array size
sbatch --array=1-$total_size classical_svm.bash $out_dir $data_dir
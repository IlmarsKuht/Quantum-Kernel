# Benchmarking Quantum Kernel SVM vs classical SVM
You can use the code just like htis, but it is meant for a SLURM environemnt.

# In the batch file:
- Add all parameters you want to test
- Change array to the size of all the datasets you have (if you have 3 datasets then 1-3)
- I have commented out the GPU version, if you want to use GPUS, then just uncomment it
(GPU only useful for around >15 qubits and >15 layers)

## How to use without batch file:
- python main.py {dir to dataset} {wire_count} {layer_count} {batch_size} {optim_iter} {prune_after} {LR}

## How to use with batch file:
- sbatch batch.bash

## Features:
- Creates a randomly initialized quantum kernel
- Aligns the kernel
- Tests the accuracy and compares vs sklearn all kernels

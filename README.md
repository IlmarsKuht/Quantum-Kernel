# Benchmarking Quantum Kernel SVM vs classical SVM
You can use the code just like this, but it is meant for a SLURM environemnt.

# In the batch file:
- Add all parameters you want to test
- Change array to the size of all the datasets you have (if you have 3 datasets then 1-3)
- I have commented out the GPU version, if you want to use GPUS, then just uncomment it (you also need to change the code, check below)

## How to use without batch file:
- python main.py {dir to dataset} {wire_count} {layer_count} {batch_size} {optim_iter} {prune_after} {LR}

## How to use with batch file:
- sbatch batch.bash

## Features:
- Creates a randomly initialized quantum kernel
- Aligns the kernel
- Tests the accuracy and compares vs sklearn all kernels


For GPU change the simulator in main.py to lightning.gpu
for CPU change the simulator in main.py to default.qubit/lightning.qubit
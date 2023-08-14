## Quantum kernel implementation

### For SLURM batch jobs
Disclaimer! You need to update the path to your container file.
the folder 'extra' contains bash scripts to interact with the SLURM environemnt

- All parameters for SLURM jobs can be changed in parameters.py
- Run the scripts with the command 'bash batch.bash', which is a wrapper for the other bash file.
- Run the bash script while in the same directory so the parameters.py generates the combinations.txt file in the correct directory

### How to use without a SLURM environment:
go to the src folder: python main.py --dataset-dir {path to file}

- If you want to change the parameters in the main.py file, pass them in as arguments e.g python main.py --dataset-dir ./train --lr 0.1 --optim_iter 1000
- You can switch to a different kernel by passing the --projected_kernel arg
- You can turn on kernel alignment by passing --align_kernel
- Some parameters are only required when doing kernel alignment or using projected kernel, everything can be seen in the main.py file

### GPU and CPU:

- For GPU use lightning.gpu simulator
If using SLURM environement uncomment in the bash file #SBATCH --gres=gpu and  #export SING_FLAGS=--nv

- For CPU use default.qubit/lightning.qubit and comment out the previously mentioned lines

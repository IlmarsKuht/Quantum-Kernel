import itertools

# Define your parameters
wires = [3]
layers = [3,4]
lr_rates = [0.1]
batch_sizes = [5]
optim_iters = [11]
prune_afters = [3]
acc_test_everys = [5]
gammas = [1]
# 1 is true, 0 is false for kernel
projected_kernel = [0,1]
align_kernel = [0,1]

# Generate all combinations
combinations = list(itertools.product(wires, layers, lr_rates, batch_sizes, optim_iters, prune_afters, acc_test_everys, gammas, projected_kernel, align_kernel))

# Write combinations to a file
with open('combinations.txt', 'w') as f:
    for combination in combinations:
        f.write(f'--num_wires {combination[0]} --num_layers {combination[1]} --lr {combination[2]} --batch_size {combination[3]} --optim_iter {combination[4]} --prune_after {combination[5]} --acc_test_every {combination[6]} --gamma {combination[7]} --projected_kernel {combination[8]} --align_kernel {combination[9]}\n')


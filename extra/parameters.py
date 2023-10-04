import itertools

# Define your parameters
wires = [5]
layers = [1,2,3,4,5,6,7,8]
lr_rates = [0.2]
batch_sizes = [5]
optim_iters = [3000]
prune_afters = [2]
acc_test_everys = [100]
gammas = [1]
seeds = [100,101,102,103,104]
# 1 is true, 0 is false or just write true or false, I don't control your life
projected_kernel = [0]
align_kernel = [1]
new_architectures = [0,1]

# Generate all combinations
combinations = list(itertools.product(wires, layers, lr_rates, batch_sizes, optim_iters, prune_afters, acc_test_everys, gammas, seeds, projected_kernel, align_kernel, new_architectures))

# Write combinations to a file
with open('combinations.txt', 'w') as f:
    for combination in combinations:
        f.write(f'--num_wires {combination[0]} --num_layers {combination[1]} --lr {combination[2]} --batch_size {combination[3]} --optim_iter {combination[4]} --prune_after {combination[5]} --acc_test_every {combination[6]} --gamma {combination[7]} --seed {combination[8]} --projected_kernel {combination[9]} --align_kernel {combination[10]} --new_architecture {combination[11]}\n')


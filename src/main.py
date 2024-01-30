import argparse
import pennylane.numpy as np

from auxilary.data import load_dataset
from kernels.kernel import KernelBase
from kernels.projectedKernel import ProjectedKernel

def str2bool(v : str) -> bool:
    """Convert a string to a boolean

    Args:
        v (str): input string

    Raises:
        argparse.ArgumentTypeError: string not a boolean

    Returns:
        bool: 
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Kernel training")

    parser.add_argument('--dataset_dir', type=str, default="./datasets/train/iris_small.csv", help='Directory of the dataset')
    parser.add_argument('--num_wires', type=int, default=3, help='Number of wires')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--seed', type=int, default=42, help='Randomization seed')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--optim_iter', type=int, default=500, help='Number of optimization iterations')
    parser.add_argument('--acc_test_every', type=int, default=50, help='Test data accuracy every x iterations')
    parser.add_argument('--prune_after', type=int, default=2, help='Prune after prune_after*acc_test_every iterations with no improvement')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='Gamma value for projected kernel')
    parser.add_argument('--align_kernel', type=str2bool, nargs='?', const=True, default=False, help="Whether to to align the kernel")
    parser.add_argument('--projected_kernel', type=str2bool, nargs='?', const=True, default=False, help="Whether to use projected kernel")
    parser.add_argument('--new_architecture', type=str2bool, nargs='?', const=True, default=False, help="Whether to use new or old architecture stacks")
    parser.add_argument('--linear_kernel', type=str2bool, nargs='?', const=True, default=False, help="Whether to use linear kernel (no transformation)")
    parser.add_argument('--x', type=str2bool, nargs='?', const=True, default=False, help="Use X, XFX, XFXFX Architecture")
    
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    num_wires = args.num_wires
    num_layers = args.num_layers
    seed = args.seed

    #if kernel aligning
    batch_size = args.batch_size
    optim_iter = args.optim_iter
    acc_test_every = args.acc_test_every
    prune_after = args.prune_after
    lr = args.lr

    #if using projected kernel
    gamma = args.gamma

    #others
    align_kernel = args.align_kernel
    projected_kernel = args.projected_kernel
    new_architecture = args.new_architecture
    linear_kernel = args.linear_kernel
    x = args.x

    np.random.seed(seed) 

    x_train, x_test, y_train, y_test = load_dataset(dataset_dir, normalize=True)

    if linear_kernel:
        num_wires = len(x_train[0])

    print("---------------------------")
    print(f"Dataset Directory: {dataset_dir} | Wires: {num_wires} | Layers: {num_layers} | Seed: {seed}")
    print(f"Batch Size: {batch_size} | Optim Iterations: {optim_iter} | Acc Test Every: {acc_test_every} | Prune After: {prune_after} | Learning Rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Align Kernel: {align_kernel} | Projected Kernel: {projected_kernel} | New Architecture: {new_architecture} | Linear Kernel: {linear_kernel} | X Architecture: {x}")

    if projected_kernel:
        kernel = ProjectedKernel(num_wires, num_layers, batch_size, optim_iter,
                                    acc_test_every, prune_after, lr, new_architecture,
                                    align_kernel, linear_kernel, x,
                                    gamma)
    else:
        kernel = KernelBase(num_wires, num_layers, batch_size, optim_iter,
                                acc_test_every, prune_after, lr, new_architecture,
                                align_kernel, linear_kernel, x)
    if align_kernel:
        kernel.train_and_align(x_train, x_test, y_train, y_test)
    else:
        kernel.train(x_train, x_test, y_train, y_test)


    print("---------------------------")


if __name__ == "__main__":
    main()

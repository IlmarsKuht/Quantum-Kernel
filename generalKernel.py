import pennylane as qml
from pennylane import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import time
import sys
import os

np.random.seed(42)
train_dir = sys.argv[1]
num_wires = int(sys.argv[2])
num_layers = int(sys.argv[3])
batch_size = int(sys.argv[4])
optim_iter = int(sys.argv[5])
prune_after = int(sys.argv[6])
lr = float(sys.argv[7])

#Change to lightning.gpu if you want to use GPU
simulator = "lightning.qubit"

#DEFINING THE CIRCUIT

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))

#check if I need batch_obs
dev = qml.device(simulator, wires=num_wires, shots=None)
wires = dev.wires.tolist()

#I use adjoint so it works for both default.qubit and lightning.gpu (backprop doesn't work for lightning.gpu)
@qml.qnode(dev, interface="autograd", diff_method="adjoint") 
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    qml.adjoint(ansatz)(x2, params, wires=wires)
    return [qml.expval(qml.PauliZ(w)) for w in wires]

def kernel(x1, x2, params):
    expectations = kernel_circuit(x1, x2, params)
    return np.prod(expectations)



#END OF DEFINING THE CIRCUIT


# AUXILARY FUNCTIONS

def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

#Prints a custom message in description
#The alignment of the kernel
#The accuracy of the classifier
def printInfo(datasets, kernel):
    start = time.time()
    total_acc = 0
    for dataset, name in datasets:
        x_train, x_test, y_train, y_test = dataset

        alignment = target_alignment(x_test, y_test, kernel)
        print(f"kernel alignment: {alignment:.3f}")

        svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, kernel)).fit(x_train, y_train)

        acc = accuracy(svm, x_test, y_test)
        print(f"The accuracy for {name} is  {acc:.3f}")
        total_acc += acc

    print(f"The average accuracy for test datasets is {total_acc/len(datasets):.3f}")
    print(f"Time taken for printInfo: {time.time()-start}")


def cosine_similarity(A, B):
    A = A.flatten()
    B = B.flatten()
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def target_alignment(X, Y, kernel):
    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=True,
    )
    #Target kernel 1s where class is the same and 0 otherwise
    T = (Y[:, None] == Y[None, :]).astype(float)
    #turn them into vectors for cosine similarity
    K = K.reshape(1, -1)
    T = T.reshape(1, -1)

    #Returns similarity 1 being identical -1 being opposite
    return cosine_similarity(K, T)

# END OF AUXILARY FUNCTIONS


#DATASET LOADING AND PROCESSING

files = os.listdir(train_dir)
datasets = []

for file in files:
    file_path = os.path.join(train_dir, file)
    
    df = pd.read_csv(file_path)
    dataset = np.array(df, requires_grad=False)
    X = dataset[:, :-1]
    Y = dataset[:, -1]

    #fit the features between 0 and 1
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    datasets.append(((x_train, x_test, y_train, y_test), file))

#END OF DATASET LOADING AND PROCESSING


#START OF MAIN FILE

def trainSVM(datasets):
    print(f"Wires: {num_wires} | Layers: {num_layers} | Batch_Size: {batch_size} | iterations: {optim_iter} | lr: {lr}")

    params = random_params(num_wires=num_wires, num_layers=num_layers)

    init_kernel = lambda x1, x2: kernel(x1, x2, params)
    printInfo(datasets, init_kernel)

    # Kernel alignment

    opt = qml.GradientDescentOptimizer(lr)
    alignments = []
    
    start = time.time()

    for i in range(optim_iter):
        # counter to track when to stop the optimization
        counter = 0
        init_kernel = lambda x1, x2: kernel(x1, x2, params)

        def cost(_datasets):
            total_cost = 0
            for dataset, _ in _datasets:
                x_train, _, y_train, _ = dataset
                subset = np.random.choice(list(range(len(x_train))), batch_size, requires_grad=False)
                total_cost += -target_alignment(x_train[subset],y_train[subset], init_kernel)
            return total_cost / len(_datasets)
        
        params, curr_cost = opt.step_and_cost(cost, datasets)

        if (i + 1) % 50 == 0:
            curr_alignment = 0
            for x_train, _, y_train, _ in datasets:
                subset = np.random.choice(list(range(len(x_train))), batch_size, requires_grad=False)
                curr_alignment += -target_alignment(x_train[subset],y_train[subset], init_kernel)
            curr_alignment /= len(datasets)

            alignments.append(curr_alignment)
            # if we've had at least x iterations, check for improvement
            if len(alignments) >= prune_after:
                if all([curr_alignment <= c for c in alignments[-prune_after:]]):
                    counter += 1
                else:
                    counter = 0
            # if we haven't improved for x iterations, stop
            print(curr_alignment)
            if counter >= prune_after:
                print(f"Stopping optimization, the cost hasn't improved for {i+1} iterations.")
                break

    end = time.time()
    alignment_time = end-start

    init_kernel = lambda x1, x2: kernel(x1, x2, params)
    printInfo(datasets, init_kernel)
    

trainSVM(datasets)
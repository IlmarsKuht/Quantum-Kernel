import pennylane as qml
from pennylane import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas

import time
import sys
import os

np.random.seed(42)

num_wires = int(sys.argv[2])
num_layers = int(sys.argv[3])
batch_size = int(sys.argv[4])
optim_iter = int(sys.argv[5])
prune_after = int(sys.argv[6])
lr = float(sys.argv[7])

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
dev = qml.device("lightning.gpu", wires=num_wires, shots=None)
wires = dev.wires.tolist()

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
def printInfo(description, X, Y, classifier, kernel):
    print(description)
    alignment = target_alignment(X, Y, kernel)
    print(f"kernel alignment: {alignment:.3f}")
    accuracy_init = accuracy(classifier, X, Y)
    print(f"The accuracy of the kernel for dataset {name} is {accuracy_init:.3f}")


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

    #Returns similarity - 1 being identical -1 being opposite
    return cosine_similarity(K, T)

# END OF AUXILARY FUNCTIONS


#DATASET LOADING AND PROCESSING

#directory of the dataset is supposed to be passed as an argument
directory = sys.argv[1]
name = os.path.basename(directory)
dataset = np.array(pandas.read_csv(directory), requires_grad=False)

X = dataset[:, :-1]
Y = dataset[:, -1]

#fit the features between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#END OF DATASET LOADING AND PROCESSING


#START OF MAIN FILE

def trainSVM(x_train, x_test, y_train, y_test):
    print(f"Wires: {num_wires} | Layers: {num_layers} | Batch_Size: {batch_size} | iterations: {optim_iter} | lr: {lr}")

    params = random_params(num_wires=num_wires, num_layers=num_layers)
    init_kernel = lambda x1, x2: kernel(x1, x2, params)

    # NOW ALIGN THE KERNEL TO TRY TO IMPROVE THE PERFORMANCE

    opt = qml.GradientDescentOptimizer(lr)

    alignments = []
    
    start = time.time()
    for i in range(optim_iter):
        # counter to track when to stop the optimization
        counter = 0
        subset = np.random.choice(list(range(len(x_train))), batch_size, requires_grad=False)
        cost = lambda _params: -target_alignment(
            x_train[subset],
            y_train[subset],
            lambda x1, x2: kernel(x1, x2, _params)
        )
        params, curr_cost = opt.step_and_cost(cost, params)

        if (i + 1) % 100 == 0:
            curr_alignment = target_alignment(
                x_test,
                y_test,
                lambda x1, x2: kernel(x1, x2, params)
            )
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

    start = time.time()
    svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(x_train, y_train)
    end = time.time()
    fitting_time = end-start
    
    printInfo(f"Finished kernel alignment in {alignment_time:.3f} seconds \n \
               Finished fitting in {fitting_time:.3f} seconds",
                x_test, y_test, svm, init_kernel)
    
    

print(f"Training dataset {name}")

trainSVM(x_train, x_test, y_train, y_test)


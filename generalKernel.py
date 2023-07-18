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
@qml.qnode(dev, interface="autograd", diff_method="finite-diff") 
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    qml.adjoint(ansatz)(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]



#END OF DEFINING THE CIRCUIT


# AUXILARY FUNCTIONS

def random_params(num_wires, num_layers):
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)
#Prints a custom message in description
#The alignment of the kernel
#The accuracy of the classifier
def printInfo(description, datasets, classifiers, kernel):
    print(description)
    total_alignment = 0
    total_accuracy = 0
    for i, ((X, Y), name) in enumerate(datasets):
        alignment = target_alignment(X, Y, kernel)
        total_alignment += alignment
        curr_accuracy = accuracy(classifiers[i], X, Y)
        total_accuracy += curr_accuracy
        print(f"Alignment for dataset {name}: {alignment:.4f}")
        print(f"Accuracy: {curr_accuracy:.3f}")

    print(f"Total alignment: {total_alignment/len(datasets):.4f}")
    print(f"Total accuracy {total_accuracy/len(datasets):.3f}")

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
train_data = []
test_data = []

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
    train_data.append(((x_train, y_train), file))
    test_data.append(((x_test, y_test), file))

#END OF DATASET LOADING AND PROCESSING


#START OF MAIN FILE

def trainSVM(train_data, test_data):
    print(f"Wires: {num_wires} | Layers: {num_layers} | Batch_Size: {batch_size} | iterations: {optim_iter} | lr: {lr}")

    params = random_params(num_wires=num_wires, num_layers=num_layers)
    init_kernel = lambda x1, x2: kernel(x1, x2, params)
    classifiers = []

    start = time.time()
    for (X, Y), _ in train_data:
        classifiers.append(SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y))

    printInfo("Before kernel alignment", test_data, classifiers, init_kernel)

    end = time.time()
    print(f"Training and accuracy testing time: {end-start:.3f}")

    # Kernel alignment
    opt = qml.GradientDescentOptimizer(lr)
    alignments = []
    max_alignment = -1
    counter = 0
    start = time.time()

    for i in range(optim_iter):
        # counter to track when to stop the optimization
        
        init_kernel = lambda x1, x2: kernel(x1, x2, params)

        def cost(curr_params):
            init_kernel = lambda x1, x2: kernel(x1, x2, curr_params)
            total_cost = 0
            for (X, Y), _ in train_data:
                subset = np.random.choice(list(range(len(X))), batch_size, requires_grad=False)
                total_cost += -target_alignment(X[subset], Y[subset], init_kernel)
            return total_cost / len(train_data)
        
        params = opt.step(cost, params)

        if (i + 1) % 1000 == 0:
            init_kernel = lambda x1, x2: kernel(x1, x2, params)
            curr_alignment = 0
            for (X, Y), _ in test_data:
                curr_alignment += target_alignment(X, Y, init_kernel)
            curr_alignment /= len(test_data)
            
            if curr_alignment > max_alignment:
                max_alignment = curr_alignment
                counter = 0
            else:
                counter += 1 
            alignments.append(curr_alignment)
            
            # if we haven't improved for x iterations, stop
            print(f"{curr_alignment:.4f}")
            if counter >= prune_after:
                print(f"Stopping optimization, the cost hasn't improved for {counter*1000} iterations.")
                break

        #check accuracy
        if i % 70000 == 0:
            print(i)
            init_kernel = lambda x1, x2: kernel(x1, x2, params)
            classifiers = []

            for (X, Y), _ in train_data:
                classifiers.append(SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y))

            printInfo("After kernel alignment", test_data, classifiers, init_kernel)
            np.save(f"./saved_params/{num_wires}_{num_layers}_{lr}_{i}.npy", params)

    end = time.time()
    alignment_time = end-start
    print(f"Kernel alignment time: {alignment_time:.3f}")

    init_kernel = lambda x1, x2: kernel(x1, x2, params)
    classifiers = []

    for (X, Y), _ in train_data:
        classifiers.append(SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y))

    printInfo("After kernel alignment", test_data, classifiers, init_kernel)
    
trainSVM(train_data, test_data)
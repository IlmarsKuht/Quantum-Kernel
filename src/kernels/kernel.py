import time
import pennylane as qml
from sklearn.svm import SVC
import pennylane.numpy as np


class KernelBase:
    def __init__(self, num_wires, num_layers, batch_size, optim_iter, acc_test_every, prune_after, lr):
        
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.optim_iter = optim_iter
        self.acc_test_every = acc_test_every
        self.prune_after = prune_after
        self.lr = lr
        self.dev = qml.device("lightning.qubit", wires=self.num_wires, shots=None)
        self.params = self.random_params()

    def random_params(self):
        return np.random.uniform(0, 2 * np.pi, (self.num_layers, 2, self.num_wires), requires_grad=True)

    def accuracy(self, classifier, X, Y_target):
        return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

    def printInfo(self, description, X, Y, classifier, kernel):
        print(description)
        alignment = self.target_alignment(X, Y, kernel)
        print(f"kernel alignment: {alignment:.3f}")
        accuracy_init = self.accuracy(classifier, X, Y)
        print(f"The accuracy of the kernel is {accuracy_init:.3f}")

    def cosine_similarity(self, A, B):
        A = A.flatten()
        B = B.flatten()
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def target_alignment(self, X, Y, kernel):
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
        return self.cosine_similarity(K, T)
    

    #CIRCUIT STUFF
    def layer(self, x, params, wires, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for j, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            qml.RY(params[0, j], wires=[wire])
            
            qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

    def ansatz(self, x, params, wires):
        """The embedding ansatz"""
        for j, layer_params in enumerate(params):
            self.layer(x, layer_params, wires, i0=j * len(wires))

    def kernel_circuit(self, x1, x2, params):
        # Define the circuit that will be turned into a QNode
        wire_list = range(self.num_wires)
        def circuit(x1, x2):
            self.ansatz(x1, params, wires=wire_list)
            qml.adjoint(self.ansatz)(x2, params, wires=wire_list)
            return qml.probs(wires=wire_list)
        
        # Create the QNode
        qnode = qml.QNode(circuit, self.dev, interface="autograd", diff_method="finite-diff")
        
        # Run and return the result of the QNode
        return qnode(x1, x2)

    def kernel(self, x1, x2, params):
        return self.kernel_circuit(x1, x2, params)[0]
    
    # END OF CIRCUIT STUFF

    def train(self, x_train, x_test, y_train, y_test):

        init_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)

        start = time.time()
        svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(x_train, y_train)
        end = time.time()
        fitting_time = end-start
        
        self.printInfo(f"Finished fitting in {fitting_time:.3f} seconds",
                    x_test, y_test, svm, init_kernel)
        
    def train_and_align(self, x_train, x_test, y_train, y_test):

        opt = qml.GradientDescentOptimizer(self.lr)
        alignments = []
        start = time.time()
        counter = 0
        max_alignment = -1
        for i in range(self.optim_iter):
            # counter to track when to stop the optimization
            
            subset = np.random.choice(list(range(len(x_train))), self.batch_size, requires_grad=False)
            cost = lambda _params: -self.target_alignment(
                x_train[subset],
                y_train[subset],
                lambda x1, x2: self.kernel(x1, x2, _params)
            )
            self.params = opt.step(cost, self.params)

            if (i + 1) % self.acc_test_every == 0:
                curr_alignment = self.target_alignment(
                    x_test,
                    y_test,
                    lambda x1, x2: self.kernel(x1, x2, self.params)
                )
                if curr_alignment > max_alignment:
                    max_alignment = curr_alignment
                    counter = 0
                else:
                    counter += 1 
                alignments.append(curr_alignment)
                
                print(curr_alignment)
                if counter >= self.prune_after:
                    print(f"Stopping optimization, the cost hasn't improved for {counter*self.acc_test_every} iterations.")
                    break

        end = time.time()
        alignment_time = end-start

        init_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)

        start = time.time()
        svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(x_train, y_train)
        end = time.time()
        fitting_time = end-start
        
        self.printInfo(f"Finished kernel alignment in {alignment_time:.3f} seconds \n \
                   Finished fitting in {fitting_time:.3f} seconds",
                    x_test, y_test, svm, init_kernel)
    


        


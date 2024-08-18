import time
import pennylane as qml
from sklearn.svm import SVC
import pennylane.numpy as np
from typing import Callable

from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel


class KernelBase:
    """Basic kernel implementation
    """
    def __init__(self, num_wires: int, num_layers: int, batch_size: int,
                 optim_iter: int, acc_test_every: int, prune_after: int, lr: float,
                 new_architecture: bool, align_kernel: bool,
                 linear_kernel: bool, x: bool):
        """Initialize kernel

        Args:
            num_wires (int): number of wires/qubits in the circuit
            num_layers (int): ansatz layers
            batch_size (int): samples to use in one training iteration
            optim_iter (int): max number of training iterations
            acc_test_every (int): test on testing data every x iterations   
            prune_after (int): stop training after prune_after * acc_test_every iterations
            lr (float): Learning rate for the optimizer
        """
        
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.optim_iter = optim_iter
        self.acc_test_every = acc_test_every
        self.prune_after = prune_after
        self.lr = lr
        self.new_architecture = new_architecture
        self.align_kernel = align_kernel
        self.linear_kernel = linear_kernel
        self.x = x
        self.dev = qml.device("lightning.qubit", wires=self.num_wires, shots=None)
        self.params = self.random_params()

    def random_params(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: shape: (self.num_layers, 2, self.num_wires) of random parameters
        """
        if self.x:
            return np.random.uniform(0, 2 * np.pi, (self.num_layers-1, 2, self.num_wires), requires_grad=True)
        else:
            return np.random.uniform(0, 2 * np.pi, (self.num_layers, 2, self.num_wires), requires_grad=True)

    def accuracy(self, classifier: SVC, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Args:
            classifier (SVC): sklearn SVC (but anything with .predict() method works)
            X (np.ndarray): input features
            Y (np.ndarray): target labels

        Returns:
            float: accuracy of the model vs target labels between 0 and 1
        """
        prediction = classifier.predict(X)
        return (prediction == Y).sum() / len(Y)
    
    def geometric_diff(self, k1: np.ndarray, k2: np.ndarray) -> float:
        """Calculate geometric difference between two kernels

        Args:
            k1 (np.ndarray): kernel 1
            k2 (np.ndarray): kernel 2

        Returns:
            float: geometric difference
        """
        sqrt_k1 = np.sqrt(k1)
        inv_k2 = np.linalg.inv(k2)
        diff = sqrt_k1 @ inv_k2 @ sqrt_k1
        return np.linalg.norm(diff, 'fro')  # Frobenius norm


    def printInfo(self, X: np.ndarray, Y: np.ndarray,
                  classifier: SVC, kernel: Callable[[np.ndarray, np.ndarray], float]):
        """Print given description, kernel alignment and accuracy

        Args:
            X (np.ndarray): input features
            Y (np.ndarray): target labels
            classifier (SVC): sklearn SVC (but anything with .predict() method works)
            kernel (Callable[[np.ndarray, np.ndarray], float]): kernel function with parameters already provided
        """
        alignment = self.target_alignment(X, Y, kernel)
        print(f"kernel alignment: {alignment:.3f}")
        accuracy_init = self.accuracy(classifier, X, Y)
        print(f"The accuracy of the kernel is {accuracy_init:.3f}")

        # Would be nice to let user change the params of the kernels
        # k1 = {"value": rbf_kernel(X), "name": "rbf"}
        # k2 = {"value": polynomial_kernel(X), "name": "poly"}
        # k3 = {"value": sigmoid_kernel(X), "name": "sigmoid"}

        # kernel = qml.kernels.square_kernel_matrix(
        #     X,
        #     kernel,
        #     assume_normalized_kernel=True,
        # )

        # for k in [k1, k2, k3]:
        #     print(f"kernel {k['name']} geometric difference vs Quantum kernel is {self.geometric_diff(k['value'], kernel)}")

    #this is a very primitive similarity measure, try other ones
    def cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        """returns similarity between two kernels

        Args:
            A (np.ndarray): kernel matrix A
            B (np.ndarray): kernel matrix B

        Returns:
            float: The higher the value the more similar (in range -1 to 1)
            IMPORTANT!!! for non negative vectors the range is 0 to 1
        """

        #flatten for cosine similarity
        A = A.flatten()
        B = B.flatten()
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def target_alignment(self, X: np.ndarray, Y: np.ndarray,
                         kernel: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """Improvement function for calculating how far off the target the kernel function is

        Args:
            X (np.ndarray): input features
            Y (np.ndarray): target labels
            kernel (Callable[[np.ndarray, np.ndarray], float]): kernel value calculator for two inputs

        Returns:
            float: depending on the measure, returns similarity between current kernel and target
        """
        K = qml.kernels.square_kernel_matrix(
            X,
            kernel,
            assume_normalized_kernel=True,
        )
        #Target kernel, if the class is the same then value is 1, otherwise 0
        T = (Y[:, None] == Y[None, :]).astype(float)
        #T = (Y[:, None] == Y[None, :]).astype(float) * 2 - 1
        #If you need values 1 otherwise -1 use this instead ^^
    
        return self.cosine_similarity(K, T)
    

    #CIRCUIT STUFF
    def layer_xf(self, x: np.ndarray, params: np.ndarray, i0: int=0, inc: int=1):
        """Ansatz building block

        Args:
            x (np.ndarray): input features
            params (np.ndarray): parameters/weights of the circuit
            i0 (int, optional): keeps track of embedded features to know which to embbed next. Defaults to 0.
            inc (int, optional): embbed features every x qubit. Defaults to 1.
        """
        i = i0
        wire_list = range(self.num_wires)
        for j, wire in enumerate(wire_list):
            #superpositions
            qml.Hadamard(wires=[wire])
            #data embedding
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            #parameterized rotations
            qml.RY(params[0, j], wires=[wire])

        #entanglement
        qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wire_list, parameters=params[1])

    def layer_fx(self, x: np.ndarray, params: np.ndarray, i0: int=0, inc: int=1):
        """Ansatz building block

        Args:
            x (np.ndarray): input features
            params (np.ndarray): parameters/weights of the circuit
            i0 (int, optional): keeps track of embedded features to know which to embed next. Defaults to 0.
            inc (int, optional): embbed features every x qubit. Defaults to 1.
        """
        wire_list = range(self.num_wires)
        for j, wire in enumerate(wire_list):
            #parameterized rotations
            qml.RY(params[0, j], wires=[wire])

        #entanglement
        qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wire_list, parameters=params[1])

        i = i0
        for j, wire in enumerate(wire_list):
            #superpositions
            qml.Hadamard(wires=[wire])
            #data embedding
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
        

    #test quantum kernel with a linear kernel
    def linear_layer(self, x: np.ndarray):
        for j, val in enumerate(x):
            qml.RZ(val, wires=[j])

    def ansatz(self, x: np.ndarray, params: np.ndarray):
        """Applies layers according to parameter amount

        Args:
            x (np.ndarray): input features
            params (np.ndarray): paramteres/weights for the circuit
        """
        if self.x:
            wire_list = range(self.num_wires)
            i = 0
            for j, wire in enumerate(wire_list):
                qml.Hadamard(wires=[wire])
                #data embedding
                qml.RZ(x[i % len(x)], wires=[wire])
                i += 1
        for j, layer_params in enumerate(params):
            if self.new_architecture:
                self.layer_fx(x, layer_params, i0=j * len(range(self.num_wires)))
            elif self.x:
                self.layer_fx(x, layer_params, i0=(j+1) * len(range(self.num_wires)))
            elif self.linear_kernel:
                self.linear_layer(x)
            else:
                self.layer_xf(x, layer_params, i0=j * len(range(self.num_wires)))

    def kernel_circuit(self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray):
        """Runs the circuit with given features and parameters

        Args:
            x1 (np.ndarray): input feature 1
            x2 (np.ndarray): input feature 2
            params (np.ndarray): parameters/weights for the circuit

        Returns:
            array: probabilites of each state
        """

        # Define the circuit that will be turned into a QNode
        def circuit(x1, x2):
            self.ansatz(x1, params)
            qml.adjoint(self.ansatz)(x2, params)
            return qml.probs(wires=range(self.num_wires))
            
        qnode = qml.QNode(circuit, self.dev, interface="autograd", diff_method="finite-diff")
        # #Create a drawable version of the QNode
        # drawable_circuit = qml.draw(qnode)

        # # Call the drawable circuit with the same arguments and print
        # print(drawable_circuit(x1, x2))
        return qnode(x1, x2)

    def kernel(self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> float:
        """kernel function for two features

        Args:
            x1 (np.ndarray): input feature 1
            x2 (np.ndarray): input feature 2
            params (np.ndarray): parameters/weights for the circuit

        Returns:
            float: probability of all 0s state
        """
        return self.kernel_circuit(x1, x2, params)[0]
    
    # END OF CIRCUIT STUFF

    def train(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """fit an SVM with the kernel

        Args:
            x_train (np.ndarray): training features
            x_test (np.ndarray): testing features
            y_train (np.ndarray): training labels
            y_test (np.ndarray): testing labels
        """
        init_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)

        start = time.time()
        svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(x_train, y_train)
        end = time.time()
        fitting_time = end-start
        
        print(f"Finished fitting in {fitting_time:.3f} seconds")
        self.printInfo(x_test, y_test, svm, init_kernel)
        
    def train_and_align(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """fit an SVM with the kernel and optimize the parameters of the kernel

        Args:
            x_train (np.ndarray): training features
            x_test (np.ndarray): testing features
            y_train (np.ndarray): training labels
            y_test (np.ndarray): testing labels
        """
        opt = qml.GradientDescentOptimizer(self.lr)
        alignments = []
        start = time.time()
        counter = 0
        max_alignment = -1
        for i in range(self.optim_iter):
            
            subset = np.random.choice(list(range(len(x_train))), self.batch_size, requires_grad=False)
            cost = lambda _params: -self.target_alignment(
                x_train[subset],
                y_train[subset],
                lambda x1, x2: self.kernel(x1, x2, _params)
            )
            self.params = opt.step(cost, self.params)

            if (i + 1) % self.acc_test_every == 0:
                # curr_alignment = self.target_alignment(
                #     x_test,
                #     y_test,
                #     lambda x1, x2: self.kernel(x1, x2, self.params)
                # )
                # if curr_alignment > max_alignment:
                #     max_alignment = curr_alignment
                #     counter = 0
                # else:
                #     counter += 1 
                # alignments.append(curr_alignment)
                
                # print(curr_alignment)

                init_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)
                svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(x_train, y_train)
                self.printInfo(x_test, y_test, svm, init_kernel)

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
        
        print(f"Finished kernel alignment in {alignment_time:.3f} seconds\
                \nFinished fitting in {fitting_time:.3f} seconds")
        self.printInfo(x_test, y_test, svm, init_kernel)

    #I don't know what is this, some test I wanted to do?
    def full_test(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        #First a test of full qubits the same amount as there are features
        #One test where normalization is done and where there is no normalization. 
        # I assume no normalization should be very bad
        self.num_wires = len(x_train[0]) #check if it is a feature set
        self.dev = qml.device("lightning.qubit", wires=self.num_wires, shots=None)
        self.params = self.random_params()

        def circuit(x1, x2):
            self.testLayer(x1)
            qml.adjoint(self.testLayer(x2))
            return qml.probs(wires=range(self.num_wires))
        
        qnode = qml.QNode(circuit, self.dev, interface="autograd", diff_method="finite-diff")
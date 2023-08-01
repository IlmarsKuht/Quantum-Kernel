import pennylane as qml
import pennylane.numpy as np

from .kernel import KernelBase

class ProjectedKernel(KernelBase):
    """Kernel value calculated in classical space not Hilbert
    """
    def __init__(self, num_wires, num_layers, batch_size, optim_iter, acc_test_every, prune_after, lr, gamma: float=1.0):
        """Initialize Projected kernel

        Args:
            All the same as base kernel except:
            gamma (float, optional): Non-negative hyperparameter, affects the resulting kernel value. Defaults to 1.
        """
        super().__init__(num_wires, num_layers, batch_size, optim_iter, acc_test_every, prune_after, lr)
        
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma

    def kernel_circuit(self, x: np.ndarray, params: np.ndarray):
        """Run the input features through the circuit with params

        Args:
            x (np.ndarray): input features
            params (np.ndarray): parameters/weights for circuit

        Returns:
            array: expectation values of each rotation for each qubit separately
            These are separate local measurements for each qubit, not global
        """

        # Define the circuit that will be turned into a QNode
        def circuit(x):
            self.ansatz(x, params)
            return [qml.expval(P(wire)) for wire in range(self.num_wires) for P in [qml.PauliX, qml.PauliY, qml.PauliZ]]
        
        qnode = qml.QNode(circuit, self.dev, interface="autograd", diff_method="finite-diff")
        
        return qnode(x)

    def kernel(self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> float:
        """Projected kernel value
        Expectation values are projected to classical space, where kernel value is calculated

        Args:
            x1 (np.ndarray): input feature 1
            x2 (np.ndarray): input feature 2
            params (np.ndarray): parameters/weights for the circuit

        Returns:
            float: Simple Gaussian kernel bounded between 0 and 1
        """
        expectations_x1 = np.array(self.kernel_circuit(x1, params))
        expectations_x2 = np.array(self.kernel_circuit(x2, params))
            
        diff = expectations_x1 - expectations_x2
        sum_sq_diff = np.sum(diff**2)
        kernel_value = np.exp(-self.gamma * sum_sq_diff)

        return kernel_value
import pennylane as qml
import pennylane.numpy as np

from .kernel import KernelBase

class ProjectedKernel(KernelBase):
    def __init__(self, num_wires, num_layers, batch_size, optim_iter, acc_test_every, prune_after, lr, gamma=1):
        super().__init__(num_wires, num_layers, batch_size, optim_iter, acc_test_every, prune_after, lr)

        self.gamma = gamma

    def kernel_circuit(self, x, params):
        # Define the circuit that will be turned into a QNode
        wire_list = range(self.num_wires)
        def circuit(x):
            self.ansatz(x, params, wires=wire_list)
            return [qml.expval(P(wire)) for wire in wire_list for P in [qml.PauliX, qml.PauliY, qml.PauliZ]]
        # Create the QNode
        qnode = qml.QNode(circuit, self.dev, interface="autograd", diff_method="finite-diff")
        
        # Run and return the result of the QNode
        return qnode(x)

    def kernel(self, x1, x2, params):
        expectations_x1 = np.array(self.kernel_circuit(x1, params))
        expectations_x2 = np.array(self.kernel_circuit(x2, params))
            
        # Compute the squared differences and sum them up
        diff = expectations_x1 - expectations_x2
        sum_sq_diff = np.sum(diff**2)

        # Return the Gaussian of the sum of squared differences
        kernel_value = np.exp(-self.gamma * sum_sq_diff)
        return kernel_value
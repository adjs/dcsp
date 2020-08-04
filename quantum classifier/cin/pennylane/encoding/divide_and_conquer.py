import pennylane as qml
from pennylane import numpy as np

class bin_heap:
    size = None
    values = None

    def __init__(self, values):
        self.size = len(values)
        self.values = values

    def parent(self, key):
        return int((key-0.5)/2)

    def left(self, key):
        return int(2 * key + 1)

    def right(self, key):
        return int(2 * key + 2)

    def root(self):
        return 0

    def __getitem__(self, key):
        return self.values[key]


class Encoding:
    quantum_data = None
    classical_data = None
    num_qubits = None
    heap = None
    output_qubits = []

    def __init__(self, input_vector, encode_type='amplitude_encoding', entangle=False):
        self.output_qubits = []
        if encode_type == 'dc_amplitude_encoding':
            self.dc_amplitude_encoding(input_vector, entangle=entangle)

    @staticmethod
    def _recursive_compute_beta(input_vector, betas):
        if len(input_vector) > 1:
            new_x = []
            beta = []
            for k in range(0, len(input_vector), 2):
                norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2) # pylint: disable=no-member
                new_x.append(norm)
                if norm == 0:
                    beta.append(0)
                else:
                    if input_vector[k] < 0:
                        beta.append(2 * np.pi - 2 * np.arcsin(input_vector[k + 1] / norm)) # pylint: disable=no-member
                    else:
                        beta.append(2 * np.arcsin(input_vector[k + 1] / norm)) # pylint: disable=no-member
            Encoding._recursive_compute_beta(new_x, betas)
            betas.append(beta)

    def dc_amplitude_encoding(self, input_vector, entangle):
        self.num_qubits = int(len(input_vector))-1
        self.quantum_data = range(self.num_qubits)
        newx = np.copy(input_vector) # pylint: disable=no-member
        betas = []
        Encoding._recursive_compute_beta(newx, betas)
        self._dc_generate_circuit(betas, self.quantum_data, entangle)

    def _dc_generate_circuit(self, betas, quantum_input, entangle):
        '''
        doc
        '''
        k = 0
        linear_angles = []
        for angles in betas:
            linear_angles = linear_angles + angles
            for angle in angles:
                qml.RY(angle, wires=quantum_input[k])
                k += 1

        self.heap = bin_heap(quantum_input)
        my_heap = self.heap

        last = my_heap.size - 1
        actual = my_heap.parent(last)
        level = my_heap.parent(last)
        while actual >= 0:
            left_index = my_heap.left(actual)
            right_index = my_heap.right(actual)
            while right_index <= last:
                #if ((my_heap[left_index]==3 and my_heap[right_index]==5)==False):
                qml.CSWAP(wires=[my_heap[actual], my_heap[left_index], my_heap[right_index]])
                left_index = my_heap.left(left_index)
                right_index = my_heap.left(right_index)
            actual -= 1
            if level != my_heap.parent(actual):
                level -= 1

        # set output qubits
        next_index = 0
        while next_index < my_heap.size:
            self.output_qubits.append(next_index)
            next_index = my_heap.left(next_index)


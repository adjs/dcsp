import pennylane as qml
from pennylane import numpy as np
from cin.pennylane.qml.hierarchical_classifier import circuit as circuit_hierarchical_classifier
from cin.pennylane.encoding.divide_and_conquer import Encoding

def config(X):
    n = int(np.ceil(np.log2(len(X[0]))))             # pylint: disable=no-member
    n = 2**int(np.ceil(np.log2(n)))                  # n tem que ser potência de 2. # pylint: disable=no-member
    N = 2**n-1                                       # len(X[0])-1 # N precisa ser tal que n seja potência de 2 mais próxima de log_2(X[0]). O hierarchical exige que n seja dessa maneira.
    w = 2*n - 1                                      # número de parâmetros do circuito (weights)
    X = np.c_[X, np.zeros((len(X), 2**n-len(X[0])))] # o número de qubits necessários para codificar os dados (log_2(N)) precisa ser uma potencia de 2. # pylint: disable=no-member

    return n, N, w, X

def circuit(weights, state_vector=None, n=4):
    encode = Encoding(state_vector, 'dc_amplitude_encoding', entangle=True)
    q = encode.output_qubits # o comprimento de q deve ser igual a n.

    return circuit_hierarchical_classifier(n , q, weights)
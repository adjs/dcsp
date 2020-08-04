import pennylane as qml
from pennylane import numpy as np
from cin.pennylane.qml.hierarchical_classifier import circuit as circuit_hierarchical_classifier

def config(X):
    n = int(np.ceil(np.log2(len(X[0]))))             # pylint: disable=no-member
    n = 2**int(np.ceil(np.log2(n)))                  # n tem que ser potência de 2. # pylint: disable=no-member
    N = n                                            # número total de qubits no circuito.
    w = 2*n - 1                                      # número de parâmetros do circuito (weights)
    X = np.c_[X, np.zeros((len(X), 2**n-len(X[0])))] # o número de qubits necessários para codificar os dados (log_2(N)) precisa ser uma potencia de 2. # pylint: disable=no-member
    
    return n, N, w, X

def circuit(weights, state_vector=None, n=4):
    q=range(n)
    
    X = state_vector / np.linalg.norm(state_vector) # pylint: disable=no-member
    qml.templates.AmplitudeEmbedding(X, wires=q)
    
    return circuit_hierarchical_classifier(n, q, weights)
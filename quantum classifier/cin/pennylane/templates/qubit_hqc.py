import pennylane as qml
from pennylane import numpy as np
from cin.pennylane.qml.hierarchical_classifier import circuit as circuit_hierarchical_classifier

def config(X):
    n = 2**int(np.ceil(np.log2(len(X[0]))))       # len(X[0]) # número de qubits necessário para armazenar o dado codificado. # pylint: disable=no-member
    N = n                                         # número total de qubits no circuito.
    w = 2*n - 1                                   # número de parâmetros do circuito (weights)
    X = np.c_[X, np.zeros((len(X), n-len(X[0])))] # pylint: disable=no-member

    return n, N, w, X

def circuit(weights, state_vector=None, n=4):
    q=range(n)
    
    X = state_vector * np.pi # re-scale the data vectors element-wise to lie in [0, pi]. # pylint: disable=no-member
    qml.templates.AngleEmbedding(X, wires=q, rotation="Y")
    
    return circuit_hierarchical_classifier(n, q, weights)
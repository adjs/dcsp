import numpy as np
from sklearn import datasets

def load():
    dataset = datasets.load_breast_cancer() # load the breast cancer dataset.
    X = dataset.data                        # separate the data from the target attributes. # pylint: disable=no-member
    Y = dataset.target                      #                                               # pylint: disable=no-member
    
    X = np.array([ e[0] for e in list(zip(X, Y)) if e[1] in [0, 1]]) # select intended classes.

    Y = np.array([ 2*e-1 for e in Y if e in [0, 1]])                     #

    normalization = np.sqrt(np.sum(X ** 2, -1)) # pylint: disable=no-member
    X_norm = (X.T / normalization).T
    
    return X_norm, Y


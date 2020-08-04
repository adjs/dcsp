import numpy as np
from sklearn import datasets

def load(classes):
    iris = datasets.load_iris() # load the iris dataset.
    X = iris.data               # separate the data from the target attributes. # pylint: disable=no-member
    Y = iris.target             #                                               # pylint: disable=no-member
    
    X = np.array([ e[0] for e in list(zip(X, Y)) if e[1] in classes])  # select intended classes.
    
    min_class = min(classes)                                             #
    c = max(classes) - min_class                                         #
    Y = np.array([ 2*((e-min_class)//c)-1 for e in Y if e in classes]) # +1 ou -1.

    #normalization = np.sqrt(np.sum(X ** 2, -1)) # pylint: disable=no-member
    #X_norm = (X.T / normalization).T
    X_norm = X / X.max(axis=0) # reescala as colunas para valores entre 0 e 1. Há apenas valores positivos no Iris.
    
    return X_norm, Y

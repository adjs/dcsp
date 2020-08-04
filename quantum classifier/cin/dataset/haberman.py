import numpy as np
import urllib

def load():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    raw_data = urllib.request.urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",") # pylint: disable=no-member

    X = np.array(dataset[:,0:-1])
    Y = np.array([2*(int(e)-1)-1 for e in dataset[:,-1]]) # +1 ou -1;

    #normalization = np.sqrt(np.sum(X ** 2, -1)) # pylint: disable=no-member
    #X_norm = (X.T / normalization).T
    X_norm = X / X.max(axis=0) # reescala as colunas para valores entre 0 e 1.
    
    return X_norm, Y

import numpy as np
import urllib

def load():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    #url = "https://datahub.io/machine-learning/banknote-authentication/r/banknote-authentication.csv"
    raw_data = urllib.request.urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",") # , skiprows=1) # pylint: disable=no-member

    X = np.array(dataset[:,0:-1])
    Y = np.array([2*int(e)-1 for e in dataset[:,-1]]) # +1 ou -1.

    #normalization = np.sqrt(np.sum(X ** 2, -1)) # pylint: disable=no-member
    #X_norm = (X.T / normalization).T
    X_norm = X - X.min(axis=0) # banknote possui valores negativos, então precisa deslocar os valores das colunas de acordo.
    X_norm = X_norm / X_norm.max(axis=0)

    return X_norm, Y

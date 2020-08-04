import numpy as np

def split_data(X, Y, validation_size=0.1, test_size=0.1):
    np.random.seed(0) # pylint: disable=no-member
    num_data = len(Y)
    num_test = int(test_size * num_data)
    num_val = int(validation_size * num_data)
    
    index = np.random.permutation(range(num_data)) # pylint: disable=no-member
    
    X_test = X[index[:num_test]]
    Y_test = Y[index[:num_test]]
    X_val = X[index[num_test:num_val+num_test]]
    Y_val = Y[index[num_test:num_val+num_test]]
    X_train = X[index[num_val+num_test:]]
    Y_train = Y[index[num_val+num_test:]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

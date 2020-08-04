import pennylane as qml
from pennylane import numpy as np
import time

def variational_classifier(node, var, state_vector=None, n=4):
    weights = var[0:-1]
    bias = var[-1]
    return node(weights, state_vector=state_vector, n=n) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + ((l - p)/2) ** 2

    loss = loss / len(labels)
    return loss

def cost(node, n, var, state_vectors, labels):
    predictions = [variational_classifier(node, var, state_vector=state_vector, n=n) for state_vector in state_vectors]
    return square_loss(labels, predictions)

def accuracy(labels, predictions):
    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc

def numpy_interface(init_node, init_var, n, lr, steps, batch_size, X_train, X_val, X_test, Y_train, Y_val, Y_test, show=False):
    if (show):
        print('\nNumPy interface:')

    X = [*X_train, *X_val]
    Y = [*Y_train, *Y_val]

    best = [0, 1.0, 0.0, 0.0, 0.0, []]

    node = init_node
    var = init_var
    opt = qml.optimize.AdamOptimizer(stepsize=lr)

    time1 = time.time()
    for it in range(steps):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,)) # pylint: disable=no-member
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(node, n, v, X_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [2*(variational_classifier(node, var, state_vector=f, n=n)>0.0)-1 for f in X_train] 
        predictions_val = [2*(variational_classifier(node, var, state_vector=f, n=n)>0.0)-1 for f in X_val]     
        
        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        # Compute cost on complete dataset
        cost_set = cost(node, n, var, X, Y)

        #if (cost_set < best[1]):
        if (acc_val > best[3] or (acc_val == best[3] and cost_set < best[1])):
          best[0] = it + 1
          best[1] = cost_set
          best[2] = acc_train
          best[3] = acc_val
          best[4] = 0.0
          best[5] = var

        if (show):
            print(
                "Iter:{:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                "".format(it + 1, cost_set, acc_train, acc_val)
            )
    
    # Compute predictions on test set
    predictions_test = [2*(variational_classifier(node, best[5], state_vector=f, n=n)>0.0)-1 for f in X_test]
    # Compute accuracy on test set
    acc_test = accuracy(Y_test, predictions_test)
    best[4] = acc_test

    time2 = time.time()

    if (show):
        print("Optimized rotation angles: {}".format(best[5][:-1]))
        print("Optimized bias: {}".format(best[5][-1]))
        print("Optimized test accuracy: {:0.7f}".format(acc_test))
        print(f'Run time={((time2-time1)*1000.0):.3f}')

    return best

def pytorch_interface(init_node, init_var, n, lr, steps, batch_size, X, X_train, X_val, Y, Y_train, Y_val):
    print('\n\nPyTorch interface:')

    import torch
    from torch.autograd import Variable

    best = [0, 1.0, 0.0, 0.0, 0.0, []]

    node = init_node.to_torch()
    var = Variable(torch.tensor(init_var), requires_grad=True) # noqa pylint: disable=not-callable
    opt = torch.optim.Adam([var], lr = lr)

    time1 = time.time()
    def closure(n, var, X_train_batch, Y_train_batch):
        opt.zero_grad()
        loss = cost(node, n, var, X_train_batch, Y_train_batch)
        loss.backward() # noqa pylint: disable=no-member

        return loss
    for it in range(steps):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,)) # pylint: disable=no-member
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]

        opt.step(lambda: closure(n, var, X_train_batch, Y_train_batch))

        # Compute predictions on train and validation set
        predictions_train = [2*(variational_classifier(node, var, state_vector=f, n=n) > 0.0)-1 for f in X_train] 
        predictions_val = [2*(variational_classifier(node, var, state_vector=f, n=n) > 0.0)-1 for f in X_val]     
        
        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        # Compute cost on complete dataset
        cost_set = cost(node, n, var, X, Y)

        if (cost_set < best[1]):
          best[0] = it + 1
          best[1] = cost_set
          best[2] = acc_train
          best[3] = acc_val
          best[4] = var[-1]
          best[5] = var[:-1]

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format(it + 1, cost_set, acc_train, acc_val)
        )
    time2 = time.time()

    print("Optimized rotation angles: {}".format(var[:-1]))
    print("Optimized bias: {}".format(var[-1]))
    print(f'Run time={((time2-time1)*1000.0):.3f}')

    return best

def tensorflow_interface(init_node, init_var, n, lr, steps, batch_size, X, X_train, X_val, Y, Y_train, Y_val):
    print('\n\nTensorFlow interface:')

    import tensorflow as tf

    best = [0, 1.0, 0.0, 0.0, 0.0, []]

    node = init_node.to_tf()
    var = tf.Variable(init_var, dtype=tf.float64)
    opt = tf.optimizers.Adam(learning_rate=lr)

    time1 = time.time()
    for it in range(steps):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,)) # pylint: disable=no-member
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]

        with tf.GradientTape() as tape:
            loss = cost(node, n, var, X_train_batch, Y_train_batch)
            grads = tape.gradient(loss, [var])

        opt.apply_gradients(zip(grads, [var]))

        # Compute predictions on train and validation set
        predictions_train = [np.sign(variational_classifier(node, var, state_vector=f, n=n)) for f in X_train] # pylint: disable=no-member
        predictions_val = [np.sign(variational_classifier(node, var, state_vector=f, n=n)) for f in X_val]     # pylint: disable=no-member
        
        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        # Compute cost on complete dataset
        cost_set = cost(node, n, var, X, Y)

        if (cost_set < best[1]):
          best[0] = it + 1
          best[1] = cost_set
          best[2] = acc_train
          best[3] = acc_val
          best[4] = var[-1]
          best[5] = var[:-1]
          
        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format(it + 1, cost_set, acc_train, acc_val)
        )
    time2 = time.time()

    print("Optimized rotation angles: {}".format(var[:-1]))
    print("Optimized bias: {}".format(var[-1]))
    print(f'Run time={((time2-time1)*1000.0):.3f}')

    return best

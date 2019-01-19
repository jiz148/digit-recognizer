"""

"""
import numpy as np


def change_to_multi_class(y, num_of_labels):
    """

    :param y:
    :param num_of_labels:
    :return:
    """

    m = y.shape[1]
    multi_class_y = np.zeros([num_of_labels, m])

    for i in range(m):
        label = y[0, i]
        multi_class_y[int(label), i] = 1

    return multi_class_y


def compute_cost(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = sum(sum((1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL ).T))))

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def compute_cost_with_l2_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    num_of_parameters = len(parameters) // 2
    w_square_sum = 0
    for i in range(num_of_parameters):
        w_square_sum += np.sum(np.square(parameters['W'+str(i+1)]))

    cross_entropy_cost = compute_cost(A3, Y)

    l2_regularization_cost = (lambd / (2 * m)) * w_square_sum

    cost = cross_entropy_cost + l2_regularization_cost

    return cost


def initialize_parameters_deep_he(layer_dims):

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def initialize_parameters_deep(layer_dims):

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters



def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                                           activation="leaky_relu") #was relu
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid") # changed
    caches.append(cache)

    assert (AL.shape == (10, X.shape[1])) # shape[0] should be same with shape[0] of output layer

    return AL, caches


def L_model_backward_with_l2(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_with_l2(dAL, current_cache,
                                                                                                              lambd, activation="sigmoid") # changed

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_l2(grads["dA" + str(l + 1)], current_cache,
                                                                                     lambd, activation="leaky_relu") # was relu
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def linear_activation_backward_with_l2(dA, cache, lambd, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_l2(dZ, linear_cache, lambd)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_l2(dZ, linear_cache, lambd)

    elif activation == "leaky_relu":
        dZ = leaky_relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_l2(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "leaky_relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leaky_relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def linear_backward_with_l2(dZ, cache, lambd):

    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T) + (lambd/m)*W
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    #dA_prev = dropouts_backward(dA_prev, D, keep_prob)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_forward(A, W, b):

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def one_vs_all_prediction(prob_matrix):
    """
    Compare every probability, get the maximum and output the index.

    :param prob_matrix: probability matrix
    :return: prediction generated from probability matrix
    """
    m = prob_matrix.shape[1]

    prediction = np.argmax(prob_matrix, axis=0)
    prediction = np.array([prediction])  # keep dimensions

    assert (prediction.shape == (1, m))

    return prediction


def relu(Z):

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def leaky_relu(Z):

    A = np.maximum(0.01*Z, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def leaky_relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z < 0, you should set dz to 0.01  as well.
    #temp = np.ones(Z.shape)
    #temp[Z <= 0] = 0.01
    #dZ = dZ*temp

    #Z[Z > 0] = 1
    #Z[Z != 1] = 0.01
    #dZ = dZ*Z

    temp = np.ones_like(Z)
    temp[Z < 0] = 0.01
    dZ = dZ*temp

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


"""
def dropouts_forward(A,  activation_cache, keep_prob):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = A * D
    A = A / keep_prob
    cache = (activation_cache, D)
    return A, cache


def dropouts_backward(dA, D, keep_prob):
    dA = dA*D
    dA = dA/keep_prob
    return dA
"""

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters





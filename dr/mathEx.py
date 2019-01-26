"""
mathEX.py
@author: Jinchi Zhang
@email: jizjiz148148@gmail.com

Math-related functions, extensions.
"""
import numpy as np


def change_to_multi_class(label_set, num_of_labels):
    """
    change the input prediction y to array-wise multi_class classifiers.

    :param label_set:input prediction y, numpy arrays
    :param num_of_labels: number of class we want to classify, ints
    :return: array-wise multi_class classifiers
    """

    number_of_samples = label_set.shape[1]
    multi_class_y = np.zeros([num_of_labels, number_of_samples])

    for i in range(number_of_samples):
        label = label_set[0, i]
        multi_class_y[int(label), i] = 1

    return multi_class_y


def compute_cost(training_result, label_set):
    """
    compute costs between output results and actual results y. NEEDS TO BE MODIFIED.

    :param training_result: output results, numpy arrays
    :param label_set: actual result, numpy arrays
    :return: cost, floats
    """

    num_of_samples = label_set.shape[1]

    # Compute loss from aL and y.
    cost = sum(sum((1. / num_of_samples) * (-np.dot(label_set, np.log(training_result).T) - np.dot(1 - label_set, np.log(1 - training_result).T))))

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def compute_cost_with_l2_regularization(training_result, label_set, parameters, lambd):
    """
    compute costs with L2 regularization, uses the original cost function.

    :param training_result: results AL, numpy arrays
    :param label_set: actual results y, numpy arrays
    :param parameters: parameters got from forward propagation, dictionaries
    :param lambd: lambda for regularization, floats
    :return: cost, floats
    """
    num_of_samples = label_set.shape[1]
    num_of_parameters = len(parameters) // 2
    w_square_sum = 0

    # adding up Ws
    for i in range(num_of_parameters):
        w_square_sum += np.sum(np.square(parameters['W'+str(i+1)]))

    # compute regular costs
    cross_entropy_cost = compute_cost(training_result, label_set)

    # combine regular costs and regularization term
    l2_regularization_cost = (lambd / (2 * num_of_samples)) * w_square_sum

    cost = cross_entropy_cost + l2_regularization_cost

    return cost


def initialize_parameters_deep_he(layer_dims):
    """
    initialization for deep learning with HE random algorithm to prevent fading & exploding gradients.

    :param layer_dims: dimensions of layers, lists
    :return: initialized parameters
    """

    np.random.seed(1)
    parameters = {}
    num_of_layers = len(layer_dims)  # number of layers in the network

    for l in range(1, num_of_layers):

        # initialized W with random and HE term
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


"""
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
"""


def l_model_forward(input_set, parameters):
    """
    Forward propagation of deep learning.

    :param input_set: input x, numpy arrays
    :param parameters:
    :return: output aL and caches for following calculations, numpy arrays and indexes
    """

    caches = []
    last_set = input_set
    num_of_layers = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, num_of_layers):
        a_prev = last_set

        # use relu or leaky relu in hiden layers
        last_set, cache = linear_activation_forward(a_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="leaky_relu") #was relu
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    # output layer with sigmoid activation
    training_result, cache = linear_activation_forward(last_set, parameters['W' + str(num_of_layers)], parameters['b' + str(num_of_layers)], activation="sigmoid")

    caches.append(cache)

    assert (training_result.shape == (10, input_set.shape[1])) # shape[0] should be same with shape[0] of output layer

    return training_result, caches


def L_model_backward_with_l2(training_result, label_set, caches, lambd):
    """
    Backward propagation for deep learning with L2 regularization.

    :param training_result: output AL, numpy arrays
    :param label_set: actual answers y, numpy arrays
    :param caches: caches from forward propagation, dictionaries
    :param lambd: regularization parameter lambda, floats
    :return: gradients for gradient decent, dictionaries
    """
    grads = {}
    num_of_layers = len(caches)  # the number of layers
    num_of_samples = training_result.shape[1]
    label_set = label_set.reshape(training_result.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    d_training_result = - (np.divide(label_set, training_result) - np.divide(1 - label_set, 1 - training_result))

    # Lth layer Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[num_of_layers - 1]
    grads["dA" + str(num_of_layers - 1)], grads["dW" + str(num_of_layers)], grads["db" + str(num_of_layers)] = linear_activation_backward_with_l2(d_training_result, current_cache, lambd, activation="sigmoid")

    for l in reversed(range(num_of_layers - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]

        # use relu or leaky relu for hiden layers
        da_prev_temp, dw_temp, db_temp = linear_activation_backward_with_l2(grads["dA" + str(l + 1)], current_cache,
                                                                            lambd, activation="leaky_relu")
        grads["dA" + str(l)] = da_prev_temp
        grads["dW" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def linear_activation_backward_with_l2(d_current_set, cache, lambd, activation):
    """
    activation step for backward propagation with multiple choices of activation function.

    :param d_current_set: dA from last step of backward propagation, numpy arrays
    :param cache: caches in deep learning, dictionaries
    :param lambd: regularization parameter lambda, floats
    :param activation: choice of activation, strings
    :return: last dA, dW, db, numpy arrays
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        d_z = relu_backward(d_current_set, activation_cache)
        d_a_prev, d_w, d_b = linear_backward_with_l2(d_z, linear_cache, lambd)

    elif activation == "sigmoid":
        d_z = sigmoid_backward(d_current_set, activation_cache)
        d_a_prev, d_w, d_b = linear_backward_with_l2(d_z, linear_cache, lambd)

    elif activation == "leaky_relu":
        d_z = leaky_relu_backward(d_current_set, activation_cache)
        d_a_prev, d_w, d_b = linear_backward_with_l2(d_z, linear_cache, lambd)

    return d_a_prev, d_w, d_b


def linear_activation_forward(a_prev, parameter_w, parameter_b, activation):
    """
    activation step for forward propagation with multiple choices of activation function.

    :param a_prev: previous A from last step of forward propagation, numpy arrays
    :param parameter_w: parameter W in current layer, numpy arrays
    :param parameter_b: parameter b in current layer, numpy arrays
    :param activation: choice of activation, strings
    :return: current A and cache for following calculation
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        current_z, linear_cache = linear_forward(a_prev, parameter_w, parameter_b)
        A, activation_cache = sigmoid(current_z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        current_z, linear_cache = linear_forward(a_prev, parameter_w, parameter_b)
        A, activation_cache = relu(current_z)

    elif activation == "leaky_relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        current_z, linear_cache = linear_forward(a_prev, parameter_w, parameter_b)
        A, activation_cache = leaky_relu(current_z)

    assert (A.shape == (parameter_w.shape[0], a_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def linear_backward_with_l2(d_z, cache, lambd):
    """
    linear step in backward propagation.

    :param d_z: current dZ, numpy arrays
    :param cache: caches from previous calculation, dictionaries
    :param lambd: regularization parameter lambda, floats
    :return: previous dA, current dW, db, numpy arrays
    """

    a_prev, w, b = cache
    m = a_prev.shape[1]
    d_w = 1. / m * np.dot(d_z, a_prev.T) + (lambd / m) * w
    db = 1. / m * np.sum(d_z, axis=1, keepdims=True)
    da_prev = np.dot(w.T, d_z)

    # dA_prev = dropouts_backward(dA_prev, D, keep_prob)

    assert (da_prev.shape == a_prev.shape)
    assert (d_w.shape == w.shape)
    assert (db.shape == b.shape)

    return da_prev, d_w, db


def linear_forward(current_set, parameter_w, parameter_b):
    """
    linear step for forward propagation
    :param current_set: current A, numpy arrays
    :param parameter_w: current parameter W, numpy arrays
    :param parameter_b: current parameter b, numpy arrays
    :return: current z, and caches for following calculations, numpy arrays and dictionaries
    """

    current_z = parameter_w.dot(current_set) + parameter_b

    assert (current_z.shape == (parameter_w.shape[0], current_set.shape[1]))
    cache = (current_set, parameter_w, parameter_b)

    return current_z, cache


def one_vs_all_prediction(prob_matrix):
    """
    Compare every probability, get the maximum and output the index.

    :param prob_matrix: probability matrix, numpy arrays
    :return: prediction generated from probability matrix, numpy arrays
    """
    num_of_samples = prob_matrix.shape[1]

    prediction = np.argmax(prob_matrix, axis=0)
    prediction = np.array([prediction])  # keep dimensions

    assert (prediction.shape == (1, num_of_samples))

    return prediction


def relu(current_z):
    """
    relu function

    :param current_z: input A, numpy arrays or numbers
    :return: output A, numpy arrays or numbers
    """

    current_set = np.maximum(0, current_z)

    assert (current_set.shape == current_z.shape)

    cache = current_z
    return current_set, cache


def relu_backward(d_current_set, cache):
    """
    compute gradient of relu function.

    :param d_current_set: input dA, numpy arrays or numbers
    :param cache: caches with Z, dictionaries
    :return: result dZ, numpy arrays or numbers
    """

    current_z = cache
    dz = np.array(d_current_set, copy=True)  # just converting dz to a correct object.

    # When z <= 0s you should set dz to 0 as well.
    dz[current_z <= 0] = 0

    assert (dz.shape == current_z.shape)

    return dz


def leaky_relu(current_z):
    """
    leaky relu function

    :param current_z: input Z, numpy arrays or numbers
    :return: result A and caches for following calculation
    """

    current_set = np.maximum(0.01 * current_z, current_z)

    assert (current_set.shape == current_z.shape)

    cache = current_z
    return current_set, cache


def leaky_relu_backward(d_current_set, cache):
    """
    compute gradients of leaky relu function.

    :param d_current_set: input dA, numpy arrays or numbers
    :param cache: cache with Z, dictionaries
    :return: result dZ, numpy arrays or numbers
    """

    current_z = cache
    d_z = np.array(d_current_set, copy=True)  # just converting dz to a correct object.

    # When z < 0, you should set dz to 0.01  as well.
    #temp = np.ones(Z.shape)
    #temp[Z <= 0] = 0.01
    #dZ = dZ*temp


    #Z[Z > 0] = 1
    #Z[Z != 1] = 0.01
    #dZ = dZ*Z

    temp = np.ones_like(current_z)
    temp[current_z < 0] = 0.01
    d_z = d_z*temp

    assert (d_z.shape == current_z.shape)

    return d_z


def sigmoid(current_z):
    """
    sigmoid function.

    :param current_z: input Z, numpy arrays or numbers
    :return: result A, caches for following calculations, numpy arrays or numbers, dictionaries
    """

    a = 1 / (1 + np.exp(-current_z))
    cache = current_z

    return a, cache


def sigmoid_backward(current_set, cache):
    """
    compute gradients of sigmoid function.

    :param current_set: input dA, numpy arrays or numbers
    :param cache: caches with Z, dictionaries
    :return: result dZ, numpy arrays or numbers
    """

    current_z = cache

    sigmoid = 1 / (1 + np.exp(-current_z))
    d_z = current_set * sigmoid * (1 - sigmoid)

    assert (d_z.shape == current_z.shape)

    return d_z


"""
unused dropout functions
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
    """
    update parameters with gradients.

    :param parameters: input parameters, dictionaries
    :param grads: gradients, dictionaries
    :param learning_rate: hyper-parameter alpha for deep learning, floats
    :return: updated parameters, dictionaries
    """

    num_of_layers = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(num_of_layers):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters





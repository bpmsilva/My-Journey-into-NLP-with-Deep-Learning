"""
Module containing backprogation utils functions for Word2Vec
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import numpy as np

def cross_entropy(softmax_output, Y):
    """Compute cross entropy loss

    Args:
        softmax_output (numpy.ndarray): output of a softmax layer (vocab_size, number_examples)
        Y (numpy.ndarray): ground-truth labels?

    Returns:
        numpy.ndarray: cross entropy loss
    """
    number_examples = softmax_output.shape[1]
    # TODO: check if the 1e-8 is adequate (I changed it from 1e-3 to 1e-8)
    return -(1 / number_examples) * np.sum(np.sum(Y * np.log(softmax_output + 1e-8),
                                            axis=0,
                                            keepdims=True),
                                            axis=1)

def softmax_backward(Y, softmax_output):
    """Compute the backpropagation of the softmax layer

    Args:
        Y (numpy.ndarray): labels of the training data (shape: (vocab_size, num_examples))
        softmax_output (numpy.ndarray): output of the softmax (shape: (vocab_size, num_examples))

    Returns:
        numpy.ndarray: backpropagation pass of the softmax layer
    """
    return softmax_output - Y

def dense_backward(dL_dZ, caches):
    """Compute the dense layer backpropagation

    Args:
        dL_dZ (numpy.ndarray): backpropgation from softmax layer
        caches (dict): dictionary containing 'W' (numpy.ndarray) and 'word_vec' (numpy.ndarray)

    Returns:
        tuple: backpropagation of W and the word_vec
    """
    W = caches['W']
    word_vec = caches['word_vec']
    num_examples = word_vec.shape[1]

    dL_dW = (1 / num_examples) * np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)

    assert W.shape == dL_dW.shape, 'Derivate of W w.r.t. loss does not match with W dimensions'
    assert word_vec.shape == dL_dword_vec.shape, \
        'Derivate of word_vec w.r.t. loss does not match with word_vec dimensions'

    return dL_dW, dL_dword_vec

def backward_propagation(Y, softmax_output, caches):
    """Compute backpropagation of the entire neural network

    Args:
        Y (numpy.ndarray): labels of the training data (shape: (vocab_size, num_examples))
        softmax_output (numpy.ndarray): output of a softmax layer (vocab_size, number_examples)
        caches (dict): dictionary containing 'W' (numpy.ndarray) and 'word_vec' (numpy.ndarray)

    Returns:
        dict: [description]
    """
    dL_dZ = softmax_backward(Y, softmax_output)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)

    return {
        'dL_dZ': dL_dZ,
        'dL_dW': dL_dW,
        'dL_dword_vec': dL_dword_vec
    }

def update_parameters(parameters, caches, gradients, learning_rate):
    """Update the parameters of the neural network

    Args:
        parameters (dict): dictionary containing the 'W' and 'WRD_EMD' matrices
        caches (dict): dictionary containing the 'inds' list
        gradients (dict): dictionary containing 'dL_dword_vec' and 'dL_dW'
        learning_rate (float): training learning rate
    """
    word_embedding = parameters['WRD_EMB']
    indices = caches['inds']
    dL_dword_vec = gradients['dL_dword_vec']

    # update
    word_embedding[indices.flatten(), :] -= dL_dword_vec.T * learning_rate
    parameters['W'] -= gradients['dL_dW'] * learning_rate

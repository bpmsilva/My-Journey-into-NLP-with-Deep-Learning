"""
Module containing forward utils functions for Word2Vec
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import numpy as np

def indices_to_word_vecs(indices, wrd_emb):
    """Convert a list of word indices in their corresponding feature vectors

    Args:
        indices (list of int): a list of integers, each corresponding to a token/word
        wrd_emb (numpy.ndarray): word embedding matrix of shape (vocab size, embedding size)

    Returns:
        numpy.ndarray: matrix of embedding for each word index
    """
    num_examples = indices.shape[1]
    word_vec = wrd_emb[indices.flatten(), :].T

    assert word_vec.shape == (wrd_emb.shape[1], num_examples), \
           'Shape of word vector does not match (num features, training examples)'

    return word_vec

def linear_dense(word_vec, W):
    """Multiply the output of the embedding layer with the dense layer

    Args:
        word_vec (numpy.ndarray): matrix with shape (emb_size, num_examples)
        W (numpy.ndarray): dense layer weights

    Returns:
        tuple of numpy.ndarray: matrix W and the product of W and word_vec
    """
    Z = np.dot(W, word_vec)

    return W, Z

def softmax(Z):
    """Apply softmax to a layer output

    Args:
        Z (numpy.ndarray): a layer output

    Returns:
        [numpy.ndarray]: softmax output
    """
    # TODO: check if the 1e-8 in the numerator is adequate (I changed it from 1e-3 to 1e-8)
    return np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 1e-8)

def forward_propagation(indices, parameters):
    """Compute forward pass of the Word2Vec Algorithm

    Args:
        indices (list of int): a list of integers, each corresponding to a token/word
        parameters (dict): a dictionary containing the 'WRD_EMB' and 'W' keys for the
                           word_embedding matrix and dense layer weights

    Returns:
        [type]: [description]
    """
    W = parameters['W']
    word_embedding = parameters['WRD_EMB']

    word_vec = indices_to_word_vecs(indices, word_embedding)
    W, Z = linear_dense(word_vec, W)
    softmax_output = softmax(Z)

    caches = {
        'inds': indices,
        'word_vec': word_vec,
        'W': W,
        'Z': Z
    }

    return softmax_output, caches

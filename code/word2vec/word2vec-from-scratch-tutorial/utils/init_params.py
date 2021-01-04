"""
Module containing utils functions to initialize the Word2Vec weights
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import numpy as np

def initialize_wrd_emb(vocab_size, emb_size):
    """Initialize the word embedding matrix

    Args:
        vocab_size (int): your corpus vocabulary size (number of tokens/words)
        emb_size (int): number of features for each token/word

    Returns:
        numpy.ndarray: word embedding matrix of shape (vocab_size, emb_size)
    """
    # TODO: replace the initialization constant 0.01 in a more systematic approach
    return np.random.randn(vocab_size, emb_size) * 0.01

def initialize_dense(input_size, output_size):
    """Initialize dense layer weights

    Args:
        input_size (int): input size of the dense layer
        output_size (int): output size of the dense layer

    Returns:
        numpy.ndarray: dense layer weights
    """
    # TODO: replace the initialization constant 0.01 in a more systematic approach
    return np.random.randn(output_size, input_size) * 0.01

def initialize_parameters(vocab_size, emb_size):
    """Initialize all necessary parameters for the Word2Vec training

    Args:
        vocab_size (int): your corpus vocabulary size (number of tokens/words)
        emb_size (int): number of features for each token/word

    Returns:
        dict: map of str to numpy.ndarray for the word embedding matrix,
              and for the dense layer weights
    """
    word_embedding_matrix = initialize_wrd_emb(vocab_size, emb_size)
    dense_layer_weights = initialize_dense(emb_size, vocab_size)

    return {
        'WRD_EMB': word_embedding_matrix,
        'W': dense_layer_weights
    }

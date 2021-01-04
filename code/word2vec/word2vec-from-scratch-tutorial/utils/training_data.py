"""
Module containing utils functions to prepare the Word2Vec training data
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import re
import numpy as np

def tokenize(text):
    """Create tokens with at least 1 alphabet for a given text

    Args:
        text (str): a string where the tokens will be generated from

    Returns:
        list: a list containing all tokens matches
    """
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    """Create two dictionaries that map words (tokens) to indices

    Args:
        tokens (list of str): [description]

    Returns:
        tuple of dict: two dictionaries to map word (srt) to index (int),
                       and index (int) to word (str)
    """
    word2id, id2word = dict(), dict()

    # TODO: Check if this is correct: I sorted the set of words alphabetically
    #       before mapping
    for i, token in enumerate(sorted(set(tokens))):
        word2id[token] = i
        id2word[i] = token

    return word2id, id2word

def generate_training_data(tokens, word2id, window_size):
    """Generate input and output pairs to train the Word2Vec algorithm

    Args:
        tokens (list of str): document with the tokens (words) to be used
        word2id (dict): a map of a token (str) to an number (int)
        window_size (int): window of distance for pair generation

    Returns:
        tuple of list: two lists for the training pairs
    """
    X, Y = [], []
    N = len(tokens)

    for i in range(N):
        # In the original code neighboor_indices variable was nbr_inds
        neighboor_indices = list(range(max(0, i - window_size), i)) + \
                            list(range(i + 1, min(N, i + window_size + 1)))

        for j in neighboor_indices:
            X.append(word2id[tokens[i]])
            Y.append(word2id[tokens[j]])

    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y

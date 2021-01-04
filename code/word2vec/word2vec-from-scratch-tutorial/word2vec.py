"""
Main module to train the Word2Vec algorithm
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import numpy as np

from utils.training_data import tokenize, mapping, generate_training_data
from utils.init_params import initialize_parameters
from utils.forward_propagation import forward_propagation
from utils.training import skipgram_model_training

if __name__ == '__main__':
    # This should be a larger corpus
    doc = "After the deduction of the costs of investing," \
        "beating the stock market is a loser's game."

    # pre-process the training data
    tokens = tokenize(doc)
    word2id, id2word = mapping(tokens)
    X, Y = generate_training_data(tokens, word2id, window_size=3)

    # initialize the paramaters
    vocab_size = len(id2word)
    parameters = initialize_parameters(vocab_size, 100)

    # get the one hot representation
    num_examples = Y.shape[1]
    Y_one_hot = np.zeros((vocab_size, num_examples))
    Y_one_hot[Y.flatten(), np.arange(num_examples)] = 1

    # train the model
    parameters = skipgram_model_training(
        X,
        Y_one_hot,
        vocab_size,
        50,
        0.05,
        5000,
        batch_size=128,
        parameters=None,
        print_cost=True,
        plot_cost=False
    )

    # evaluation
    X_eval = np.arange(vocab_size)
    X_eval = np.expand_dims(X_eval, axis=0)
    softmax_eval, _ = forward_propagation(X_eval, parameters)
    top_sorted_indices = np.argsort(softmax_eval, axis=0)[-4:,:]

    for input_indice in range(vocab_size):
        input_word = id2word[input_indice]
        output_words = \
            [id2word[output_indice] for output_indice in top_sorted_indices[::-1, input_indice]]
        print("{}'s neighboor word: {} - {} - {} - {}".format(input_word, *output_words))

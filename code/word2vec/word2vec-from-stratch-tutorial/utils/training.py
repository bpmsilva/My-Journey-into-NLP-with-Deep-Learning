"""
Module containing utils functions to train the Word2Vec model
Author: Ivan Chen (https://github.com/ujhuyz0110)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import numpy as np
import matplotlib.pyplot as plt

from .init_params import initialize_parameters
from .forward_propagation import forward_propagation
from .backpropagation import backward_propagation, cross_entropy, update_parameters

def skipgram_model_training(
    X,
    Y,
    vocab_size,
    emb_size,
    learning_rate,
    epochs,
    batch_size=256,
    parameters=None,
    print_cost=True,
    plot_cost=True
):
    """Train Word2Vec model

    Args:
        X (list): list of the word indices
        Y (numpy.ndarray): one-hot matrix (vocab_size, num_examples) of the training samples
        vocab_size (int): total number of tokens/words
        emb_size (int): size of the embedding representation
        learning_rate (float): gradient descent learning rate
        epochs (int): number of epochs
        batch_size (int, optional): size of the batch. Defaults to 256.
        parameters (dict, optional): dictionary containing the 'W' and 'WRD_EMB' matrices.
                                     Defaults to None.
        print_cost (bool, optional): Set this to True if you want to print the cost periodically.
                                     Defaults to True.
        plot_cost (bool, optional): Set this to True if you want to plot the costs at the end of
                                    training. Defaults to True.
    Returns:
        dict: dictionary containing the parameters 'W' and 'WRD_EMB'
    """

    costs = []
    num_examples = X.shape[1]
    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)

    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0, num_examples, batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = X[:, i:i+batch_size]
            Y_batch = Y[:, i:i+batch_size]

            softmax_output, caches = forward_propagation(X_batch, parameters)
            gradients = backward_propagation(Y_batch, softmax_output, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            cost = cross_entropy(softmax_output, Y_batch)
            epoch_cost += np.squeeze(cost)

        costs.append(epoch_cost)

        if print_cost and epoch % 10 == 0:
            print(f'Cost after epoch {epoch}: {epoch_cost}')
        if epoch % 50 == 0:
            learning_rate *= 0.98

    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
        plt.show()

    return parameters

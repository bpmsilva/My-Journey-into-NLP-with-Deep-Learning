"""
Module to generate t-sne scatter plot
Author: Pierre Megret (https://www.kaggle.com/pierremegret)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def tsne_scatterplot(model, word, list_names):
    """Generate and plot t-sne scatter plot of a word2vec model

    Args:
        model (gensim.models.word2vec.Word2Vec): model of the words/tokens
        word (str): main word from which the 10 most similar words will be computed
        list_names (list of str): list of strings to compare to the main word
    """

    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for word_score in close_words:
        word_vector = model.wv.__getitem__([word_score[0]])
        word_labels.append(word_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, word_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for word in list_names:
        word_vector = model.wv.__getitem__([word])
        word_labels.append(word)
        color_list.append('green')
        arrays = np.append(arrays, word_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduced_dims = PCA(n_components=19).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduced_dims)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    p1 = sns.regplot(
        data=df,
        x='x',
        y='y',
        fit_reg=False,
        marker='o',
        scatter_kws={
            's': 40,
            'facecolors': df['color']
        }
    )

    # add annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(
            df['x'][line],
            df['y'][line],
            ' ' + df['words'][line].title(),
            horizontalalignment='left',
            verticalalignment='bottom',
            size='medium',
            color=df['color'][line],
            weight='normal'
        ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.xlim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title(f't-SNE visualization for {word.title()}')

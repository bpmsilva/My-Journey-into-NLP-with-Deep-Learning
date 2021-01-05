"""
Main module for the character level RNN
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import os
import time
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn

from models.rnn import RNN
from utils.preprocessing import num_letters, read_lines, category_from_output
from utils.train import time_since, random_training_example, train_step, single_prediction

def parse_args():
    """Read the user input arguments or apply the default ones

    Returns:
        argparse.Namespace: arguments
    """
    parser = argparse.ArgumentParser(description='Training and evaluation RNN script')

    parser.add_argument('--learning-rate', help='training learning rate', default=5e-3, type=float)
    parser.add_argument(
        '--num-iters',
        help='number of training iterations',
        default=100000,
        type=int
    )
    parser.add_argument(
        '--size-hidden',
        help='number of neurons of the RNN hidden layer',
        default=128,
        type=int
    )
    parser.add_argument(
        '--num-confusion',
        help='number of predictions for the confusion matrix',
        default=10000,
        type=int
    )
    parser.add_argument('--print-every', help='training log period', default=5000, type=int)
    parser.add_argument('--plot-every', help='plot loss period', default=1000, type=int)

    return parser.parse_args()

def main():
    """
    Main function (training and simple prediction)
    """
    # get arguments
    args = parse_args()

    # Iterate through all data files and build a category list and a line dictionary
    all_categories, category_lines = [], {}
    for filepath in glob.glob('./data/names/*.txt'):
        filename = os.path.basename(filepath)
        category = os.path.splitext(filename)[0]
        all_categories.append(category)
        lines = read_lines(filepath)
        category_lines[category] = lines

    # small log
    num_categories = len(all_categories)
    print(f'There are {num_categories} categories')
    for i in range(10):
        category, line, category_tensor, line_tensor = \
                random_training_example(all_categories, category_lines)
        print('category =', category, '/ line =', line)

    # instantiate the criterion and the RNN model
    criterion = nn.NLLLoss()
    rnn = RNN(num_letters(), args.size_hidden, num_categories)

    # train the RNN, keeping track of the losses and the elapsed time
    all_losses = []
    current_loss = 0
    start_time = time.time()
    for step in range(1, args.num_iters + 1):
        category, line, category_tensor, line_tensor = \
                random_training_example(all_categories, category_lines)
        output, loss = train_step(rnn, category_tensor, line_tensor, args.learning_rate, criterion)
        current_loss += loss

        if step % args.print_every == 0:
            pred, pred_idx = category_from_output(output, all_categories)
            correct = '✓' if pred == category else f'✗ ({category})'
            print(f'{step} {step / args.num_iters * 100:.0f}% ' \
                    f'{time_since(start_time)} {loss:.4f} {line} / {pred} {correct}')

        if step % args.plot_every == 0:
            all_losses.append(current_loss / args.plot_every)
            current_loss = 0

    # evaluate (this does not follow the good practice of spliting the dataset into train and test)
    confusion_matrix = torch.zeros(num_categories, num_categories)
    for i in range(args.num_confusion):
        category, line, category_tensor, line_tensor = \
                random_training_example(all_categories, category_lines)
        output = single_prediction(line_tensor, rnn)
        pred, pred_idx = category_from_output(output, all_categories)
        category_idx = all_categories.index(category)
        confusion_matrix[category_idx][pred_idx] += 1

    # normalize the confusion matrix by dividing every row by its sum
    for i in range(num_categories):
        confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

    # plot the loss
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    # set up the confusion matrix plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix.numpy())
    fig.colorbar(cax)

    # set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    print(type(parse_args()))
    plt.show()

if __name__ == '__main__':
    main()

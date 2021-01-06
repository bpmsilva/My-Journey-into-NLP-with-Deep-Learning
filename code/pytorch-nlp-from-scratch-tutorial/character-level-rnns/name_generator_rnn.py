"""
Main module for the character level name generator RNN
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import os
import time
import glob
import string
import argparse
import matplotlib.pyplot as plt

import torch.nn as nn

from models.name_generator_rnn import NameGeneratorRNN
from utils.preprocessing import read_lines
from utils.train import \
        time_since, train_step_many_to_many, random_training_example_and_target, samples

def parse_args():
    """Read the user input arguments or apply the default ones

    Returns:
        argparse.Namespace: arguments
    """
    parser = argparse.ArgumentParser('Training and samples of character level RNN name generator')

    parser.add_argument('--hidden-size', help='hidden state size', default=128, type=int)
    parser.add_argument(
        '--learning-rate',
        help='gradient descent learning rate',
        default=0.0005,
        type=float
    )
    parser.add_argument('--num-iters', help='number of iterations', default=100000, type=int)
    parser.add_argument('--print-every', help='training log period', default=5000, type=int)
    parser.add_argument('--plot-every', help='plot loss period', default=500, type=int)
    parser.add_argument('--max-len', help='generated name max len', default=20, type=int)

    return parser.parse_args()

def main():
    """
    Main function for the character level name generator RNN
    """
    # retrieve arguments
    args = parse_args()

    # letters used for the name generation
    all_letters = string.ascii_letters + " .,;'-"
    num_letters = len(all_letters) + 1 # +1 for the EOS marker

    # Iterate through all data files
    # and build a category list and a dictionary with each category lines
    all_categories, category_lines = [], {}
    for filepath in glob.glob('./data/names/*.txt'):
        filename = os.path.basename(filepath)
        category = os.path.splitext(filename)[0]
        all_categories.append(category)
        lines = read_lines(filepath, all_letters)
        category_lines[category] = lines

    # small log
    num_categories = len(all_categories)
    print(f'There are {num_categories} categories ({all_categories})')

    # instantiate the rnn model
    rnn = NameGeneratorRNN(num_letters, args.hidden_size, num_letters, num_categories)

    # training procedure
    total_loss = 0
    all_losses = []
    start_time = time.time()
    for step in range(1, args.num_iters + 1):
        category_tensor, line_tensor, target_tensor = \
                random_training_example_and_target(
                    all_categories,
                    category_lines,
                    all_letters,
                    num_letters
                )

        _, loss = \
            train_step_many_to_many(
                rnn, category_tensor,
                line_tensor,
                target_tensor,
                args.learning_rate,
                nn.NLLLoss()
            )

        total_loss += loss

        if step % args.print_every == 0:
            print(f'{time_since(start_time)} ({step} {step / args.num_iters * 100}%) {loss:.4f}')

        if step % args.plot_every == 0:
            all_losses.append(total_loss / args.plot_every)
            total_loss = 0

    # plot losses
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    # print some samples
    samples(rnn, 'Russian', all_categories, all_letters, 'RUS', num_letters, args.max_len)
    samples(rnn, 'German', all_categories, all_letters, 'GER', num_letters, args.max_len)
    samples(rnn, 'Spanish', all_categories, all_letters, 'SPA', num_letters, args.max_len)
    samples(rnn, 'Chinese', all_categories, all_letters, 'CHI', num_letters, args.max_len)

if __name__ == '__main__':
    main()

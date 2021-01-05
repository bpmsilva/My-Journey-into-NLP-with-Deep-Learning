"""
Module containing the training helper functions
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import time
import random

import torch

from .preprocessing import line2tensor

def time_since(since):
    """Compute elapsed time from a starting period

    Args:
        since (float): start timestamp

    Returns:
        str: elapsed time in minutes:seconds
    """
    now = time.time()
    seconds = now - since
    minutes = seconds // 60
    seconds -= minutes * 60
    return f'{int(minutes):02d}:{int(seconds):02d}'

def random_choice(examples):
    """Randomly choose an element from a list

    Args:
        examples (list): list of examples

    Returns:
        Any: random element of the given list
    """
    return examples[random.randint(0, len(examples) - 1)]

def random_training_example(all_categories, category_lines, all_letters):
    """Generate a random training example

    Args:
        all_categories (list of str): list of the category names
        category_lines (list of str): lines of each category
        all_letters (str): string containing all valid letters

    Returns:
        (str, str, torch.tensor, torch.tensor): category name, line and their corresponding tensors
    """
    category = random_choice(all_categories)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)

    line = random_choice(category_lines[category])
    line_tensor = line2tensor(line, all_letters)

    return category, line, category_tensor, line_tensor

def train_step(rnn, category_tensor, line_tensor, learning_rate, criterion):
    """Single step RNN training

    Args:
        rnn (RNN): RNN model
        category_tensor (torch.tensor): tensor representation of the category
        line_tensor (torch.tensor): tensor representation of the line
        learning_rate (float): gradient descent learning rate
        criterion (Any loss function): valid criterion, e.g. torch.nn.NLLLoss

    Returns:
        [type]: [description]
    """
    hidden = rnn.init_hidden()

    rnn.zero_grad()
    for i in range(line_tensor.size()[0]): # for each character in line
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # TODO: return to this later. I didn't understand it
    for param in rnn.parameters():
        param.data.add_(param.grad.data, alpha=-learning_rate)

    return output, loss.item()

def single_prediction(line_tensor, rnn):
    """Compute a single prediction

    Args:
        line_tensor ([tensor.Torch): tensor representation of the line
        rnn (RNN): RNN model

    Returns:
        torch.tensor: neural network output
    """
    hidden = rnn.init_hidden()

    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

    return output

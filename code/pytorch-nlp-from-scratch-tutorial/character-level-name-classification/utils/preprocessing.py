"""
Module with preprocessing and helper functions for the character level RNN
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import string
import unicodedata

import torch

# string of ascii letters
ALL_LETTERS = string.ascii_letters + " .,;'"

def num_letters():
    """Return ALL_LETTERS length

    Returns:
        int: ALL_LETTERS length
    """
    return len(ALL_LETTERS)

def unicode2ascii(s):
    """Convert a unicode string into plain ASCII (e.g. 'Ślusàrski' -> Slusarski),
       thanks to https://stackoverflow.com/a/518232/2809427

    Args:
        s (str): input string

    Returns:
        str: ascii version of the input string
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn' and char in ALL_LETTERS
    )

def read_lines(filepath):
    """Read a file and split its contents into ascii encoded lines

    Args:
        filepath (str): path to file

    Returns:
        list of str: list containing ascii encoded lines
    """
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')

    return [unicode2ascii(line) for line in lines]

def letter2idx(letter):
    """Return the letter index, e.g. "a" -> 0

    Args:
        letter (str): single-character string

    Returns:
        int: the letter index
    """
    return ALL_LETTERS.find(letter)

def letter2tensor(letter):
    """Get the one-hot representation of the given letter

    Args:
        letter (str): single-character string

    Returns:
        torch.tensor: one-hot vector representation of the given letter
    """
    tensor = torch.zeros(1, len(ALL_LETTERS))
    tensor[0][letter2idx(letter)] = 1

    return tensor

def line2tensor(line):
    """Convert a line into a one-hot vector of shape (len(line), 1, len(ALL_LETTERS))

    Args:
        line (str): input line

    Returns:
        torch.tensor: a one-hot vector of shape (len(line), 1, len(ALL_LETTERS))
    """
    tensor = torch.zeros(len(line), 1, len(ALL_LETTERS))
    for idx, letter in enumerate(line):
        tensor[idx][0][letter2idx(letter)] = 1

    return tensor

def category_from_output(output, all_categories):
    """Return category name and category index for a RNN output

    Args:
        output (torch.tensor): RNN output
        all_categories (list of str): list of all categories names

    Returns:
        tuple (str, name): category name and category index
    """
    _, top_idx = output.topk(1)
    cat_idx = top_idx[0].item()

    return all_categories[cat_idx], cat_idx

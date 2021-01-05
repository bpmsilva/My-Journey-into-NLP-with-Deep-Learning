"""
Module with preprocessing and helper functions for the character level RNN
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import unicodedata

import torch

def unicode2ascii(s, all_letters):
    """Convert a unicode string into plain ASCII (e.g. 'Ślusàrski' -> Slusarski),
       thanks to https://stackoverflow.com/a/518232/2809427

    Args:
        s (str): input string
        all_letters (str): string containing all valid letters

    Returns:
        str: ascii version of the input string
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn' and char in all_letters
    )

def read_lines(filepath, all_letters):
    """Read a file and split its contents into ascii encoded lines

    Args:
        filepath (str): path to file
        all_letters (str): string containing all valid letters

    Returns:
        list of str: list containing ascii encoded lines
    """
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')

    return [unicode2ascii(line, all_letters) for line in lines]

def letter2idx(letter, all_letters):
    """Return the letter index, e.g. "a" -> 0

    Args:
        letter (str): single-character string
        all_letters (str): string containing all valid letters

    Returns:
        int: the letter index
    """
    return all_letters.find(letter)

def letter2tensor(letter, all_letters):
    """Get the one-hot representation of the given letter

    Args:
        letter (str): single-character string
        all_letters (str): string containing all valid letters

    Returns:
        torch.tensor: one-hot vector representation of the given letter
    """
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letter2idx(letter, all_letters)] = 1

    return tensor

def line2tensor(line, all_letters):
    """Convert a line into a one-hot vector of shape (len(line), 1, len(all_letters))

    Args:
        line (str): input line
        all_letters (str): string containing all valid letters

    Returns:
        torch.tensor: a one-hot vector of shape (len(line), 1, len(all_letters))
    """
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for idx, letter in enumerate(line):
        tensor[idx][0][letter2idx(letter, all_letters)] = 1

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

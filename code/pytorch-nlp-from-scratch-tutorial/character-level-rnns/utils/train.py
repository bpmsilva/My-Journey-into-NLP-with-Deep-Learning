"""
Module containing the training helper functions
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import time
import random

import torch

from .preprocessing import line2tensor, category2tensor, target2tensor

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

def random_training_example(all_categories, category_lines, all_letters, num_letters=None):
    """Generate a random training example

    Args:
        all_categories (list of str): list of the category names
        category_lines (dict): lines of each category
        all_letters (str): string containing all valid letters

    Returns:
        (str, str, torch.tensor, torch.tensor):
        category name, line, category num tensor (non-one-hot), and line tensor
    """
    category = random_choice(all_categories)
    category_num_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)

    line = random_choice(category_lines[category])
    line_tensor = line2tensor(line, all_letters, num_letters)

    return category, line, category_num_tensor, line_tensor

def random_training_example_and_target(
    all_categories,
    category_lines,
    all_letters,
    num_letters=None
):
    """Generate a random training example and a target

    Args:
        all_categories (list of str): list of category names
        category_lines (dict): lines of each category
        all_letters (str): string containing all valid letters

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): category, line and ground truth tensors
    """
    category, line, _, line_tensor = \
            random_training_example(all_categories, category_lines, all_letters, num_letters)
    cat_one_hot_tensor = category2tensor(category, all_categories)
    target_tensor = target2tensor(line, all_letters, num_letters)

    return cat_one_hot_tensor, line_tensor, target_tensor

def train_step(rnn, category_tensor, line_tensor, learning_rate, criterion):
    """Single step RNN training

    Args:
        rnn (RNN): RNN model
        category_tensor (torch.tensor): tensor representation of the category
        line_tensor (torch.tensor): tensor representation of the line
        learning_rate (float): gradient descent learning rate
        criterion (Any loss function): valid criterion, e.g. torch.nn.NLLLoss

    Returns:
        (torch.Tensor, float): final output of the RNN, step training loss
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

def train_step_many_to_many(
    rnn,
    category_tensor,
    line_tensor,
    target_tensor,
    learning_rate,
    criterion
):
    """Train a many-to-many rnn for a single step

    Args:
        rnn (Any RNN): RNN model
        category_tensor (torch.Tensor): one-hot vector representation of the category
        line_tensor (torch.Tensor): one-hot vector representation of the input line
        target_tensor (torch.Tensor): one-hot vector representation of the target
        learning_rate (float): gradient descent learning rate
        criterion (Any loss function): valid criterion, e.g. torch.nn.NLLLoss

    Returns:
        (torch.Tensor, float): final output of the RNN, step training loss
    """
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    loss = 0
    rnn.train()
    rnn.zero_grad()
    for i in range(line_tensor.size(0)):
        output, hidden = rnn(category_tensor, line_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    # backpropgation
    loss.backward()

    # TODO: return to this later. I didn't understand it
    for param in rnn.parameters():
        param.data.add_(param.grad.data, alpha=-learning_rate)

    return output, loss.item() / line_tensor.size(0)

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

def sample_rnn(
    rnn,
    category_name,
    all_categories,
    all_letters,
    start_letter='A',
    num_letters=None,
    max_len=20
):
    """Generate a name name for a given category

    Args:
        rnn (NameGeneratorRNN | Any RNN): RNN model
        category_name (str): category name (e. g. 'Portuguese')
        all_categories (list of str): list of category names
        all_letters (str): string containing all valid letters
        start_letter (str, optional): name start letter. Defaults to 'A'.
        num_letters (int, optional): number of letters (it can be different from
                           len(all_letters) due to special tokens). Defaults to None.
        max_len (int, optional): max name length. Defaults to 20.

    Returns:
        str: generated name
    """
    with torch.no_grad():
        cat_tensor = category2tensor(category_name, all_categories)
        input_tensor = line2tensor(start_letter, all_letters, num_letters)
        hidden_tensor = rnn.init_hidden()

        output_name = start_letter

        for _ in range(max_len):
            output, hidden_tensor = rnn(cat_tensor, input_tensor[0], hidden_tensor)
            _, topi = output.topk(1)
            topi = topi[0][0]
            if topi == num_letters - 1:
                # if EOS token
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_tensor = line2tensor(letter, all_letters, num_letters)

        return output_name

def samples(
    rnn,
    category_name,
    all_categories,
    all_letters,
    start_letters,
    num_letters=None,
    max_len=20
):
    """Print name samples for a given string of start letters

    Args:
        rnn (NameGeneratorRNN | Any RNN): RNN model
        category_name (str): category name (e. g. 'Portuguese')
        all_categories (list of str): list of category names
        all_letters (str): string containing all valid letters
        start_letters (str): string of start letters
        num_letters (int, optional): number of letters (it can be different from
                           len(all_letters) due to special tokens). Defaults to None.
        max_len (int, optional): max name length. Defaults to 20.
    """
    for start_letter in start_letters:
        generated_name = \
            sample_rnn(
                rnn,
                category_name,
                all_categories,
                all_letters,
                start_letter,
                num_letters,
                max_len
            )
        print(generated_name)

"""
Module with utils for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import re
import time
import random
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim

from lang import Lang

def unicode2ascii(s):
    """
    Returns a plain ASCII string from a unicode one
    (thanks to https://stackoverflow.com/a/518232/2809427)
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
    )

def normalize_string(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)     # add a space before '.', '!' and '?'
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # replace special characters

    return s

def read_langs(lang1, lang2, reverse=False):
    """Read sentences and instantiate Lang objects

    Args:
        lang1 (str): name of the first language
        lang2 (str): name of the second language
        reverse (bool, optional): reverse language orders? Defaults to False.

    Returns:
        (Lang, Lang, list): input language, output language, sentence pairs
    """
    print('Reading the lines...')

    # read the file and split into lines
    lines = open(f'data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')

    # create pairs (they are separated by a tab ('\t'))
    pairs = [[normalize_string(sentence) for sentence in line.split('\t')] for line in lines]

    # reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filter_pair(pair, max_length, prefixes=None):
    """Check if the pair should be kept

    Args:
        pair (list of str): translation string pairs
        max_length (int): max length of the pairs in terms of tokens
        prefixes (tuple of str, optional): tuple of wanted prefixes. Defaults to None.

    Returns:
        bool: indicates if the pair should be kept (True) or not (False)
    """
    is_smaller = (len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length)

    if prefixes is None:
        return is_smaller
    else:
        return is_smaller and pair[1].startswith(prefixes)

def filter_pairs(pairs, max_length, prefixes):
    """Filter pairs according to max_length and prefixes

    Args:
        pairs (list): list of string pairs (list)
        max_length (int): maximum allowed length for each sentence
        prefixes (tuple of str): valid prefixes

    Returns:
        list: filtered list of string pairs (list)
    """
    return [pair for pair in pairs if filter_pair(pair, max_length, prefixes)]

def prepare_data(lang1, lang2, reverse=False, max_length=10, prefixes=None):
    """Prepare data by instantiating language objects and reading sentence pairs

    Args:
        lang1 (str): name of first language
        lang2 (str): name of second language
        reverse (bool, optional): reverse language order? Defaults to False.
        max_length (int, optional): maximum number of tokens for each sentence. Defaults to 10.
        prefixes (tuple of str, optional): valid prefixes. Defaults to None.

    Returns:
        tuple(Lang, Lang, list): input language object, output_lang object, and list of pairs
    """
    # instantiate languages and read pairs
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')

    # filter pairs
    pairs = filter_pairs(pairs, max_length, prefixes)

    # add language sentences
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print('Counted words:')
    print(f'{input_lang.name} : {input_lang.num_words}')
    print(f'{output_lang.name} : {output_lang.num_words}')

    return input_lang, output_lang, pairs

def indices_from_sentence(lang, sentence):
    """Get the indices of the sentence tokens for a given language in a list format

    Args:
        lang (Lang): language of the sentence
        sentence (str): a word sentence

    Returns:
        list: indices of the tokens of the sentence
    """
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence, device):
    """Get the indices of the sentence tokens for a given language in a tensor format

    Args:
        lang (Lang): language of the sentence
        sentence (str): a word sentence
        device (str): device of the tensor (e.g. 'cpu')

    Returns:
        torch.Tensor: indices of the tokens of the sentence
    """
    indices = indices_from_sentence(lang, sentence)
    indices.append(lang.EOS_token)

    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair, input_lang, output_lang, device):
    """Get the indices of translation pairs in tensor format

    Args:
        pair (list of str): a translation pair
        input_lang (Lang): language to translate from
        output_lang (Lang): language to translate to
        device (str): device of the tensor (e.g. 'cpu')

    Returns:
        (torch.Tensor, torch.Tensor): a tuple of the pair tensors
    """
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)

    return input_tensor, target_tensor

def train_step(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length,
    input_lang,
    output_lang,
    teacher_forcing_ratio,
    device,
):
    """[summary]

    Args:
        input_tensor (torch.Tensor): [description]
        target_tensor (torch.Tensor): [description]
        encoder (EncoderRNN): encoder of the model
        decoder (AttentionDecoderRNN): decoder of the model
        encoder_optimizer (Any): any valid optimizer (e. g. torch.optim.SGD)
        decoder_optimizer (Any): any valid optimizer (e. g. torch.optim.SGD)
        criterion (Any): any valid criterion (e. g. torch.optim.SGD)
        max_length (int): maximum length of the sentence pairs
        input_lang (Lang): language to translate from
        output_lang (Lang): language to translate to
        teacher_forcing_ratio (float, optional): fraction of the time that teacher forcing is used.
        device (str): device of the tensor (e.g. 'cpu')

    Returns:
        [type]: [description]
    """
    encoder_hidden = encoder.init_hidden(device)

    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    input_length = input_tensor.size(0)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[input_lang.SOS_token]], device=device)

    target_length = target_tensor.size(0)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # Teacher forcing
    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)

            # TODO: check this part as I didn't understand it very well
            decoder_input = topi.squeeze().detach() # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == output_lang.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def as_minutes(seconds):
    """Converts a timestamp to a string

    Args:
        seconds (float): a timestamp in seconds

    Returns:
        str: timestamp in as a minutes:seconds string
    """
    minutes = seconds // 60
    seconds -= minutes * 60

    return f'{int(minutes):02d}:{int(seconds):02d}'

def time_since(since, percent):
    """Returns a string indicating the elapsed time since a period

    Args:
        since (float): a timestamp in seconds
        percent (float): percentage concluded of the task

    Returns:
        str: a string representation of the elapsed time and the remaining time
    """
    now = time.time()
    seconds = now - since
    remaining_seconds = seconds/percent - seconds

    return f'{as_minutes(seconds)} (- {as_minutes(remaining_seconds)})'

def train_iters(
    encoder,
    decoder,
    pairs,
    max_length,
    input_lang,
    output_lang,
    num_iters,
    learning_rate=0.01,
    print_every=1000,
    plot_every=100,
    teacher_forcing_ratio=0.5,
    device='cpu'
):
    """[summary]

    Args:
        encoder (EncoderRNN): encoder of the model
        decoder (AttentionDecoderRNN): decoder of the model
        pairs (list of list of str): sentence pairs
        max_length (int): maximum length of the pairs
        input_lang (Lang): language to translate from
        output_lang (Lang): language to translate to
        num_iters (int): number of training iterations
        learning_rate (float, optional): gradient descent learning rate. Defaults to 0.01.
        print_every (int, optional): print loss period. Defaults to 1000.
        plot_every (int, optional): plot loss period. Defaults to 100.
        teacher_forcing_ratio (float, optional): fraction of the time that teacher forcing is used.
                                                 Defaults to 0.5.
        device (str, optional): training device. Defaults to 'cpu'.
    """
    start = time.time()
    plot_losses = []
    print_loss_total, plot_loss_total = 0, 0

    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs), input_lang, output_lang, device)
                      for i in range(num_iters)]

    for step in range(1, num_iters + 1):
        training_pair = training_pairs[step - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_step(input_tensor, target_tensor, encoder, decoder, \
                          encoder_optimizer, decoder_optimizer, criterion, \
                          max_length, input_lang, output_lang, teacher_forcing_ratio, device)

        print_loss_total += loss
        plot_loss_total += loss

        if step % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{time_since(start, step/num_iters)} ({step} ' \
                  f'{step*100/num_iters:.2f}%) {print_loss_avg:.4f}')

        if step % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)

def show_plot(points):
    """Plot the given points

    Args:
        points (list of float): list of points to be plotted
    """
    plt.switch_backend('agg')

    plt.figure()
    _, ax = plt.subplots()
    # this locator puts ticks as regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length, device):
    """Evaluate the trained RNN with a given sentence

    Args:
        encoder (EncoderRNN): encoder of the model
        decoder (AttentionDecoderRNN): decoder of the model
        sentence (str): sentence to translate
        input_lang (Lang): language to translate from
        output_lang (Lang): language to translate to
        max_length (int): maximum length of the pairs
        device (str): device for the evaluation (e.g. 'cpu')

    Returns:
        (list of str, torch.Tensor): list of decoded words and attention weight tensor
    """
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        input_tensor = tensor_from_sentence(input_lang, sentence, device)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden(device)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[input_lang.SOS_token]], device) # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            _, topi = decoder_output.data.topk(1)
            if topi.item() == output_lang.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_randomly(encoder, decoder, pairs, input_lang, output_lang, max_length, device, n=10):
    """Evaluate the trained RNN with random sentences

    Args:
        encoder (EncoderRNN): encoder of the model
        decoder (AttentionDecoderRNN): decoder of the model
        pairs (list of list of str): sentence pairs
        input_lang (Lang): language to translate from
        output_lang (Lang): language to translate to
        max_length (int): maximum length of the pairs
        device (str): device for the evaluation (e.g. 'cpu')
        n (int, optional): [description]. Defaults to 10.
    """

    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = \
            evaluate(encoder, decoder, pair[0], input_lang, output_lang, max_length, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print()

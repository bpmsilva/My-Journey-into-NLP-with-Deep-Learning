"""
Module with utils for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import re
import unicodedata

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

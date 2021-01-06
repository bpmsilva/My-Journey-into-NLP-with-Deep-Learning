"""
Module with Lang class for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
class Lang:
    """
    Simple Language class used to track word tokens
    """
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Lang.SOS_token: 'SOS', Lang.EOS_token: 'EOS'}
        self.num_words = 2 # SOS and EOS

    def add_sentence(self, sentence):
        """
        Add the sentence words to the language
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Add a word to the language
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

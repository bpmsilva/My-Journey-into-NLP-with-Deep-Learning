"""
Main module for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
from utils import prepare_data, train_iters
from models import EncoderRNN, AttentionDecoderRNN

def main():
    """
    Main function for the translation RNN
    """
    device = 'gpu'
    dropout = 0.1
    max_length = 10
    num_iters = 75000
    hidden_size = 256
    print_every = 5000
    teacher_forcing_ratio = 0.5

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    input_lang, output_lang, pairs = \
        prepare_data('eng', 'fra', reverse=True, max_length=10, prefixes=eng_prefixes)
    # print(random.choice(pairs))

    encoder = EncoderRNN(input_lang.num_words, hidden_size).to(device)
    decoder = AttentionDecoderRNN(
            hidden_size,
            output_lang.num_words,
            max_length,
            dropout
        ).to(device)

    train_iters(
        encoder,
        decoder,
        pairs,
        max_length,
        input_lang,
        output_lang,
        num_iters,
        print_every=print_every,
        teacher_forcing_ratio=teacher_forcing_ratio)

if __name__ == "__main__":
    main()

"""
Main module for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import argparse
import matplotlib.pyplot as plt

import torch

from models import EncoderRNN, AttentionDecoderRNN
from utils import \
        prepare_data, train_iters, evaluate_randomly, evaluate, evaluate_and_show_attention

def parse_args():
    """
    Read the input arguments or apply the default ones
    """
    parser = argparse.ArgumentParser('Training and evaluation of translation RNN with attention')

    # add arguments
    parser.add_argument('--train', help='train the neural network?', default=False, type=bool)
    parser.add_argument('--device', help='device to train or evaluate on', default='cuda', type=str)
    parser.add_argument('--dropout', help='dropout rate', default=0.1, type=float)
    parser.add_argument(
        '--max-length',
        help='maximum length of the sentences',
        default=10,
        type=int
    )
    parser.add_argument(
        '--num-iters',
        help='number of training iterations',
        default=75000,
        type=int
    )
    parser.add_argument('--hidden-size', help='size of the hidden state', default=256, type=int)
    parser.add_argument('--print-every', help='loss log period', default=5000, type=int)
    parser.add_argument(
        '--teacher-forcing-ratio',
        help='fraction of teacher forcing',
        default=0.5,
        type=float
    )

    return parser.parse_args()

def main():
    """
    Main function for the translation RNN
    """
    args = parse_args()

    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    input_lang, output_lang, pairs = \
        prepare_data('eng', 'fra', reverse=True, max_length=args.max_length, prefixes=eng_prefixes)
    # print(random.choice(pairs))

    encoder = EncoderRNN(input_lang.num_words, args.hidden_size).to(args.device)
    decoder = AttentionDecoderRNN(
            args.hidden_size,
            output_lang.num_words,
            args.max_length,
            args.dropout
        ).to(args.device)

    if args.train:
        train_iters(
            encoder,
            decoder,
            pairs,
            args.max_length,
            input_lang,
            output_lang,
            args.num_iters,
            device=args.device,
            print_every=args.print_every,
            teacher_forcing_ratio=args.teacher_forcing_ratio)

        torch.save(encoder.state_dict(), 'encoder.pth')
        torch.save(decoder.state_dict(), 'decoder.pth')

    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

    encoder.eval()
    decoder.eval()

    evaluate_randomly(
        encoder,
        decoder,
        pairs,
        input_lang,
        output_lang,
        args.max_length,
        args.device,
        n=10
    )

    # visualizing attention
    _, attentions = \
        evaluate(
            encoder,
            decoder,
            'je suis trop froid .',
            input_lang,
            output_lang,
            args.max_length,
            args.device
        )

    plt.matshow(attentions.cpu().numpy())

    input_sentences = ['elle a cinq ans de moins que moi .',
                       'elle est trop petit .',
                       'je ne crains pas de mourir .',
                       'c est un jeune directeur plein de talent .']

    for input_sentence in input_sentences:
        evaluate_and_show_attention(
            encoder,
            decoder,
            input_sentence,
            input_lang,
            output_lang,
            args.max_length,
            args.device
        )

if __name__ == "__main__":
    main()

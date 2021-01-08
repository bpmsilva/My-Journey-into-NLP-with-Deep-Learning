"""
Module with RNN encoder e decoder classes for the translation RNN with attention
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """
    RNN encoder class
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden_tensor):
        """
        Forward pass of the neural netword
        """
        embedded_tensor = self.embedding(input_tensor).view(1, 1, -1)
        output_tensor, hidden_tensor = self.gru(embedded_tensor, hidden_tensor)
        return output_tensor, hidden_tensor

    def init_hidden(self, device):
        """
        Init the hidden state of the RNN
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    """
    Simple RNN decoder class (no attention)
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        """
        Forward pass of the neural netword
        """
        output_tensor = self.embedding(input_tensor).view(1, 1, -1)
        output_tensor = F.relu(output_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor, hidden_tensor)
        output_tensor = self.softmax(self.out(output_tensor[0]))

        return output_tensor, hidden_tensor

class AttentionDecoderRNN(nn.Module):
    """
    RNN decoder class with attention
    """
    def __init__(self, hidden_size, output_size, max_length, dropout=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        # TODO: understand this nn.Embedding better
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(2 * self.hidden_size, self.max_length)
        self.attention_combine = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden_tensor, encoder_outputs):
        """
        Forward pass of the neural netword
        """
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden_tensor[0]), dim=1))
        )
        attention_applied = torch.bmm(
            attention_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        output_tensor = torch.cat((embedded[0], attention_applied[0]), 1)
        output_tensor = self.attention_combine(output_tensor).unsqueeze(0)

        output_tensor = F.relu(output_tensor)
        output_tensor, hidden_tensor = self.gru(output_tensor, hidden_tensor)

        output_tensor = F.log_softmax(self.out(output_tensor[0]), dim=1)

        return output_tensor, hidden_tensor, attention_weights

    def init_hidden(self, device):
        """
        Init the hidden state of the RNN
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""
Module with character level RNN class for name generation
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import torch
import torch.nn as nn

class NameGeneratorRNN(nn.Module):
    """
    Character level name generator RNN
    """
    def __init__(self, input_size, hidden_size, output_size, num_categories, dropout=0.1):
        """Constructor of the level name generator RNN

        Args:
            input_size (int): input size (excluding number of categories and hidden output size)
            hidden_size (int): hidden state size
            output_size (int): output size (usually the number of characters)
            num_categories (int): number of categories (increases the number of neurons)
            dropout (float, optional): dropout of the final layer. Defaults to 0.1.
        """
        super(NameGeneratorRNN, self).__init__()
        self.hidden_size = hidden_size

        num_input_neurons = input_size + hidden_size + num_categories

        # layers with weights
        self.i2h = nn.Linear(num_input_neurons, hidden_size)
        self.i2o = nn.Linear(num_input_neurons, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

        # besides preventing overfitting,
        # the dropout layer also helps to increase the sampling variety
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category_tensor, input_tensor, hidden_tensor):
        """Forward pass of the RNN

        Args:
            category_tensor (torch.Tensor): category one-hot vector representation
            input_tensor (torch.Tensor): input one-hot vector representation
            hidden_tensor (torch.Tensor): hidden state tensor

        Returns:
            (torch.Tensor, torch.Tensor): output and hidden state tensors
        """
        # create current input
        combined_input = torch.cat((category_tensor, input_tensor, hidden_tensor), 1)

        # compute "hidden" tensors (output_tensor variable name maybe misleading)
        hidden_tensor = self.i2h(combined_input)
        output_tensor = self.i2o(combined_input)

        # compute output
        output_combined = torch.cat((hidden_tensor, output_tensor), 1)
        output_tensor = self.o2o(output_combined)
        output_tensor = self.dropout(output_tensor)
        output_tensor = self.softmax(output_tensor)

        return output_tensor, hidden_tensor

    def init_hidden(self):
        """Initializes the hidden state

        Returns:
            torch.Tensor: null hidden state
        """
        return torch.zeros(1, self.hidden_size)

"""
Module with simple RNN class
Original Author: Sean Robertson (https://github.com/spro)
Modified by: Bernardo Silva (https://github.com/bpmsilva)
"""
import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Simple RNN class
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        """RNN forward pass

        Args:
            input_tensor (torch.tensor): neural network input tensor
            hidden_tensor (torch.tensor): neural network hidden state tensor

        Returns:
            (torch.tensor, torch.tensor): output and hidden state tensors
        """
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        # TODO: shouldn't have a nonlinearity here?
        hidden_tensor = self.i2h(combined)
        output_tensor = self.i2o(combined)
        output_tensor = self.softmax(output_tensor)

        return output_tensor, hidden_tensor

    def init_hidden(self):
        """Initialize hidden state with zeros

        Returns:
            torch.tensor: zeros hidden state tensor
        """
        return torch.zeros(1, self.hidden_size)

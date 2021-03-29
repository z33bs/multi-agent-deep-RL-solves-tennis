import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, output_activation, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = []

        for i in range(len(hidden_sizes)):
            layer_output_size = hidden_sizes[i] if i + 1 == len(hidden_sizes) else hidden_sizes[i + 1]
            layer = nn.Linear(hidden_sizes[i], layer_output_size)
            self.hidden.append(layer)

        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.output_gate = output_activation
        self.reset_parameters()

    def reset_parameters(self):
        self.input.weight.data.uniform_(*self.hidden_init(self.input))
        for layer in self.hidden:
            layer.weight.data.uniform_(*self.hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.output(x)
        if self.output_gate is not None:
            x = self.output_gate(x)
        return x

    def soft_update(self, source, tau):
           for target_param, source_param in zip(self.parameters(), source.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def hidden_init(self, layer):
        lim = 1. / np.sqrt(layer.weight.data.size()[0])
        return (-lim, lim)
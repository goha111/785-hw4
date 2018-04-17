import torch
import torch.nn as nn
import torch.nn.functional as F


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()

class VLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get('hidden_size')
        bidirectional = kwargs.get('bidirectional')
        num_layers = kwargs.get('num_layers')
        num_directions = 2 if bidirectional else 1
        self.h0 = nn.Parameter(torch.zeros(num_layers * num_directions, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers * num_directions, 1, hidden_size))
        self.apply(initializer)

    def forward(self, input, hx=None):
        batch_size = input.shape[1]
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        super().forward(input, (h0, c0))

class Listener(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        self.apply(initializer)

    def forward(self, *input):
        pass

class Speller(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        self.apply(initializer)

    def forward(self, *input):
        pass

class LASNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Phi = nn.ModuleList([

        ])
        self.Psi = nn.ModuleList([

        ])

        decoder_h0 = nn.Parameter(torch.zeros(1, ))
        decoder_c0 = nn.Parameter(torch.zeros(1, ))
        # TODO
        self.apply(initializer)

    def forward(self, input, forward=0):
        pass
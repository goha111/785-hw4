import torch
import torch.nn as nn
import torch.nn.functional as F


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()


class VInitLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get('hidden_size')
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, input, hx=None):
        batch_size = input.shape[1]
        h0 = self.h0.expand(-1, batch_size, -1)
        c0 = self.c0.repeat(-1, batch_size, -1)
        super().forward(input, (h0, c0))


class Listener(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass

class Speller(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass

class LASModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass
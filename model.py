import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()

def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()

class VLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        num_directions = 2 if self.lstm.bidirectional else 1
        self.h_0 = nn.Parameter(torch.zeros(self.lstm.num_layers * num_directions, 1, self.lstm.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(self.lstm.num_layers * num_directions, 1, self.lstm.hidden_size))

    def forward(self, seqs, seq_lens):
        batch_size = seqs.shape[1]
        h_0 = self.h_0.expand(-1, batch_size, -1).contiguous()
        c_0 = self.c_0.expand(-1, batch_size, -1).contiguous()
        h = pack_padded_sequence(seqs, seq_lens)
        h, _ = self.lstm(h, (h_0, c_0))
        return pad_packed_sequence(h)

class SequencePooling(nn.Module):
    def forward(self, seqs, seq_lens):
        if seqs.shape[0] % 2 == 1:  # odd number of timestamps
            seqs = seqs[:-1]        # remove the last frame
        L, N, C = seqs.shape
        # (L, N, C) -> (N, L, C) -> (N, L/2, C*2) -> (L/2, N, C*2)
        seqs = seqs.transpose(0, 1).view(N, L / 2, C * 2).transpose(0, 1).contiguous()
        seq_lens = seq_lens / 2
        return seqs, seq_lens

class PBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.pooling = SequencePooling()
        self.blstm = VLSTM(input_size=input_size * 2, hidden_size=hidden_size, bidirectional=True)

    def forward(self, seqs, seq_lens):
        seqs, seq_lens = self.pooling(seqs, seq_lens)
        seqs, seq_lens = self.blstm(seqs, seq_lens)
        return seqs, seq_lens

class Listener(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.blstms = nn.ModuleList([
            VLSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
        ])

    def forward(self, seqs, seq_lens):
        for module in self.blstms:
            seqs, seq_lens = module(seqs, seq_lens)
        return seqs, seq_lens

class Speller(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, *input):
        pass

class LASModel(nn.Module):
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
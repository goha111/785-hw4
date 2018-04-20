from char_list import *

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()
        elif param.dim() == 1:
            nn.init.xavier_uniform(param.data)
        else:
            nn.init.xavier_uniform(param.data)

def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return -torch.log(eps - torch.log(U + eps))

def to_variable(array):
    return Variable(torch.from_numpy(array).contiguous())

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    # input (N, L, C)
    # target(N, L)
    def forward(self, input, target):
        N, L, _ = input.shape
        return super().forward(input.view(-1, input.shape[2]), target.view(-1)).view(N, L)

class DataLoader():
    def __init__(self, x, y, batch_size=16):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.size = len(x)
        self.channel = x[0].shape[1]
        self.perm_idx = np.arange(self.size)
        self.len = ((self.size - self.batch_size) // self.batch_size) + 1

    def __iter__(self):
        np.random.shuffle(self.perm_idx)
        for i in range(0, self.size - self.batch_size, self.batch_size):
            idx = self.perm_idx[i:i + self.batch_size]
            seqs, labels = self.x[idx], self.y[idx]
            seq_lengths = torch.IntTensor(list(map(lambda x: x.shape[0], seqs)))
            # because transcript contains SOS and EOS already
            label_lengths = torch.IntTensor(list(map(lambda x: x.shape[0], labels))) - 1
            max_seq_len = seq_lengths.max()
            max_label_len = label_lengths.max()

            # allocate spaces
            seq_padded = np.zeros((self.batch_size, max_seq_len, self.channel), dtype=float)  # (n, max_len, channel)
            label_in_padded = np.zeros((self.batch_size, max_label_len), dtype=int)
            label_out_padded = np.zeros((self.batch_size, max_label_len), dtype=int)
            label_mask = np.zeros((self.batch_size, max_label_len), dtype=float)

            for i, (seq, seq_len, label, label_len) in enumerate(zip(seqs, seq_lengths, labels, label_lengths)):
                seq_padded[i, :seq_len, :] = seq
                label_in_padded[i, :label_len] = label[: -1]
                label_out_padded[i, :label_len] = label[1: ]
                label_mask[i, :label_len] = 1

            # sort tensors by lengths
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_padded = to_variable(seq_padded[perm_idx]).float().transpose(0, 1).contiguous()
            label_in_padded = to_variable(label_in_padded[perm_idx]).long()
            label_out_padded = to_variable(label_out_padded[perm_idx]).long()
            label_mask = to_variable(label_mask[perm_idx]).float()

            # seq_padded: (max_seq_len, n, channel)
            # seq_lengths: (n,)
            # lable_length: (n,)
            # label_in_padded, label_out_padded, label_mask: (n, max_label_len)
            yield (seq_padded, seq_lengths, label_in_padded, label_out_padded, label_mask)

    def __len__(self):
        return self.len

class MLLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=.0):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            self.lstm_layers.append(nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size))

    def forward(self, input, hidden, cell):
        for i, lstm in enumerate(self.lstm_layers):
            hidden[i], cell[i] = lstm(input, (hidden[i], cell[i]))
            input = self.dropout(hidden[i])
        return hidden, cell

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
        c_0 = Variable(h_0.data.new(h_0.shape).zero_())
        input = pack_padded_sequence(seqs, seq_lens)
        output, _ = self.lstm(input, (h_0, c_0))
        seqs, _ = pad_packed_sequence(output)
        return seqs, seq_lens

class SequencePooling(nn.Module):

    # seqs: (L, N, C)
    # seq_lens (N, )
    def forward(self, seqs, seq_lens):
        L, N, C = seqs.shape
        if L % 2 == 1:      # odd number of timestamps
            seqs = seqs[:-1]     # remove the last frame
        # (L, N, C) -> (N, L, C) -> (N, L/2, C*2) -> (L/2, N, C*2)
        seqs = seqs.transpose(0, 1).contiguous().view(N, L // 2, C * 2).transpose(0, 1).contiguous()
        seq_lens = np.clip(seq_lens // 2, a_min=1, a_max=None)
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

class MLP(nn.ModuleList):
    def __init__(self, layers, activation=None):
        super().__init__()
        for i, layer in enumerate(layers):
            self.append(layer)
            if activation is not None and i != len(layers) - 1:
                self.append(activation)

    def forward(self, input):
        h = input
        for module in self:
                h = module(h)
        return h

class Listener(nn.Module):
    def __init__(self, input_size=40, hidden_size=256):
        super().__init__()
        self.blstms = nn.ModuleList([
            # TODO: batchNorm1d at the beginning
            VLSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size),
        ])

    def forward(self, seqs, seq_lens):
        for lstm in self.blstms:
            seqs, seq_lens = lstm(seqs, seq_lens)
        return seqs, seq_lens

class Speller(nn.Module):
    def __init__(self, char_dict_size=33, input_size=512, hidden_size=256, query_size=128):
        super().__init__()
        self.char_dict_size = char_dict_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(char_dict_size, hidden_size)
        self.inith = nn.ParameterList()
        self.rnns = MLLSTMCell(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=3, dropout=.2)
        for i in range(3):
            self.inith.append(nn.Parameter(torch.zeros(1, hidden_size)))

        activation = nn.ELU()
        # map hidden states to queries
        self.query_net = MLP([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, query_size)
        ], activation)

        # map input to keys
        self.key_net = MLP([
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, query_size)
        ], activation)

        # map input to values
        self.value_net = MLP([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ], activation)

        self.output_layer = MLP([
            nn.Linear(hidden_size*2, hidden_size),
            nn.Linear(hidden_size, char_dict_size)
        ])

        # weight tying
        self.output_layer[-1].weight= self.embedding.weight

    def apply_projection_bias(self, projection_bias):
        self.output_layer[-1].bias.data = projection_bias

    # query:    (N, Q, 1)
    # key:      (N, L, Q)
    # value:    (N, L, V)
    # seq_lens: (N, )
    #
    # return    (N, V)
    def attention_context(self, query, key, value, seq_lens):
        N, L, _ = key.shape

        # (N, L, Q) @ (N, Q, 1) -> (N, L, 1) -> (N, 1, L) -> softmax along L -> (N, 1, L)
        attention = F.softmax(torch.bmm(key, query).transpose(1, 2), dim=2)
        attention_mask = attention.data.new(N, L).zero_()   # create mask on the same device and set to zeros
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = 1.
        attention_mask = Variable(attention_mask).unsqueeze(1)   # (N, L) -> (N, 1, L)
        attention = (attention * attention_mask).clamp(min=1e-9)   # multiplied by mask (N, 1, L) and make it num stable
        attention_sum = attention.sum(dim=2, keepdim=True)    # (N, 1, 1) and it will auto broadcast
        attention = attention / attention_sum     # re-normalize the attention weights

        # (N, 1, L) @ (N, L, V) -> (N, 1, V) -> (N, V)
        context = torch.bmm(attention, value).view(N, -1)
        return context

    # seqs(h): (L, N, C)
    # seq_lens: (N)
    # label_in(y): (N, T)
    def forward(self, seqs, seq_lens, label_in):
        N, T = label_in.shape
        # expand initial states of LSTMCell to batch size
        hidden = [tensor.repeat(N, 1) for tensor in self.inith]
        cell = [Variable(h.data.new(h.shape).zero_()) for h in hidden]
        output = []
        char_input = self.embedding(label_in)     # (N, max_trans_len, E)
        key = self.key_net(seqs).transpose(0, 1)        # (L, N, input_size) -> (L, N, Q) -> (N, L, Q)
        value = self.value_net(seqs).transpose(0, 1)    # (L, N, input_size) -> (L, N, V) -> (N, L, V)
        query = self.query_net(hidden[-1]).unsqueeze(-1)    # (N, hidden_size) -> (N, Q) -> (N, Q, 1)
        prev_context = self.attention_context(query, key, value, seq_lens)    # (N, V)

        # 1. do LSTM with prev context and prev states
        # 2. get query with new hidden states
        # 3. get current context
        # 4. concatenate current context with current hidden state and then feed into output_network
        for t in range(T):
            # (N, E) concat (N, V) -> (N, E+V)
            # TODO: sample from prev prediction with some probability
            rnn_input = torch.cat((prev_context, char_input[:,t,:]), dim=1)
            hidden, cell = self.rnns(rnn_input, hidden, cell)
            query = self.query_net(hidden[-1]).unsqueeze(-1)    # calculate current query
            curr_context = self.attention_context(query, key, value, seq_lens)   # calculate current context
            output.append(self.output_layer(torch.cat((curr_context, hidden[-1]), dim=1)))
            prev_context = curr_context
        return torch.stack(output)

class LASModel(nn.Module):
    def __init__(self, char_dict_size, projection_bias=None):
        super().__init__()
        self.listener = Listener()
        self.speller = Speller(char_dict_size=char_dict_size)
        self.apply(initializer)
        if projection_bias is not None:
            self.speller.apply_projection_bias(projection_bias)

    def forward(self, seqs, seq_lens, label_in):
        h, h_len = self.listener(seqs, seq_lens)
        output = self.speller(h, h_len, label_in)
        return output
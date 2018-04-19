import argparse
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

class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    # input (N, L, C)
    # target(N, L)
    def forward(self, input, target):
        assert(input.shape[0] == target.shape[0])
        return super().forward(input.view(-1, input.shape[2]), target.view(-1))

class DataLoader():
    def __init__(self, x, y, batch_size=16):
        self.x = x
        self.y = y
        assert(len(x) == len(y))
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
            # because transcripts contains SOS and EOS already
            label_lengths = torch.IntTensor(list(map(lambda x: x.shape[0], labels))) - 1
            max_seq_len = seq_lengths.max()
            max_label_len = label_lengths.max()

            # allocate spaces
            seq_padded = torch.FloatTensor(self.batch_size, max_seq_len, self.channel).zero_()  # (n, max_len, channel)
            label_in_padded = torch.IntTensor(self.batch_size, max_label_len).zero_()
            label_out_padded = torch.IntTensor(self.batch_size, max_label_len).zero_()
            label_mask = torch.FloatTensor(self.batch_size, max_label_len).zero_()

            for i, (seq, seq_len, label, label_len) in enumerate(zip(seqs, seq_lengths, labels, label_lengths)):
                seq_padded[i, :seq_len, :] = torch.FloatTensor(seq)
                label_in_padded[i, :label_len] = torch.FloatTensor(label[: -1])
                label_out_padded[i, :label_len] = torch.FloatTensor(label[1: ])
                label_mask[i, :label_len] = 1

            # sort tensors by lengths
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_lengths = Variable(seq_lengths)
            seq_padded = Variable(seq_padded[perm_idx]).transpose(0, 1).continuous()
            label_lengths = Variable(label_lengths[perm_idx])
            label_in_padded = Variable(label_in_padded[perm_idx])
            label_out_padded = Variable(label_out_padded[perm_idx])
            label_mask = Variable(label_mask[perm_idx])

            # seq_padded: (max_seq_len, n, channel)
            # seq_lengths: (n,)
            # lable_length: (n,)
            # label_in_padded, label_out_padded, label_mask: (n, max_label_len)
            yield (seq_padded, seq_lengths, label_in_padded, label_out_padded, label_lengths, label_mask)

class MLLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.lstm_layers.append(nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size))

    def forward(self, input, hidden, cell):
        for i, lstm in enumerate(self.lstm_layers):
            hidden[i], cell[i] = lstm(input, (hidden[i], cell[i]))
            input = hidden[i]
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
        c_0 = self.c_0.expand(-1, batch_size, -1).contiguous()
        h = pack_padded_sequence(seqs, seq_lens)
        h, _ = self.lstm(h, (h_0, c_0))
        seqs, seq_lens = pad_packed_sequence(h)
        return seqs, seq_lens

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

class MLP(nn.Module):
    def __init__(self, layers, activation=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if activation:
            self.activation = activation
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        h = input
        for i, module in enumerate(self.layers):
            if i == len(self.layers) - 1:
                h = module(h)
            else:
                h = self.activation(module(h))
        return h

class Listener(nn.Module):
    def __init__(self, input_size=40, hidden_size=256):
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
    def __init__(self, char_dict_size=34, input_size=512, hidden_size=256, query_size=64, projection_bias=None):
        super().__init__()
        self.char_dict_size = char_dict_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(char_dict_size, hidden_size)
        self.init_context = nn.Parameter((torch.zeros(1, hidden_size)))
        self.inith = nn.ParameterList()
        self.initc = nn.ParameterList()
        self.rnns = MLLSTMCell(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=3)
        self.elu = nn.ELU()
        for i in range(3):
            self.inith.append(nn.Parameter(torch.zeros(1, hidden_size)))
            self.initc.append(nn.Parameter(torch.zeros(1, hidden_size)))

        # map hidden states to queries
        self.query_net = MLP([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, query_size)
        ], self.elu)

        # map input to keys
        self.key_net = MLP([
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, query_size)
        ], self.elu)

        # map input to values
        self.value_net = MLP([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ], self.elu)

        self.output_layer = nn.Linear(hidden_size*2, char_dict_size)
        if projection_bias:
            self.output_layer.bias.data = projection_bias

        # weight tying
        self.output_layer.weight= self.embedding.weight

    # query:    (N, Q, 1)
    # key:      (N, L, Q)
    # value:    (N, L, V)
    # seq_lens: (N, )
    #
    # return    (N, V)
    def attention_context(self, query, key, value, seq_lens):
        N = query.shape[0]
        L = seq_lens.data.max()  # L

        # (N, L, Q) @ (N, Q, 1) -> (N, L, 1) -> (N, 1, L) -> softmax along L -> (N, 1, L)
        # TODO: here should use masked softmax, and the result should be normalized with respect to the length of L
        attention = F.softmax(torch.bmm(key, query).transpose(1, 2), dim=2)
        attention_mask = attention.data.new(N, L)
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = 1
        attention_mask = Variable(attention_mask).unsqueeze_(1)   # (N, L) -> (N, 1, L)
        attention = attention * attention_mask    # multiplied by mask  (N, 1, L)
        attention_sum = attention.sum(dim=2).expand(-1, -1, L)    # (N, 1, 1)
        attention = attention / attention_sum

        # (N, 1, L) @ (N, L, V) -> (N, 1, V) -> (N, V)
        context = torch.bmm(attention, value).view(attention.shape[0], -1)
        return context

    # seqs(h): (Lmax, N, C)
    # seq_lens: (N)
    # label(y): (max_trans_len, N)
    # label_lens: (N)
    def forward(self, seqs, seq_lens, labels, label_lens):
        T, N = labels.shape
        # expand initial states of LSTMCell to batch size
        hidden = [tensor.repeat(N, 1) for tensor in self.inith]
        cell = [tensor.repeat(N, 1) for tensor in self.initc]
        output = torch.zeros((T, N, self.char_dict_size))
        char_input = self.embedding(labels)     # (max_trans_len, N, E)
        key = self.key_net(seqs).transpose(0, 1)        # (L, N, input_size) -> (L, N, Q) -> (N, L, Q)
        value = self.value_net(seqs).transpose(0, 1)    # (L, N, input_size) -> (L, N, V) -> (N, L, V)
        query = self.query_net(hidden[-1]).unsqueeze(-1)    # (N, hidden_size) -> (N, Q) -> (N, Q, 1)
        prev_context = self.attention_context(query, key, value, seq_lens)    # (N, V)
        for t in range(T):
            # (N, E) concat (N, V) -> (N, E+V)
            rnn_input = torch.cat((prev_context, char_input[t]), dim=1)
            hidden, cell = self.rnns(rnn_input, hidden, cell)
            query = self.query_net(hidden[-1]).unsqueeze(-1)
            curr_context = self.attention_context(query, key, value)
            output[t,:,:] = self.output_layer(torch.cat((curr_context, hidden)))
            prev_context = curr_context
        return output

class LASModel(nn.Module):
    def __init__(self, projection_bias=None):
        super().__init__()
        self.listener = Listener()
        self.speller = Speller(projection_bias=projection_bias)
        self.apply(initializer)

    def forward(self, seqs, seq_lens, labels, label_lens):
        h, h_len = self.listener(seqs, seq_lens)
        output = self.speller(h, h_len, labels, label_lens)
        return output
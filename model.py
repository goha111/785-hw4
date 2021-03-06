from char_list import *

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


def initializer(m):
    for name, param in m.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.zero_()
        elif param.dim() > 1:
            nn.init.xavier_uniform(param.data)
        else:
            param.data.zero_()

def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return -torch.log(eps - torch.log(U + eps))

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
        self.inith = nn.Parameter(torch.zeros(self.lstm.num_layers * num_directions, 1, self.lstm.hidden_size))
        self.initc = nn.Parameter(torch.zeros(self.lstm.num_layers * num_directions, 1, self.lstm.hidden_size))

    def forward(self, seqs, seq_lens):
        batch_size = seqs.shape[1]
        inith = self.inith.expand(-1, batch_size, -1).contiguous()
        initc = self.initc.expand(-1, batch_size, -1).contiguous()
        input = pack_padded_sequence(seqs, seq_lens)
        output, _ = self.lstm(input, (inith, initc))
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
    def __init__(self, input_size, hidden_size, dropout=.0):
        super().__init__()
        self.pooling = SequencePooling()
        self.dropout = nn.Dropout(dropout)
        self.blstm = VLSTM(input_size=input_size * 2, hidden_size=hidden_size, bidirectional=True)

    def forward(self, seqs, seq_lens):
        seqs, seq_lens = self.pooling(seqs, seq_lens)
        seqs = self.dropout(seqs)
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
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size, dropout=.2),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size, dropout=.2),
            PBLSTM(input_size=hidden_size*2, hidden_size=hidden_size, dropout=.2),
        ])

    # seq: (L, N, C)
    def forward(self, seqs, seq_lens):
        for lstm in self.blstms:
            seqs, seq_lens = lstm(seqs, seq_lens)
        return seqs, seq_lens

class Speller(nn.Module):
    def __init__(self, char_dict_size=CHAR_SIZE, input_size=512, hidden_size=256, query_size=256, feed_forward_ratio=.1):
        super().__init__()
        self.char_dict_size = char_dict_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feed_forward_ratio = feed_forward_ratio
        self.embedding = nn.Embedding(char_dict_size, hidden_size)
        self.rnns = MLLSTMCell(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=3, dropout=.2)
        self.inith = nn.ParameterList()
        self.initc = nn.ParameterList()
        for i in range(3):
            self.inith.append(nn.Parameter(torch.zeros(1, hidden_size)))
            self.initc.append(nn.Parameter(torch.zeros(1, hidden_size)))

        activation = nn.ELU()
        # map hidden states to queries
        self.query_net = nn.Linear(hidden_size, query_size)

        # map input to keys
        self.key_net = nn.Linear(input_size, query_size)

        # map input to values
        self.value_net = nn.Linear(input_size, hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size + query_size, hidden_size),
            nn.Linear(hidden_size, char_dict_size)
        )

        # weight tying
        self.output_layer[-1].weight= self.embedding.weight

    def apply_projection_bias(self, projection_bias):
        self.output_layer[-1].bias.data = projection_bias

    # query:    (N, Q)
    # key:      (N, L, Q)
    # value:    (N, L, V)
    # seq_lens: (N, )
    #
    # return    (N, V)
    def attention_context(self, query, key, value, seq_lens):
        N, L, _ = key.shape
        query = query.unsqueeze(-1)   # (N, Q) -> (N, Q, 1)

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

    # logit: (N, char_size)

    # output:(N, 1) each is one single number of char_index
    def sample_output(self, logit):
        gumbel = Variable(sample_gumbel(shape=logit.size(), out=logit.data.new()))
        logit = logit + gumbel   # (N, char_size)
        char_input = logit.max(dim=1, keepdim=True)[1]  # (N, 1)
        return char_input

    # key:      (N, L, Q)
    # value:    (N, L, V)
    # seq_lens: (N,)
    # prev_cont:(N, V)
    # char_input: (N, E)
    # hidden: list of hidden_states
    # cell:   list of cell_states
    def attention_and_spell(self, key, value, seq_lens, context, char_input, hidden, cell):
        # (N, E) concat (N, V) -> (N, E+V)
        rnn_input = torch.cat([context, char_input], dim=1)
        hidden, cell = self.rnns(rnn_input, hidden, cell)
        query = self.query_net(hidden[-1])  # calculate current query
        context = self.attention_context(query, key, value, seq_lens)  # calculate current context
        logit = self.output_layer(torch.cat([context, query], dim=1))  # (N, char_size)
        return logit, context, hidden, cell

    # seqs(h): (L, N, C)
    # seq_lens: (N)
    # label_in(y): (N, T)

    #output: (T, N, char_size)
    def forward(self, seqs, seq_lens, label_in):
        N, T = label_in.shape
        # expand initial states of LSTMCell to batch size
        hidden = [tensor.repeat(N, 1) for tensor in self.inith]
        cell = [tensor.repeat(N, 1) for tensor in self.initc]
        output = []
        label_in = self.embedding(label_in)     # (N, max_trans_len, E)
        key = self.key_net(seqs).transpose(0, 1)        # (L, N, input_size) -> (L, N, Q) -> (N, L, Q)
        value = self.value_net(seqs).transpose(0, 1)    # (L, N, input_size) -> (L, N, V) -> (N, L, V)
        query = self.query_net(hidden[-1])      # (N, hidden_size) -> (N, Q)
        context = self.attention_context(query, key, value, seq_lens)    # (N, V)

        # 1. do LSTM with prev context and prev states
        # 2. get query with new hidden states
        # 3. get current context
        # 4. concatenate current context with current hidden state and then feed into output_network
        for t in range(T):
            rand_num = np.random.rand()
            if rand_num < self.feed_forward_ratio and t > 0:  # sample from previous output
                char_input = self.sample_output(output[-1])   # (N, 1)
                embedded = self.embedding(char_input)   # (N, 1, E)
                char_input = embedded.view(N, -1)  #(N, E)
            else:
                char_input = label_in[:,t,:]
            logit, context, hidden, cell = self.attention_and_spell(key, value, seq_lens, context, char_input, hidden, cell)
            output.append(logit)
        return torch.stack(output)

    # seqs(h): (L, N, C)
    # seq_lens: (N)
    # label_in(y): (N, T)

    #output: (len_sentence)
    def predict(self, seqs, seq_lens, label_in):
        N, T = label_in.shape
        assert(N == T == 1)   #  one batch, one input, which is SOS
        # expand initial states of LSTMCell to batch size
        hidden = [tensor.repeat(N, 1) for tensor in self.inith]
        cell = [tensor.repeat(N, 1) for tensor in self.initc]
        output = []
        key = self.key_net(seqs).transpose(0, 1)  # (L, N, input_size) -> (L, N, Q) -> (N, L, Q)
        value = self.value_net(seqs).transpose(0, 1)  # (L, N, input_size) -> (L, N, V) -> (N, L, V)
        query = self.query_net(hidden[-1])   # (N, hidden_size) -> (N, Q)
        context = self.attention_context(query, key, value, seq_lens)  # (N, V)
        output.append(label_in)
        while True:
            embedded = self.embedding(label_in)  # (1, 1, E)
            char_input = embedded.view(N, -1)   # (1, E)
            logit, context, hidden, cell = self.attention_and_spell(key, value, seq_lens, context, char_input, hidden, cell)
            label_in = self.sample_output(logit)  # (1, 1)
            output.append(label_in)
            if label_in.data[0][0] == EOS_IDX:
                break
        return torch.cat(output, dim=1)[0]

class LASModel(nn.Module):
    dump_patches = True
    def __init__(self, feed_forward_ratio=.1, projection_bias=None):
        super().__init__()
        self.listener = Listener()
        self.speller = Speller(feed_forward_ratio=feed_forward_ratio)
        self.apply(initializer)
        if projection_bias is not None:
            self.speller.apply_projection_bias(projection_bias)

    def forward(self, seqs, seq_lens, label_in, predict=False):
        h, h_len = self.listener(seqs, seq_lens)
        if predict:
            output = self.speller.predict(h, h_len, label_in)
        else:
            output = self.speller(h, h_len, label_in)
        return output

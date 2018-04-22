import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable


def to_variable(array):
    return Variable(torch.from_numpy(array).contiguous())

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def collate_fn(samples):
    batch_size = len(samples)
    seqs = [seq for seq, _ in samples]
    labels = [label for _, label in samples]
    channel = seqs[0].shape[1]
    seq_lengths = torch.IntTensor(list(map(lambda x: x.shape[0], seqs)))

    # because transcript contains SOS and EOS already
    label_lengths = np.array([len(x) for x in labels]) - 1
    max_seq_len = seq_lengths.max()
    max_label_len = label_lengths.max()

    # allocate spaces
    seq_padded = np.zeros((batch_size, max_seq_len, channel), dtype=float)  # (n, max_len, channel)
    label_in_padded = np.zeros((batch_size, max_label_len), dtype=int)
    label_out_padded = np.zeros((batch_size, max_label_len), dtype=int)
    label_mask = np.zeros((batch_size, max_label_len), dtype=float)

    for i, (seq, seq_len, label, label_len) in enumerate(zip(seqs, seq_lengths, labels, label_lengths)):
        seq_padded[i, :seq_len, :] = seq
        label_in_padded[i, :label_len] = label[: -1]
        label_out_padded[i, :label_len] = label[1:]
        label_mask[i, :label_len] = 1

    # sort tensors by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_padded = to_variable(seq_padded[perm_idx]).float().transpose(0, 1).contiguous()
    label_in_padded = to_variable(label_in_padded[perm_idx]).long()
    label_out_padded = to_variable(label_out_padded[perm_idx]).long()
    label_mask = to_variable(label_mask[perm_idx]).float()
    label_lengths = label_lengths[perm_idx]

    # seq_padded: (max_seq_len, n, channel)
    # seq_lengths: (n,)
    # lable_length: (n,)
    # label_in_padded, label_out_padded, label_mask: (n, max_label_len)
    return (seq_padded, seq_lengths.numpy(),
           label_in_padded, label_out_padded, label_mask, label_lengths)

class TestLoader():
    def __init__(self, x):
        self.x = x
        self.len = len(x)
        self.channel = x[0].shape[1]

    def __iter__(self):
        for i in range(self.len):
            seq = self.x[i]
            seq_len = seq.shape[0]
            seq = to_variable(seq[None,:]).float().transpose(0, 1).contiguous()
            seq_len = np.array([seq_len])
            # seq_padded: (max_seq_len, n, channel)
            # seq_lengths: (n,)
            yield (seq, seq_len)

    def __len__(self):
        return self.len

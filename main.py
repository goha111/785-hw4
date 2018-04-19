from model import *
from char_list import *

import argparse
import numpy as np
import torch

def load_data(name):
    return (np.load('{}.npy'.format(name)),
            np.load('{}_encode.npy'.format(name)))

def unigram_logits(stat, smoothing=.01):
    assert(type(stat) == np.ndarray)
    char_size = len(stat)
    char_sum = stat[:, 1].sum()
    p = stat[:, 1] / char_sum
    p_smoothed = p * (1 - smoothing) + smoothing / char_size
    return np.log(p_smoothed)

def main(args):
    xvalid, yvalid = load_data('dev')
    valid_loader = DataLoader(xvalid, yvalid, batch_size=args.batch_size)
    for i, (seq, seq_len, label_in, label_out, label_len) in enumerate(valid_loader):
        print(i)

if __name__ == '__main__':
    CUDA_AVAILABLE = torch.cuda.is_available()
    print('Using CUDA: {}'.format(CUDA_AVAILABLE))

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()
    main(args)